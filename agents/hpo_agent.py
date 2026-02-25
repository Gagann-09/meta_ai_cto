"""agents.hpo_agent — Hyperparameter optimisation via Ray Tune + AutoGluon.

Wraps AutoGluon training inside a Ray Tune *trainable* function, searches
over at least two hyperparameters with an ASHA scheduler, and logs the best
run to MLflow.  Does **not** modify ``automl_agent`` or ``data_agent``.

Public API
----------
train_model(df, label_col, *, time_limit, models_dir) -> TabularPredictor
    Drop-in replacement expected by ``core.orchestrator._run_hpo_agent``.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Union

import mlflow
import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

# ── Logging ──────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s │ %(name)s │ %(levelname)-8s │ %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(_handler)

# ── Constants ────────────────────────────────────────────────────────────────

_DEFAULT_MODELS_DIR: str = "models"
_NUM_TRIALS: int = 5
_TEST_SIZE: float = 0.2
_RANDOM_STATE: int = 42
_MLFLOW_EXPERIMENT: str = "HPO_Agent"

# ── Search space ─────────────────────────────────────────────────────────────
# Two tunable hyperparameters:
#   1. presets     — controls model quality / speed trade-off
#   2. time_limit  — per-trial training budget (seconds)

_SEARCH_SPACE: Dict[str, Any] = {
    "presets": tune.choice([
        "medium_quality",
        "good_quality",
        "high_quality",
        "best_quality",
    ]),
    "time_limit_per_trial": tune.choice([30, 60, 120]),
}


# ── Trainable ────────────────────────────────────────────────────────────────

def _make_trainable(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    label_col: str,
):
    """Return a closure that Ray Tune will call for each trial."""

    def _trainable(config: Dict[str, Any]) -> None:
        """Single Ray Tune trial: train AutoGluon with *config* and report
        validation accuracy back to the scheduler."""

        trial_dir = tempfile.mkdtemp(prefix="hpo_trial_")

        try:
            predictor = TabularPredictor(
                label=label_col,
                path=trial_dir,
                verbosity=0,
            )
            predictor.fit(
                train_data=train_df,
                presets=config["presets"],
                time_limit=config["time_limit_per_trial"],
            )

            y_true = val_df[label_col]
            y_pred = predictor.predict(val_df.drop(columns=[label_col]))
            val_accuracy = float(accuracy_score(y_true, y_pred))

            # Report metric to Ray Tune scheduler
            tune.report({"val_accuracy": val_accuracy})

        finally:
            # Clean up trial artefacts to save disk space
            if Path(trial_dir).exists():
                shutil.rmtree(trial_dir, ignore_errors=True)

    return _trainable


# ── Core logic ───────────────────────────────────────────────────────────────

def _run_hpo(
    df: pd.DataFrame,
    label_col: str,
    overall_time_limit: Optional[int],
    models_dir: str,
) -> TabularPredictor:
    """Execute the Ray Tune HPO loop, retrain with the best config, and
    log everything to MLflow."""

    # ── 1. Split data ────────────────────────────────────────────────────
    train_df, val_df = train_test_split(
        df,
        test_size=_TEST_SIZE,
        random_state=_RANDOM_STATE,
        stratify=df[label_col] if df[label_col].nunique() <= 50 else None,
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    logger.info(
        "📊  HPO split — train %s / val %s rows",
        f"{len(train_df):,}",
        f"{len(val_df):,}",
    )

    # ── 2. Initialise Ray (CPU only) ─────────────────────────────────────
    if not ray.is_initialized():
        ray.init(
            num_cpus=os.cpu_count() or 1,
            num_gpus=0,
            log_to_driver=False,
            ignore_reinit_error=True,
        )

    # ── 3. ASHA scheduler ───────────────────────────────────────────────
    scheduler = ASHAScheduler(
        metric="val_accuracy",
        mode="max",
        max_t=1,            # single-epoch trainable; 1 report per trial
        grace_period=1,
        reduction_factor=2,
    )

    # ── 4. Run trials ────────────────────────────────────────────────────
    logger.info("🔬  Starting Ray Tune — %d trials, CPU only", _NUM_TRIALS)

    trainable = _make_trainable(train_df, val_df, label_col)

    tuner = tune.Tuner(
        tune.with_resources(trainable, resources={"cpu": 1, "gpu": 0}),
        param_space=_SEARCH_SPACE,
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=_NUM_TRIALS,
        ),
        run_config=ray.train.RunConfig(
            name="hpo_autogluon",
            verbose=1,
        ),
    )

    results = tuner.fit()
    best_result = results.get_best_result(metric="val_accuracy", mode="max")
    best_config: Dict[str, Any] = best_result.config
    best_accuracy: float = float(best_result.metrics["val_accuracy"])

    logger.info("🏆  Best trial — accuracy=%.4f, config=%s", best_accuracy, best_config)

    # ── 5. Retrain with best config on FULL data ─────────────────────────
    logger.info("🚀  Retraining final model with best config on full dataset")

    final_model_path = str(Path(models_dir) / "hpo_best_model")
    if Path(final_model_path).exists():
        shutil.rmtree(final_model_path)

    final_predictor = TabularPredictor(
        label=label_col,
        path=final_model_path,
        verbosity=1,
    )

    final_time_limit = best_config.get("time_limit_per_trial", 60)
    if overall_time_limit is not None:
        final_time_limit = min(final_time_limit, overall_time_limit)

    final_predictor.fit(
        train_data=df,
        presets=best_config["presets"],
        time_limit=final_time_limit,
    )

    # ── 6. Log to MLflow ─────────────────────────────────────────────────
    _log_to_mlflow(best_config, best_accuracy, results, final_model_path)

    # ── 7. Shutdown Ray ──────────────────────────────────────────────────
    ray.shutdown()

    logger.info("✅  HPO pipeline complete — model at %s", final_model_path)
    return final_predictor


def _log_to_mlflow(
    best_config: Dict[str, Any],
    best_accuracy: float,
    results: Any,
    model_path: str,
) -> None:
    """Log HPO results under a dedicated MLflow experiment."""

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(_MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="hpo_best_trial"):
        # ── Params
        for key, value in best_config.items():
            mlflow.log_param(key, value)
        mlflow.log_param("num_trials", _NUM_TRIALS)
        mlflow.log_param("model_path", model_path)

        # ── Metrics (guaranteed float)
        mlflow.log_metric("best_val_accuracy", float(best_accuracy))

        # Log per-trial results as a CSV artefact
        try:
            results_df = results.get_dataframe()
            artefact_path = Path(model_path).parent / "hpo_trials.csv"
            results_df.to_csv(artefact_path, index=False)
            mlflow.log_artifact(str(artefact_path))
        except Exception as exc:  # noqa: BLE001
            logger.warning("⚠️   Could not save trial results CSV: %s", exc)

        active_run = mlflow.active_run()
        if active_run:
            logger.info("📋  MLflow HPO run — ID: %s", active_run.info.run_id)


# ── Public API (matches orchestrator contract) ───────────────────────────────

def train_model(
    df: pd.DataFrame,
    label_col: str,
    *,
    time_limit: Optional[int] = None,
    models_dir: Union[str, Path] = _DEFAULT_MODELS_DIR,
) -> TabularPredictor:
    """HPO-powered training entry-point.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset (output of ``data_agent.load_and_clean``).
    label_col : str
        Name of the target column.
    time_limit : int | None
        Overall time budget in seconds (caps per-trial limit).
    models_dir : str | Path
        Directory to persist the final trained model.

    Returns
    -------
    TabularPredictor
        The predictor retrained on the full dataset using the best
        hyperparameter configuration found by Ray Tune.
    """

    logger.info("═" * 60)
    logger.info("  🔬  HPO Agent — Ray Tune + AutoGluon")
    logger.info("═" * 60)

    return _run_hpo(
        df=df,
        label_col=label_col,
        overall_time_limit=time_limit,
        models_dir=str(models_dir),
    )
