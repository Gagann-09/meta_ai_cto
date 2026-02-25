"""hpo_agent.py — Hyperparameter optimisation wrapper around AutoGluon
using Ray Tune with ASHAScheduler.

Key design decisions
--------------------
* Does NOT modify or import internal AutoGluon training logic from
  ``automl_agent.py`` — it instantiates ``TabularPredictor`` directly
  inside the trainable so each trial is fully independent.
* Tunes **two** hyperparameters:
    1. ``presets`` — quality preset passed to AutoGluon
    2. ``time_limit`` — per-model training budget in seconds
* Uses ``ASHAScheduler`` (Async Successive Halving) for aggressive
  early-stopping of bad trials.
* Caps total trials at **5** and forces CPU-only execution.
* The best trial's validation accuracy is logged to **MLflow**.
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
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# ── logging ──────────────────────────────────────────────────────────
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

# ── defaults ─────────────────────────────────────────────────────────
_DEFAULT_TEST_SIZE: float = 0.2
_DEFAULT_RANDOM_STATE: int = 42
_MAX_TRIALS: int = 5
_MLFLOW_EXPERIMENT: str = "HPO_Agent"


# ── Ray Tune trainable ───────────────────────────────────────────────
def _make_trainable(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str,
):
    """Return a closure that Ray Tune will call for every trial.

    Hyperparameters come from ``config`` which Ray injects at runtime.
    """

    def trainable(config: Dict[str, Any]) -> None:
        presets: str = config["presets"]
        time_limit: int = int(config["time_limit"])

        # Each trial gets its own temporary model directory so trials
        # running in parallel never collide.
        trial_dir = tempfile.mkdtemp(prefix="hpo_ag_")

        try:
            predictor = TabularPredictor(
                label=label_col,
                path=trial_dir,
                verbosity=0,  # keep logs quiet during HPO
            )
            predictor.fit(
                train_data=train_df,
                presets=presets,
                time_limit=time_limit,
                verbosity=0,
            )

            y_true = test_df[label_col]
            y_pred = predictor.predict(
                test_df.drop(columns=[label_col])
            )
            acc = float(accuracy_score(y_true, y_pred))

            # Report back to Ray Tune
            tune.report({"accuracy": acc})
        finally:
            # Clean up heavy model artefacts after each trial
            if Path(trial_dir).exists():
                shutil.rmtree(trial_dir, ignore_errors=True)

    return trainable


# ── public API ───────────────────────────────────────────────────────
class HPOAgent:
    """Hyperparameter-tuning agent that wraps AutoGluon inside Ray Tune.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset (features + label).
    label_col : str
        Name of the target column.
    test_size : float
        Fraction held out for validation in each trial.
    max_trials : int
        Maximum number of Ray Tune trials (default 5).
    random_state : int
        Seed for the train/test split.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        label_col: str,
        test_size: float = _DEFAULT_TEST_SIZE,
        max_trials: int = _MAX_TRIALS,
        random_state: int = _DEFAULT_RANDOM_STATE,
    ) -> None:
        self.df = df.copy()
        self.label_col = label_col
        self.test_size = test_size
        self.max_trials = max_trials
        self.random_state = random_state

        self.best_config: Optional[Dict[str, Any]] = None
        self.best_accuracy: Optional[float] = None

    # ── search space ─────────────────────────────────────────────────
    @staticmethod
    def _search_space() -> Dict[str, Any]:
        return {
            "presets": tune.choice([
                "medium_quality",
                "best_quality",
                "good_quality",
            ]),
            "time_limit": tune.choice([30, 60, 120]),
        }

    # ── run ───────────────────────────────────────────────────────────
    def run(self) -> Dict[str, Any]:
        """Execute the HPO sweep and return the best config + accuracy."""

        self._validate()
        train_df, test_df = self._split()

        # Force CPU-only: hide any GPUs from CUDA / Ray
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        scheduler = ASHAScheduler(
            metric="accuracy",
            mode="max",
            max_t=1,            # single-step trials (no iterative loop)
            grace_period=1,
        )

        trainable_fn = _make_trainable(train_df, test_df, self.label_col)

        logger.info("🔎  Starting HPO sweep — max %d trials …", self.max_trials)

        tuner = tune.Tuner(
            trainable_fn,
            param_space=self._search_space(),
            tune_config=tune.TuneConfig(
                scheduler=scheduler,
                num_samples=self.max_trials,
            ),
            run_config=tune.RunConfig(
                name="hpo_autogluon",
                verbose=1,
            ),
        )

        results = tuner.fit()
        best_result = results.get_best_result(metric="accuracy", mode="max")

        self.best_config = best_result.config
        self.best_accuracy = float(best_result.metrics["accuracy"])

        logger.info("🏆  Best accuracy : %.4f", self.best_accuracy)
        logger.info("🏆  Best config   : %s", self.best_config)

        self._log_to_mlflow()
        self._print_summary(results)

        return {
            "best_config": self.best_config,
            "best_accuracy": self.best_accuracy,
        }

    # ── helpers ───────────────────────────────────────────────────────
    def _validate(self) -> None:
        if self.label_col not in self.df.columns:
            raise ValueError(
                f"Label column '{self.label_col}' not found. "
                f"Available: {list(self.df.columns)}"
            )
        if len(self.df) < 10:
            raise ValueError(
                f"Dataset too small ({len(self.df)} rows). Need ≥ 10."
            )
        logger.info(
            "✔️   HPO inputs validated — %s rows, label='%s'.",
            f"{len(self.df):,}", self.label_col,
        )

    def _split(self):
        train_df, test_df = train_test_split(
            self.df,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.df[self.label_col],
        )
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        logger.info(
            "📊  HPO split: %s train / %s test rows.",
            f"{len(train_df):,}", f"{len(test_df):,}",
        )
        return train_df, test_df

    def _log_to_mlflow(self) -> None:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment(_MLFLOW_EXPERIMENT)

        with mlflow.start_run(run_name="hpo_best_trial"):
            mlflow.log_param("best_presets", self.best_config["presets"])
            mlflow.log_param("best_time_limit", self.best_config["time_limit"])
            mlflow.log_param("max_trials", self.max_trials)
            mlflow.log_param("label_col", self.label_col)
            mlflow.log_metric("best_accuracy", float(self.best_accuracy))

            active_run = mlflow.active_run()
            if active_run:
                logger.info(
                    "📋  MLflow HPO run logged — ID: %s",
                    active_run.info.run_id,
                )

    def _print_summary(self, results) -> None:
        print(f"\n{'═' * 60}")
        print(f"  🔬  HPO Agent — Tuning Summary")
        print(f"{'═' * 60}")
        print(f"    Label Column   : {self.label_col}")
        print(f"    Max Trials     : {self.max_trials}")
        print(f"    Best Accuracy  : {self.best_accuracy:.4f}")
        print(f"    Best Presets   : {self.best_config['presets']}")
        print(f"    Best Time Limit: {self.best_config['time_limit']}s")
        print(f"{'─' * 60}")

        # Show per-trial results table
        result_df = results.get_dataframe()
        cols_to_show = [c for c in result_df.columns
                        if c.startswith("config/") or c == "accuracy"]
        if cols_to_show:
            print(result_df[cols_to_show].to_string(index=False))
        print(f"{'═' * 60}\n")

    def __repr__(self) -> str:
        status = "tuned" if self.best_config is not None else "pending"
        return f"HPOAgent(label='{self.label_col}', status={status})"


# ── convenience function ─────────────────────────────────────────────
def tune_hyperparameters(
    df: pd.DataFrame,
    label_col: str,
    max_trials: int = _MAX_TRIALS,
    test_size: float = _DEFAULT_TEST_SIZE,
) -> Dict[str, Any]:
    """One-liner to run an HPO sweep and return the best result."""
    agent = HPOAgent(
        df=df,
        label_col=label_col,
        test_size=test_size,
        max_trials=max_trials,
    )
    return agent.run()


# ── CLI entry point ──────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    from agents.data_agent import load_and_clean

    parser = argparse.ArgumentParser(
        description="HPOAgent — tune AutoGluon hyperparameters via Ray Tune."
    )
    parser.add_argument("csv_path", type=str, help="Path to the CSV file.")
    parser.add_argument(
        "--label", type=str, default=None,
        help="Target / label column name (auto-detected if omitted).",
    )
    parser.add_argument(
        "--max-trials", type=int, default=_MAX_TRIALS,
        help=f"Maximum number of Ray Tune trials (default: {_MAX_TRIALS}).",
    )
    args = parser.parse_args()

    df = load_and_clean(filepath=args.csv_path, label_col=args.label)

    result = tune_hyperparameters(
        df=df,
        label_col=args.label or df.columns[-1],
        max_trials=args.max_trials,
    )
