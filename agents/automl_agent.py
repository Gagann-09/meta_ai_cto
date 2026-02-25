from __future__ import annotations

import logging
import os
import sys
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from autogluon.tabular import TabularPredictor
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    mean_squared_error,
    r2_score,
    classification_report,
)
from sklearn.model_selection import train_test_split

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

_BINARY_THRESHOLD: int = 2
_MULTICLASS_MAX: int = 50
_DEFAULT_MODELS_DIR: str = "models"
_DEFAULT_PRESETS: str = "medium_quality_faster_train"
_DEFAULT_TEST_SIZE: float = 0.2
_DEFAULT_RANDOM_STATE: int = 42
_MLFLOW_EXPERIMENT: str = "AutoML_Agent"


class AutoMLAgent:

    def __init__(
        self,
        df: pd.DataFrame,
        label_col: str,
        models_dir: Union[str, Path] = _DEFAULT_MODELS_DIR,
        presets: str = _DEFAULT_PRESETS,
        test_size: float = _DEFAULT_TEST_SIZE,
        time_limit: Optional[int] = None,
        random_state: int = _DEFAULT_RANDOM_STATE,
    ) -> None:
        self.df: pd.DataFrame = df.copy()
        self.label_col: str = label_col
        self.models_dir: Path = Path(models_dir)
        self.presets: str = presets
        self.test_size: float = test_size
        self.time_limit: Optional[int] = time_limit
        self.random_state: int = random_state

        self.problem_type: Optional[str] = None
        self.predictor: Optional[TabularPredictor] = None
        self.metrics: Dict[str, Any] = {}
        self.train_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None

    def run(self) -> TabularPredictor:
        self._validate_inputs()
        self._detect_problem_type()
        self._handle_imbalance()
        self._split_data()
        self._setup_mlflow()
        with mlflow.start_run(run_name=f"autogluon_{self.problem_type}"):
            self._train()
            self._evaluate()
            self._log_to_mlflow()
            self._save_model()
            
        self._print_summary()

        logger.info("✅  AutoMLAgent pipeline complete.")
        return self.predictor  # type: ignore[return-value]

    def _validate_inputs(self) -> None:
        if self.label_col not in self.df.columns:
            raise ValueError(
                f"Label column '{self.label_col}' not found in DataFrame. "
                f"Available columns: {list(self.df.columns)}"
            )

        null_count = int(self.df[self.label_col].isnull().sum())
        if null_count > 0:
            logger.warning(
                "⚠️   Label column '%s' has %d null values — dropping those rows.",
                self.label_col, null_count,
            )
            self.df = self.df.dropna(subset=[self.label_col]).reset_index(drop=True)

        if len(self.df) < 10:
            raise ValueError(
                f"Dataset too small after cleaning ({len(self.df)} rows). "
                "Need at least 10 rows to train."
            )
        logger.info("✔️   Inputs validated — %s rows, label='%s'.", f"{len(self.df):,}", self.label_col)

    def _detect_problem_type(self) -> None:
        series = self.df[self.label_col]
        n_unique = series.nunique(dropna=True)

        if pd.api.types.is_float_dtype(series) and n_unique > _MULTICLASS_MAX:
            self.problem_type = "regression"
        elif n_unique == _BINARY_THRESHOLD:
            self.problem_type = "binary"
        elif n_unique > _BINARY_THRESHOLD:
            self.problem_type = "multiclass"
        else:
            self.problem_type = "binary"

        logger.info("🔍  Detected problem type: %s (unique=%d)", self.problem_type.upper(), n_unique)

    def _handle_imbalance(self) -> None:
        if self.problem_type == "regression":
            logger.info("📈  Regression task — imbalance handling not applicable.")
            return

        dist = self.df[self.label_col].value_counts()
        majority = dist.max()
        minority = dist.min()
        ratio = majority / minority if minority > 0 else float("inf")

        logger.info("⚖️   Class ratio (majority/minority): %.2f", ratio)

        if ratio > 3.0:
            logger.info("⚠️   Imbalanced dataset detected (ratio > 3.0) — applying sample_weight balancing.")
            self._is_imbalanced = True
        else:
            logger.info("🟢  Dataset is reasonably balanced.")
            self._is_imbalanced = False

    def _split_data(self) -> None:
        if self.problem_type in ("binary", "multiclass"):
            self.train_df, self.test_df = train_test_split(
                self.df,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=self.df[self.label_col],
            )
        else:
            self.train_df, self.test_df = train_test_split(
                self.df,
                test_size=self.test_size,
                random_state=self.random_state,
            )

        self.train_df = self.train_df.reset_index(drop=True)
        self.test_df = self.test_df.reset_index(drop=True)

        logger.info(
            "📊  Train/Test split: %s / %s rows (%.0f%% test)",
            f"{len(self.train_df):,}", f"{len(self.test_df):,}", self.test_size * 100,
        )

    def _setup_mlflow(self) -> None:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment(_MLFLOW_EXPERIMENT)
        logger.info("📋  MLflow experiment: '%s'", _MLFLOW_EXPERIMENT)

    def _get_sample_weight(self, train_data: pd.DataFrame) -> Optional[pd.Series]:
        if not getattr(self, "_is_imbalanced", False):
            return None
        if self.problem_type == "regression":
            return None

        class_counts = train_data[self.label_col].value_counts()
        total = len(train_data)
        n_classes = len(class_counts)
        weight_map = {
            cls: total / (n_classes * count) for cls, count in class_counts.items()
        }
        weights = train_data[self.label_col].map(weight_map)
        logger.info("⚖️   Sample weights applied — weight map: %s", weight_map)
        return weights

    def _train(self) -> None:
        assert self.train_df is not None

        ag_save_path = str(self.models_dir / "autogluon_model")

        if Path(ag_save_path).exists():
            logger.info("🗑️   Removing previous model directory: %s", ag_save_path)
            shutil.rmtree(ag_save_path)

        sample_weight_col: Optional[str] = None
        if getattr(self, "_is_imbalanced", False) and self.problem_type != "regression":
            weights = self._get_sample_weight(self.train_df)
            if weights is not None:
                self.train_df = self.train_df.copy()
                self.train_df["__sample_weight__"] = weights
                sample_weight_col = "__sample_weight__"

        eval_metric = self._pick_eval_metric()

        logger.info("🚀  Training AutoGluon TabularPredictor …")
        logger.info("    Presets        : %s", self.presets)
        logger.info("    Problem type   : %s", self.problem_type)
        logger.info("    Eval metric    : %s", eval_metric)
        logger.info("    Time limit     : %s", self.time_limit or "auto")
        logger.info("    Device         : CPU only")

        fit_kwargs: Dict[str, Any] = {
            "train_data": self.train_df,
            "presets": self.presets,
            "verbosity": 2,
        }
        if self.time_limit is not None:
            fit_kwargs["time_limit"] = self.time_limit

        self.predictor = TabularPredictor(
            label=self.label_col,
            problem_type=self.problem_type,
            eval_metric=eval_metric,
            path=ag_save_path,
            sample_weight=sample_weight_col,
        )
        self.predictor.fit(**fit_kwargs)
        logger.info("✅  Training complete.")

    def _pick_eval_metric(self) -> str:
        if self.problem_type == "binary":
            return "roc_auc"
        elif self.problem_type == "multiclass":
            return "accuracy"
        else:
            return "root_mean_squared_error"

    def _evaluate(self) -> None:
        assert self.predictor is not None and self.test_df is not None

        y_true = self.test_df[self.label_col]
        y_pred = self.predictor.predict(self.test_df.drop(columns=[self.label_col]))

        if self.problem_type in ("binary", "multiclass"):
            acc = accuracy_score(y_true, y_pred)
            self.metrics["accuracy"] = round(float(acc), 4)
            logger.info("📊  Accuracy : %.4f", acc)

            try:
                y_prob = self.predictor.predict_proba(
                    self.test_df.drop(columns=[self.label_col])
                )
                if self.problem_type == "binary":
                    if hasattr(y_prob, "iloc"):
                        pos_col = y_prob.columns[-1]
                        auc = roc_auc_score(y_true, y_prob[pos_col])
                    else:
                        auc = roc_auc_score(y_true, y_prob)
                else:
                    auc = roc_auc_score(
                        y_true, y_prob, multi_class="ovr", average="weighted"
                    )
                self.metrics["roc_auc"] = round(float(auc), 4)
                logger.info("📊  ROC AUC  : %.4f", auc)
            except Exception as e:
                logger.warning("⚠️   Could not compute ROC AUC: %s", e)
                self.metrics["roc_auc"] = None

            report = classification_report(y_true, y_pred, output_dict=False)
            logger.info("📋  Classification Report:\n%s", report)
        else:
            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            r2 = float(r2_score(y_true, y_pred))
            self.metrics["rmse"] = round(rmse, 4)
            self.metrics["r2"] = round(r2, 4)
            logger.info("📊  RMSE : %.4f", rmse)
            logger.info("📊  R²   : %.4f", r2)

        leaderboard = self.predictor.leaderboard(self.test_df, silent=True)
        self.metrics["leaderboard"] = leaderboard
        logger.info("🏆  Model leaderboard:\n%s", leaderboard.to_string())

    def _log_to_mlflow(self) -> None:
        assert self.predictor is not None

        mlflow.log_param("problem_type", self.problem_type)
        mlflow.log_param("presets", self.presets)
        mlflow.log_param("label_col", self.label_col)
        mlflow.log_param("train_rows", len(self.train_df) if self.train_df is not None else 0)
        mlflow.log_param("test_rows", len(self.test_df) if self.test_df is not None else 0)
        mlflow.log_param("n_features", len(self.df.columns) - 1)
        mlflow.log_param("is_imbalanced", getattr(self, "_is_imbalanced", False))

        ag_metrics = self.predictor.evaluate(self.test_df)
        for metric_name, metric_val in ag_metrics.items():
            mlflow.log_metric(metric_name, float(metric_val))

        for metric_name, metric_val in self.metrics.items():
            if metric_name == "leaderboard":
                continue
            if metric_name not in ag_metrics and metric_val is not None:
                mlflow.log_metric(metric_name, float(metric_val))

        if "leaderboard" in self.metrics:
            lb_path = self.models_dir / "leaderboard.csv"
            self.metrics["leaderboard"].to_csv(lb_path, index=False)
            mlflow.log_artifact(str(lb_path))

        mlflow.log_param("model_path", str(self.models_dir / "autogluon_model"))

        active_run = mlflow.active_run()
        if active_run:
            logger.info("📋  MLflow run logged — ID: %s", active_run.info.run_id)

    def _save_model(self) -> None:
        assert self.predictor is not None
        self.models_dir.mkdir(parents=True, exist_ok=True)
        model_path = self.models_dir / "autogluon_model"
        logger.info("💾  Model saved at: %s", model_path.resolve())

    def _print_summary(self) -> None:
        print(f"\n{'═' * 60}")
        print(f"  🤖  AutoML Agent — Training Summary")
        print(f"{'═' * 60}")
        print(f"    Problem Type   : {self.problem_type}")
        print(f"    Label Column   : {self.label_col}")
        print(f"    Train Rows     : {len(self.train_df):,}" if self.train_df is not None else "")
        print(f"    Test Rows      : {len(self.test_df):,}" if self.test_df is not None else "")
        print(f"    Presets        : {self.presets}")
        print(f"{'─' * 60}")
        for k, v in self.metrics.items():
            if k == "leaderboard":
                continue
            print(f"    {k:<15s} : {v}")
        print(f"{'═' * 60}\n")

    def __repr__(self) -> str:
        status = "trained" if self.predictor is not None else "not trained"
        return (
            f"AutoMLAgent(label='{self.label_col}', "
            f"problem_type='{self.problem_type}', status={status})"
        )


def train_model(
    df: pd.DataFrame,
    label_col: str,
    models_dir: Union[str, Path] = _DEFAULT_MODELS_DIR,
    presets: str = _DEFAULT_PRESETS,
    test_size: float = _DEFAULT_TEST_SIZE,
    time_limit: Optional[int] = None,
) -> TabularPredictor:

    agent = AutoMLAgent(
        df=df,
        label_col=label_col,
        models_dir=models_dir,
        presets=presets,
        test_size=test_size,
        time_limit=time_limit,
    )
    return agent.run()


if __name__ == "__main__":
    import argparse
    from agents.data_agent import load_and_clean

    parser = argparse.ArgumentParser(
        description="AutoMLAgent — train an AutoGluon model on a CSV dataset."
    )
    parser.add_argument("csv_path", type=str, help="Path to the CSV file.")
    parser.add_argument(
        "--label", type=str, default=None,
        help="Target / label column name (auto-detected if omitted).",
    )
    parser.add_argument(
        "--presets", type=str, default=_DEFAULT_PRESETS,
        help=f"AutoGluon presets (default: {_DEFAULT_PRESETS}).",
    )
    parser.add_argument(
        "--time-limit", type=int, default=None,
        help="Max training time in seconds (default: unlimited).",
    )
    parser.add_argument(
        "--models-dir", type=str, default=_DEFAULT_MODELS_DIR,
        help="Directory to save trained models.",
    )
    args = parser.parse_args()

    df = load_and_clean(filepath=args.csv_path, label_col=args.label)

    predictor = train_model(
        df=df,
        label_col=args.label or df.columns[-1],
        models_dir=args.models_dir,
        presets=args.presets,
        time_limit=args.time_limit,
    )
