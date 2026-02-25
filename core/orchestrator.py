"""core.orchestrator — single entry-point for the end-to-end ML pipeline.

Coordinates data ingestion, optional HPO, and AutoML training without
modifying the underlying agent modules.
"""

from __future__ import annotations

import importlib
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

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

_SEPARATOR = "═" * 60
_THIN_SEP  = "─" * 60


# ── Pipeline ─────────────────────────────────────────────────────────────────

def run_pipeline(
    data_path: str,
    label: str,
    use_hpo: bool = False,
    *,
    presets: str = "medium_quality_faster_train",
    time_limit: Optional[int] = None,
    models_dir: str = "models",
) -> Dict[str, Any]:
    """Run the full ML pipeline and return final evaluation metrics.

    Parameters
    ----------
    data_path : str
        Path to a CSV dataset.
    label : str
        Name of the target / label column.
    use_hpo : bool, optional
        If *True*, delegate training to ``agents.hpo_agent`` (must exist).
        Defaults to *False* → uses ``agents.automl_agent``.
    presets : str, optional
        AutoGluon presets string (only used when ``use_hpo=False``).
    time_limit : int | None, optional
        Maximum training time in seconds.
    models_dir : str, optional
        Directory to persist trained artefacts.

    Returns
    -------
    Dict[str, Any]
        Dictionary of evaluation metric names → float values.
    """

    _log_header("Pipeline started")
    t_start: float = time.perf_counter()

    # ── 1. Data ingestion ────────────────────────────────────────────────
    logger.info("📂  Stage 1 / 3 — Loading & cleaning data")
    from agents.data_agent import load_and_clean  # noqa: WPS433

    df: pd.DataFrame = load_and_clean(filepath=data_path, label_col=label)
    logger.info(
        "✔️   Data ready — %s rows × %s columns",
        f"{len(df):,}",
        len(df.columns),
    )

    # ── 2. Training ──────────────────────────────────────────────────────
    if use_hpo:
        logger.info("🔬  Stage 2 / 3 — Training via HPO agent")
        predictor = _run_hpo_agent(df, label, time_limit=time_limit, models_dir=models_dir)
    else:
        logger.info("🚀  Stage 2 / 3 — Training via AutoML agent")
        predictor = _run_automl_agent(
            df, label, presets=presets, time_limit=time_limit, models_dir=models_dir,
        )

    # ── 3. Final evaluation ──────────────────────────────────────────────
    logger.info("📊  Stage 3 / 3 — Final evaluation")
    metrics: Dict[str, Any] = _collect_metrics(predictor, df, label)

    elapsed: float = round(time.perf_counter() - t_start, 2)
    _log_summary(metrics, elapsed)

    return metrics


# ── Private helpers ──────────────────────────────────────────────────────────

def _run_automl_agent(
    df: pd.DataFrame,
    label: str,
    *,
    presets: str,
    time_limit: Optional[int],
    models_dir: str,
) -> Any:
    """Delegate to *agents.automl_agent.train_model*."""
    from agents.automl_agent import train_model  # noqa: WPS433

    predictor = train_model(
        df=df,
        label_col=label,
        presets=presets,
        time_limit=time_limit,
        models_dir=models_dir,
    )
    return predictor


def _run_hpo_agent(
    df: pd.DataFrame,
    label: str,
    *,
    time_limit: Optional[int],
    models_dir: str,
) -> Any:
    """Dynamically import and run *agents.hpo_agent* if available."""
    try:
        hpo_module = importlib.import_module("agents.hpo_agent")
    except ModuleNotFoundError:
        logger.error(
            "❌  'agents.hpo_agent' not found.  "
            "Falling back to the default AutoML agent."
        )
        return _run_automl_agent(
            df, label, presets="medium_quality_faster_train",
            time_limit=time_limit, models_dir=models_dir,
        )

    if not hasattr(hpo_module, "train_model"):
        raise AttributeError(
            "agents.hpo_agent exists but does not expose a 'train_model' callable."
        )

    logger.info("🔬  Dispatching to hpo_agent.train_model …")
    predictor = hpo_module.train_model(
        df=df,
        label_col=label,
        time_limit=time_limit,
        models_dir=models_dir,
    )
    return predictor


def _collect_metrics(predictor: Any, df: pd.DataFrame, label: str) -> Dict[str, Any]:
    """Call ``predictor.evaluate`` and guarantee all values are floats."""
    raw: Dict[str, Any] = predictor.evaluate(df)

    metrics: Dict[str, Any] = {}
    for key, value in raw.items():
        try:
            metrics[key] = float(value)
        except (TypeError, ValueError):
            logger.warning("⚠️   Metric '%s' could not be cast to float — skipped.", key)

    return metrics


# ── Pretty-print helpers ─────────────────────────────────────────────────────

def _log_header(title: str) -> None:
    print(f"\n{_SEPARATOR}")
    print(f"  🤖  {title}")
    print(_SEPARATOR)


def _log_summary(metrics: Dict[str, Any], elapsed: float) -> None:
    print(f"\n{_SEPARATOR}")
    print("  📋  Pipeline Results")
    print(_THIN_SEP)
    for name, value in metrics.items():
        print(f"    {name:<25s} : {value}")
    print(_THIN_SEP)
    print(f"    {'elapsed (s)':<25s} : {elapsed}")
    print(f"{_SEPARATOR}\n")
    logger.info("✅  Pipeline finished in %.2fs", elapsed)
