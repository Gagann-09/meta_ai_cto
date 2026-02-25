"""orchestrator.py — Central pipeline coordinator.

Runs the baseline flow (data cleaning → AutoML training) and optionally
calls the HPO agent for hyperparameter tuning when ``use_hpo=True``.

Usage
-----
    from core.orchestrator import run_pipeline

    # baseline only
    run_pipeline("data.csv", label_col="target")

    # baseline + HPO sweep
    run_pipeline("data.csv", label_col="target", use_hpo=True)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

from agents.data_agent import load_and_clean
from agents.automl_agent import train_model

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


def run_pipeline(
    csv_path: Union[str, Path],
    label_col: str,
    time_limit: int = 60,
    use_hpo: bool = False,
    max_trials: int = 5,
) -> Dict[str, Any]:
    """Execute the end-to-end ML pipeline.

    Parameters
    ----------
    csv_path : str | Path
        Path to the dataset CSV file.
    label_col : str
        Name of the target / label column.
    time_limit : int
        Max seconds for AutoML training (baseline flow).
    use_hpo : bool
        If ``True``, run a Ray Tune HPO sweep **after** the baseline
        training completes.  Defaults to ``False``.
    max_trials : int
        Number of HPO trials (only used when ``use_hpo=True``).

    Returns
    -------
    dict
        ``predictor`` — the trained AutoGluon predictor (always present).
        ``hpo_result`` — best config + accuracy from HPO (only when
        ``use_hpo=True``).
    """

    result: Dict[str, Any] = {}

    # ── Step 1: Data Cleaning ────────────────────────────────────────
    logger.info("── Step 1/2: Data Cleaning ──")
    df = load_and_clean(filepath=csv_path, label_col=label_col)

    # ── Step 2: AutoML Training (baseline) ───────────────────────────
    logger.info("── Step 2/2: AutoML Training ──")
    predictor = train_model(
        df=df,
        label_col=label_col,
        time_limit=time_limit,
    )
    result["predictor"] = predictor

    # ── Optional: HPO Sweep ──────────────────────────────────────────
    if use_hpo:
        logger.info("── Bonus: HPO Sweep (Ray Tune) ──")
        from agents.hpo_agent import tune_hyperparameters

        hpo_result = tune_hyperparameters(
            df=df,
            label_col=label_col,
            max_trials=max_trials,
        )
        result["hpo_result"] = hpo_result

    logger.info("✅  Pipeline complete.")
    return result
