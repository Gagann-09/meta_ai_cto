from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

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

_COMMON_LABEL_NAMES: list[str] = [
    "target",
    "label",
    "class",
    "y",
    "outcome",
    "output",
    "result",
    "default",
    "survived",
    "income",
    "diagnosis",
    "species",
    "is_fraud",
    "fraud",
]

_CLASSIFICATION_CARDINALITY_THRESHOLD: int = 25

class DataAgent:

    def __init__(
        self,
        filepath: Union[str, Path],
        label_col: Optional[str] = None,
        fill_strategy: str = "auto",
        verbose: bool = True,
    ) -> None:
        self.filepath: Path = Path(filepath)
        self._requested_label: Optional[str] = label_col
        self.fill_strategy: str = fill_strategy
        self.verbose: bool = verbose

        # Populated after .run()
        self.df: Optional[pd.DataFrame] = None
        self.label_col: Optional[str] = None
        self.is_classification: bool = False

    def run(self) -> pd.DataFrame:

        self._validate_path()
        self._load_csv()
        self._clean_column_names()
        self._resolve_label_column()
        self._handle_missing_values()
        self._report_shape()
        self._report_class_distribution()

        logger.info("✅  DataAgent pipeline complete — returning cleaned DataFrame.")
        return self.df  # type: ignore[return-value]

    def _validate_path(self) -> None:
        if not self.filepath.exists():
            raise FileNotFoundError(
                f"Dataset not found at '{self.filepath.resolve()}'. "
                "Please check the path and try again."
            )
        if self.filepath.suffix.lower() not in (".csv",):
            logger.warning(
                "File extension is '%s' — expected '.csv'.  "
                "Proceeding anyway, but results may be unexpected.",
                self.filepath.suffix,
            )
        logger.info("📂  File validated: %s", self.filepath.resolve())

    def _load_csv(self) -> None:
        logger.info("📥  Loading CSV …")
        self.df = pd.read_csv(self.filepath, engine="python", on_bad_lines="warn")
        logger.info(
            "    Loaded %s rows × %s columns.", f"{len(self.df):,}", len(self.df.columns)
        )

    def _clean_column_names(self) -> None:
        assert self.df is not None
        original_names = list(self.df.columns)
        self.df.columns = self.df.columns.str.strip()

        renamed = {
            old: new
            for old, new in zip(original_names, self.df.columns)
            if old != new
        }
        if renamed:
            logger.info("✂️   Stripped whitespace from columns: %s", renamed)
        else:
            logger.debug("    Column names are already clean.")

    def _resolve_label_column(self) -> None:

        assert self.df is not None
        columns_lower = {c.lower(): c for c in self.df.columns}

        if self._requested_label is not None:
            if self._requested_label not in self.df.columns:
                # Try case-insensitive match
                match = columns_lower.get(self._requested_label.lower())
                if match is None:
                    raise ValueError(
                        f"Supplied label_col '{self._requested_label}' not found in "
                        f"columns: {list(self.df.columns)}"
                    )
                self.label_col = match
            else:
                self.label_col = self._requested_label
            logger.info("🏷️   Label column (user-supplied): '%s'", self.label_col)
        else:
            # Auto-detect
            for candidate in _COMMON_LABEL_NAMES:
                if candidate in columns_lower:
                    self.label_col = columns_lower[candidate]
                    logger.info(
                        "🏷️   Label column (auto-detected): '%s'", self.label_col
                    )
                    break

            if self.label_col is None:
                self.label_col = self.df.columns[-1]
                logger.info(
                    "🏷️   Label column (fallback to last column): '%s'",
                    self.label_col,
                )

        # Determine classification vs regression
        self.is_classification = self._infer_task_type(self.label_col)
        task = "classification" if self.is_classification else "regression"
        logger.info("📊  Inferred task type: %s", task.upper())

    def _infer_task_type(self, col: str) -> bool:

        assert self.df is not None
        series = self.df[col]

        if series.dtype == object or pd.api.types.is_categorical_dtype(series):
            return True
        if pd.api.types.is_bool_dtype(series):
            return True
        if series.nunique(dropna=True) <= _CLASSIFICATION_CARDINALITY_THRESHOLD:
            return True
        return False

    def _handle_missing_values(self) -> None:

        assert self.df is not None

        total_missing: int = int(self.df.isnull().sum().sum())
        if total_missing == 0:
            logger.info("🟢  No missing values detected.")
            return

        logger.info(
            "⚠️   Found %s total missing values across %s column(s).",
            f"{total_missing:,}",
            int((self.df.isnull().sum() > 0).sum()),
        )

        # Per-column diagnostics
        missing_per_col: pd.Series = self.df.isnull().sum()
        for col_name in missing_per_col[missing_per_col > 0].index:
            count = int(missing_per_col[col_name])
            pct = count / len(self.df) * 100
            logger.debug(
                "    ├─ %-30s  %6d missing  (%5.1f%%)", col_name, count, pct
            )

        strategy = self.fill_strategy.lower()

        if strategy in ("auto", "median"):
            self._fill_auto()
        elif strategy == "zero":
            self._fill_zero()
        elif strategy == "mode":
            self._fill_mode()
        else:
            logger.warning(
                "Unknown fill_strategy '%s' — falling back to 'auto'.", strategy
            )
            self._fill_auto()

        remaining = int(self.df.isnull().sum().sum())
        logger.info("    Missing values remaining after fill: %s", remaining)

    def _fill_auto(self) -> None:
        """Numeric → median, categorical → mode."""
        assert self.df is not None
        for col in self.df.columns:
            if self.df[col].isnull().any():
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    fill_val = self.df[col].median()
                    self.df[col] = self.df[col].fillna(fill_val)
                else:
                    mode_vals = self.df[col].mode()
                    fill_val = mode_vals.iloc[0] if not mode_vals.empty else "unknown"
                    self.df[col] = self.df[col].fillna(fill_val)

    def _fill_zero(self) -> None:
        """Numeric → 0, categorical → 'unknown'."""
        assert self.df is not None
        for col in self.df.columns:
            if self.df[col].isnull().any():
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    self.df[col] = self.df[col].fillna(0)
                else:
                    self.df[col] = self.df[col].fillna("unknown")

    def _fill_mode(self) -> None:
        """All columns → most frequent value."""
        assert self.df is not None
        for col in self.df.columns:
            if self.df[col].isnull().any():
                mode_vals = self.df[col].mode()
                fill_val = mode_vals.iloc[0] if not mode_vals.empty else 0
                self.df[col] = self.df[col].fillna(fill_val)

    def _report_shape(self) -> None:
        """Print the dataset dimensions."""
        assert self.df is not None
        rows, cols = self.df.shape
        print(f"\n{'═' * 50}")
        print(f"  📐  Dataset Shape : {rows:,} rows × {cols} columns")
        print(f"{'═' * 50}")

    def _report_class_distribution(self) -> None:
        """Print value counts for the label column when classification."""
        assert self.df is not None and self.label_col is not None

        if not self.is_classification:
            logger.info(
                "📈  Task is regression — skipping class distribution report."
            )
            return

        dist: pd.Series = self.df[self.label_col].value_counts()
        total = len(self.df)

        print(f"\n{'─' * 50}")
        print(f"  🏷️  Class Distribution — '{self.label_col}'")
        print(f"{'─' * 50}")
        for cls_label, count in dist.items():
            pct = count / total * 100
            bar = "█" * int(pct // 2)
            print(f"    {str(cls_label):>20s}  │ {count:>8,}  ({pct:5.1f}%)  {bar}")
        print(f"{'─' * 50}\n")

    def __repr__(self) -> str:
        status = "loaded" if self.df is not None else "not loaded"
        return (
            f"DataAgent(filepath='{self.filepath}', "
            f"label_col='{self.label_col}', status={status})"
        )

def load_and_clean(
    filepath: Union[str, Path],
    label_col: Optional[str] = None,
    fill_strategy: str = "auto",
    verbose: bool = True,
) -> pd.DataFrame:

    agent = DataAgent(
        filepath=filepath,
        label_col=label_col,
        fill_strategy=fill_strategy,
        verbose=verbose,
    )
    return agent.run()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="DataAgent — load, validate & clean a CSV dataset."
    )
    parser.add_argument("csv_path", type=str, help="Path to the CSV file.")
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Name of the target / label column (auto-detected if omitted).",
    )
    parser.add_argument(
        "--fill",
        type=str,
        default="auto",
        choices=["auto", "median", "mode", "zero"],
        help="Missing-value fill strategy (default: auto).",
    )
    args = parser.parse_args()

    df = load_and_clean(
        filepath=args.csv_path,
        label_col=args.label,
        fill_strategy=args.fill,
    )
    print(df.head())
