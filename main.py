import argparse
import sys
from pathlib import Path

from core.orchestrator import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="End-to-End Meta AI CTO Agent")
    parser.add_argument("csv_path", type=str, help="Path to the dataset CSV file")
    parser.add_argument("label", type=str, help="Name of the target / label column")
    parser.add_argument(
        "--time-limit",
        type=int,
        default=60,
        help="Max time (seconds) for AutoML training default is 60s for testing",
    )
    parser.add_argument(
        "--hpo",
        action="store_true",
        default=False,
        help="Run HPO sweep via Ray Tune after baseline training",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=5,
        help="Number of HPO trials (only used with --hpo)",
    )
    args = parser.parse_args()

    result = run_pipeline(
        csv_path=args.csv_path,
        label_col=args.label,
        time_limit=args.time_limit,
        use_hpo=args.hpo,
        max_trials=args.max_trials,
    )

    print("\n✅ End-to-end pipeline completed successfully!")


if __name__ == "__main__":
    main()
