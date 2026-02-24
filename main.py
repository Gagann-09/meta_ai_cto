import argparse
import sys
from pathlib import Path

from agents.data_agent import load_and_clean
from agents.automl_agent import train_model

def main():
    parser = argparse.ArgumentParser(description="End-to-End Meta AI CTO Agent")
    parser.add_argument("csv_path", type=str, help="Path to the dataset CSV file")
    parser.add_argument("label", type=str, help="Name of the target / label column")
    parser.add_argument(
        "--time-limit", 
        type=int, 
        default=60, 
        help="Max time (seconds) for AutoML training default is 60s for testing"
    )
    args = parser.parse_args()

    # 1. Clean Data
    print(f"\n--- 1. Data Cleaning: {args.csv_path} ---")
    df = load_and_clean(filepath=args.csv_path, label_col=args.label)

    # 2. Train AutoML Model
    print("\n--- 2. AutoML Training ---")
    predictor = train_model(
        df=df,
        label_col=args.label,
        time_limit=args.time_limit
    )

    print("\n✅ End-to-end pipeline completed successfully!")

if __name__ == "__main__":
    main()
