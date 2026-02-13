import argparse
import pandas as pd
from pathlib import Path
from typing import Union


BASE_DIR = Path(__file__).resolve().parent.parent  # ml/
DEFAULT_CSV = BASE_DIR / "output" / "dataSarcoma" / "petbert_scan_categories.csv"


def read_csv(filepath: Union[str, Path]) -> pd.DataFrame:
    """Reads a CSV file and returns it as a DataFrame."""
    return pd.read_csv(filepath)


def evaluate_predictions(df: pd.DataFrame, target: str = "sarcoma") -> dict:
    """Evaluates how many valid entries have a predicted_category containing the target string.

    Returns a dict with valid_entries, matches, and success_rate.
    """
    valid = df["Clinical Diagnoses"].notna() & (df["Clinical Diagnoses"].str.strip() != "")
    valid_df = df[valid]

    matches = valid_df["predicted_category"].str.contains(target, case=False, na=False)
    num_matches = matches.sum()
    num_valid = len(valid_df)
    success_rate = num_matches / num_valid if num_valid > 0 else 0.0

    return {
        "valid_entries": num_valid,
        "matches": num_matches,
        "success_rate": success_rate,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PetBERT scan predictions.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to the predictions CSV file.")
    parser.add_argument("--keyword", type=str, default="sarcoma", help="Target keyword to match in predicted_category.")
    return parser.parse_args()


def main():
    args = parse_args()

    df = read_csv(args.csv)
    results = evaluate_predictions(df, target=args.keyword)

    print(f"Valid entries: {results['valid_entries']}")
    print(f"Matches:       {results['matches']}")
    print(f"Success rate:  {results['success_rate']:.2%}")


if __name__ == "__main__":
    main()
