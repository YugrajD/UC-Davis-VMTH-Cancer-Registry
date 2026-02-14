import argparse
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
print("[Warning]: `pyarrow` will become a required dependency of pandas 3.0.")

import pandas as pd
from pathlib import Path
from typing import Union


BASE_DIR = Path(__file__).resolve().parent.parent  # ml/
DEFAULT_CSV = BASE_DIR / "output" / "dataSarcoma" / "petbert_scan_provenance.csv"


def read_csv(filepath: Union[str, Path]) -> pd.DataFrame:
    """Reads a CSV file and returns it as a DataFrame."""
    return pd.read_csv(filepath)


def evaluate_predictions(df: pd.DataFrame, target: str = "sarcoma") -> dict:
    """Evaluates how many original entries have at least one sub-diagnosis
    whose predicted_category contains the target keyword.

    With multi-diagnosis splitting, a single clinical entry (identified by
    ``row_index``) may produce multiple output rows.  An entry counts as a
    match if **any** of its sub-diagnosis predictions contain the keyword.

    Returns a dict with valid_entries, matches, and success_rate.
    """
    # If the output was produced after multi-diagnosis splitting it will
    # have a ``row_index`` column that groups sub-diagnoses back to the
    # original CSV row.  Fall back to treating each output row as its own
    # entry when the column is absent (legacy output format).
    group_col = "row_index" if "row_index" in df.columns else None

    valid = df["diagnosis_text"].notna() & (df["diagnosis_text"].str.strip() != "")
    valid_df = df[valid].copy()

    valid_df["_hit"] = valid_df["predicted_category"].str.contains(target, case=False, na=False)

    if group_col is not None:
        # Group by original entry -- a pass if any sub-diagnosis matches.
        grouped = valid_df.groupby(group_col)["_hit"].any()
        num_valid = len(grouped)
        num_matches = int(grouped.sum())
    else:
        num_valid = len(valid_df)
        num_matches = int(valid_df["_hit"].sum())

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
