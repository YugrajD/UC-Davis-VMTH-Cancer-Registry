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


def evaluate_by_visit(df: pd.DataFrame, target: str = "sarcoma") -> dict:
    """Evaluates how many visits have at least one sub-diagnosis whose
    predicted_category contains the target keyword.

    With multi-diagnosis splitting, a single visit (identified by
    ``row_index``) may produce multiple output rows.  A visit counts as a
    match if **any** of its sub-diagnosis predictions contain the keyword.

    Returns a dict with valid_visits, matches, and success_rate.
    """
    group_col = "row_index" if "row_index" in df.columns else None

    valid = df["diagnosis_text"].notna() & (df["diagnosis_text"].str.strip() != "")
    valid_df = df[valid].copy()

    valid_df["_hit"] = valid_df["predicted_category"].str.contains(target, case=False, na=False)

    if group_col is not None:
        grouped = valid_df.groupby(group_col)["_hit"].any()
        num_valid = len(grouped)
        num_matches = int(grouped.sum())
    else:
        num_valid = len(valid_df)
        num_matches = int(valid_df["_hit"].sum())

    success_rate = num_matches / num_valid if num_valid > 0 else 0.0

    return {
        "valid_visits": num_valid,
        "matches": num_matches,
        "success_rate": success_rate,
    }


def evaluate_by_patient(
    df: pd.DataFrame,
    target: str = "sarcoma",
    id_col: str = "anon_id",
) -> dict:
    """Evaluates how many unique patients have at least one prediction
    whose predicted_category contains the target keyword.

    Groups all prediction rows sharing the same ``anon_id``.  A patient
    counts as a match if **any** of their predictions contain the keyword,
    even if other predictions for the same patient do not.

    Returns a dict with total_patients, matches, and success_rate.
    """
    if id_col not in df.columns:
        raise KeyError(
            f"Column '{id_col}' not found in CSV. "
            f"Available columns: {list(df.columns)}"
        )

    valid = df["diagnosis_text"].notna() & (df["diagnosis_text"].str.strip() != "")
    valid_df = df[valid].copy()

    valid_df["_hit"] = valid_df["predicted_category"].str.contains(
        target, case=False, na=False
    )

    grouped = valid_df.groupby(id_col)["_hit"].any()
    num_patients = len(grouped)
    num_matches = int(grouped.sum())

    success_rate = num_matches / num_patients if num_patients > 0 else 0.0

    return {
        "total_patients": num_patients,
        "matches": num_matches,
        "success_rate": success_rate,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PetBERT scan predictions.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to the provenance CSV file.")
    parser.add_argument("--keyword", type=str, default="sarcoma", help="Target keyword to match in predicted_category.")
    parser.add_argument("--group-by", choices=["visit", "patient"], default="visit", help="Group by 'visit' (row_index) or 'patient' (anon_id).")
    parser.add_argument("--id-col", type=str, default="anon_id", help="Column name for patient ID (only used in patient mode).")
    return parser.parse_args()


def main():
    args = parse_args()
    df = read_csv(args.csv)

    if args.group_by == "patient":
        results = evaluate_by_patient(df, target=args.keyword, id_col=args.id_col)
        print(f"Total patients: {results['total_patients']}")
        print(f"Matches:        {results['matches']}")
        print(f"Success rate:   {results['success_rate']:.2%}")
    else:
        results = evaluate_by_visit(df, target=args.keyword)
        print(f"Valid visits: {results['valid_visits']}")
        print(f"Matches:       {results['matches']}")
        print(f"Success rate:  {results['success_rate']:.2%}")


if __name__ == "__main__":
    main()
