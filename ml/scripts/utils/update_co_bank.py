"""Append completely-off predictions from evaluation.csv into a rolling CO-negative bank.

Deduplicates on (case_id, predicted_term) so the same wrong prediction
from multiple cycles counts only once. The bank grows monotonically.

Called automatically by run_training_cycle.py after each evaluate step (step 4.5).
The bank is then passed to build_training_pairs.py --co-neg-bank-csv in the next
cycle's step 1, replacing the single-cycle evaluation.csv as the CO source.
"""

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Append CO negatives from evaluation.csv into the rolling bank."
    )
    parser.add_argument(
        "--evaluation-csv",
        default="ml/output/evaluation/evaluation.csv",
        help="Evaluation CSV produced by evaluate_predictions.py (default: ml/output/evaluation/evaluation.csv)",
    )
    parser.add_argument(
        "--bank-csv",
        default="ml/output/evaluation/evaluation_co_bank.csv",
        help="Path to the rolling CO-negative bank (default: ml/output/evaluation/evaluation_co_bank.csv)",
    )
    args = parser.parse_args()

    bank_path = Path(args.bank_csv)
    existing: dict[tuple, dict] = {}

    # Load existing bank — keep only completely_off rows
    if bank_path.exists():
        with open(bank_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("verdict") == "completely_off":
                    key = (row["case_id"], row["predicted_term"])
                    existing[key] = row

    before = len(existing)

    # Append new completely_off rows from current evaluation
    fieldnames: list[str] | None = None
    with open(args.evaluation_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            if row.get("verdict") != "completely_off":
                continue
            key = (row["case_id"], row["predicted_term"])
            if key not in existing:
                existing[key] = row

    added = len(existing) - before

    # Write back
    bank_path.parent.mkdir(parents=True, exist_ok=True)
    with open(bank_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing.values())

    print(f"CO bank updated: +{added} new rows  ({before} → {len(existing)} unique pairs)")
    print(f"  Bank written to {bank_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
