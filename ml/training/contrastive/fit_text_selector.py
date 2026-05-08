"""Fit and save the TF-IDF vectorizers for multi-column text selection.

Run once before building contrastive pairs with the TF-IDF selector:

  ml/.venv/Scripts/python.exe ml/training/contrastive/fit_text_selector.py
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import config
from text_selection import TextSelector, SOURCE_COLS
from utils.csv_io import strip_bom_from_columns


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit per-column TF-IDF vectorizers on the report corpus for text selection."
    )
    parser.add_argument("--reports-csv", default=config.REPORTS_CSV)
    parser.add_argument("--out", default=config.TFIDF_VECTORIZER_PATH)
    args = parser.parse_args()

    print(f"Reading reports from {args.reports_csv}...")
    col_to_texts: dict[str, list[str]] = {col: [] for col in SOURCE_COLS}
    with open(args.reports_csv, encoding="latin-1") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            reader.fieldnames = strip_bom_from_columns(reader.fieldnames)
        for row in reader:
            for col in SOURCE_COLS:
                val = row.get(col, "").strip()
                if len(val) >= 10:
                    col_to_texts[col].append(val)

    for col, texts in col_to_texts.items():
        print(f"  {col}: {len(texts)} documents")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    selector = TextSelector()
    print("Fitting per-column TF-IDF vectorizers (max_features=20000, sublinear_tf=True)...")
    selector.fit(col_to_texts)
    selector.save(args.out)
    print(f"Saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
