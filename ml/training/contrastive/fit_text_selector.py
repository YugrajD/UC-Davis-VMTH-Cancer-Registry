"""Fit and save the TF-IDF vectorizer for multi-column text selection.

Run once before building contrastive pairs with the TF-IDF selector:

  ml/.venv/Scripts/python.exe ml/training/contrastive/fit_text_selector.py
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import config
from production.petbert_pipeline.text_selector import TextSelector, SOURCE_COLS


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit TF-IDF vectorizer on the report corpus for text selection."
    )
    parser.add_argument("--reports-csv", default=config.REPORTS_CSV)
    parser.add_argument("--out", default=config.TFIDF_VECTORIZER_PATH)
    args = parser.parse_args()

    print(f"Reading reports from {args.reports_csv}...")
    texts: list[str] = []
    with open(args.reports_csv, encoding="latin-1") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            reader.fieldnames = [
                c.lstrip("﻿").lstrip("ï»¿") for c in reader.fieldnames
            ]
        for row in reader:
            parts: list[str] = []
            for col in SOURCE_COLS:
                val = row.get(col, "").strip()
                if len(val) >= 10:
                    parts.append(val)
            if parts:
                texts.append(" ".join(parts))

    print(f"  {len(texts)} documents loaded.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    selector = TextSelector()
    print("Fitting TF-IDF vectorizer (max_features=20000, sublinear_tf=True)...")
    selector.fit(texts)
    selector.save(args.out)
    print(f"Saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
