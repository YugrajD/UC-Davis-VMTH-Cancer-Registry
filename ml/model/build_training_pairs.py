"""Build training pairs for the binary presence classifier.

Produces ml/data/training_pairs.csv — one row per (case, taxonomy label) pair
with a binary target: 1 if the label is a confirmed diagnosis, 0 otherwise.

Three sources of examples:
  Positives      — rows in keyword_predictions.csv where matched_term is non-blank
  Hard negatives — rows in evaluation.csv with verdict="false_positive"
                   (high cosine similarity but the case has no keyword label at all;
                   these are the most valuable training signal)
  Easy negatives — for each labeled case, randomly sampled wrong taxonomy terms
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from labels.taxonomy import load_labels_taxonomy
from petbert_scan.utils import clean_text, merge_report_columns

_TEXT_COLS = [
    "HISTOPATHOLOGICAL SUMMARY",
    "FINAL COMMENT",
    "ANCILLARY TESTS",
]


def load_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build training pairs for the presence classifier.")
    parser.add_argument("--report-csv", default="ml/data/report.csv")
    parser.add_argument("--keyword-csv", default="ml/output/diagnoses/keyword_predictions.csv")
    parser.add_argument("--evaluation-csv", default="ml/output/evaluation/evaluation.csv")
    parser.add_argument("--labels-csv", default="ml/labels/labels.csv")
    parser.add_argument("--out", default="ml/data/training_pairs.csv")
    parser.add_argument(
        "--easy-neg-per-pos",
        type=int,
        default=3,
        help="Random wrong taxonomy labels to sample per positive example (default: 3)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # --- Load report text ------------------------------------------------
    df = pd.read_csv(args.report_csv, encoding="latin-1")
    df.columns = [col.lstrip("\ufeff").lstrip("ï»¿") for col in df.columns]
    available_cols = [c for c in _TEXT_COLS if c in df.columns]
    if not available_cols:
        print(f"Warning: none of {_TEXT_COLS} found in report CSV. Found: {df.columns.tolist()}")
    df["_merged"] = df.apply(lambda row: merge_report_columns(row, available_cols), axis=1)
    case_to_text: dict[str, str] = {
        clean_text(row["case_id"]): row["_merged"]
        for _, row in df.iterrows()
        if clean_text(row.get("case_id", ""))
    }

    # --- Load taxonomy labels --------------------------------------------
    taxonomy = load_labels_taxonomy(args.labels_csv)
    term_to_group = {t.term: t.group for t in taxonomy}
    all_term_group = [(t.term, t.group) for t in taxonomy]

    # --- Load keyword positives ------------------------------------------
    kw_rows = load_csv(Path(args.keyword_csv))
    case_pos_terms: dict[str, set[str]] = {}
    for row in kw_rows:
        term = row["matched_term"].strip()
        if term:
            case_pos_terms.setdefault(row["case_id"], set()).add(term)

    # --- Build output rows -----------------------------------------------
    out_rows: list[dict] = []

    # Positives
    for cid, terms in case_pos_terms.items():
        text = case_to_text.get(cid, "")
        if not text:
            continue
        for term in terms:
            out_rows.append({
                "case_id": cid,
                "merged_text": text,
                "label_term": term,
                "label_group": term_to_group.get(term, ""),
                "target": 1,
                "source": "positive",
            })

    # Hard negatives — false positive predictions from evaluation
    eval_rows = load_csv(Path(args.evaluation_csv))
    for row in eval_rows:
        if row["verdict"] != "false_positive":
            continue
        text = case_to_text.get(row["case_id"], "")
        if not text:
            continue
        out_rows.append({
            "case_id": row["case_id"],
            "merged_text": text,
            "label_term": row["predicted_term"],
            "label_group": row["predicted_group"],
            "target": 0,
            "source": "hard_negative",
        })

    # Easy negatives — random wrong labels for labeled cases
    for cid, pos_terms in case_pos_terms.items():
        text = case_to_text.get(cid, "")
        if not text:
            continue
        candidates = [(term, group) for term, group in all_term_group if term not in pos_terms]
        sample = random.sample(candidates, min(args.easy_neg_per_pos, len(candidates)))
        for term, group in sample:
            out_rows.append({
                "case_id": cid,
                "merged_text": text,
                "label_term": term,
                "label_group": group,
                "target": 0,
                "source": "easy_negative",
            })

    # --- Write -----------------------------------------------------------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["case_id", "merged_text", "label_term", "label_group", "target", "source"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    n_pos = sum(1 for r in out_rows if r["target"] == 1)
    n_hard = sum(1 for r in out_rows if r["source"] == "hard_negative")
    n_easy = sum(1 for r in out_rows if r["source"] == "easy_negative")
    print(f"Wrote {len(out_rows)} training pairs to {out_path}")
    print(f"  Positives:      {n_pos}")
    print(f"  Hard negatives: {n_hard}")
    print(f"  Easy negatives: {n_easy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
