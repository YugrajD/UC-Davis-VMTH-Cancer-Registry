"""Build training pairs for the binary presence classifier.

Produces ml/data/training_pairs.csv — one row per (case, taxonomy label) pair
with a binary target: 1 if the label is a confirmed diagnosis, 0 otherwise.

Five sources of examples:
  Positives      — rows in keyword_predictions.csv where matched_term is non-blank
  Hard negatives — rows in evaluation.csv with verdict="false_positive"
                   (high cosine similarity but the case has no keyword label at all;
                   these are the most valuable training signal)
  FP extra neg   — for each unique false-positive case, additional randomly-sampled
                   taxonomy labels beyond those already covered by evaluation rows;
                   teaches the classifier that these cases have NO valid label at all
  CO negatives   — rows in evaluation.csv with verdict="completely_off"
                   (case has a keyword label, but the predicted label is the wrong group;
                   teaches the classifier to reject specific wrong-group predictions)
  Easy negatives — for each labeled case, randomly sampled wrong taxonomy terms
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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
    parser.add_argument(
        "--fp-neg-per-case",
        type=int,
        default=10,
        help="Extra random taxonomy labels to sample per unique false-positive case, "
             "beyond those already covered by evaluation rows (default: 10)",
    )
    parser.add_argument(
        "--co-neg-per-case",
        type=int,
        default=3,
        help="Cap completely-off negatives per case (0 = no cap, default: 3). "
             "These are wrong-group predictions from the previous cycle — "
             "the most targeted signal for reducing the completely-off rate.",
    )
    parser.add_argument(
        "--co-neg-extra-csv",
        default="",
        help="Optional path to a second evaluation.csv to pull additional CO negatives from "
             "(e.g. from a previous best cycle). Combined with --evaluation-csv CO negatives "
             "to reduce oscillation caused by single-cycle feedback.",
    )
    parser.add_argument(
        "--co-neg-bank-csv",
        default="",
        help="Path to the rolling CO-negative bank (maintained by update_co_bank.py). "
             "When provided and the file exists, CO negatives are read from the bank "
             "INSTEAD OF --evaluation-csv, avoiding double-counting across cycles. "
             "Falls back to --evaluation-csv if the bank doesn't exist yet.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-pos-per-group",
        type=int,
        default=0,
        help="Cap positive examples per taxonomy group to this count (0 = no cap). "
             "Use to prevent overrepresented groups (e.g. Adenomas with 251 examples) "
             "from dominating classifier training. Recommended: 80.",
    )
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

    # Positives (optionally capped per group to reduce training imbalance)
    group_pos_count: dict[str, int] = {}
    for cid, terms in case_pos_terms.items():
        text = case_to_text.get(cid, "")
        if not text:
            continue
        for term in terms:
            group = term_to_group.get(term, "")
            if args.max_pos_per_group > 0:
                if group_pos_count.get(group, 0) >= args.max_pos_per_group:
                    continue
                group_pos_count[group] = group_pos_count.get(group, 0) + 1
            out_rows.append({
                "case_id": cid,
                "merged_text": text,
                "label_term": term,
                "label_group": group,
                "target": 1,
                "source": "positive",
            })

    # Hard negatives — false positive predictions from evaluation
    eval_rows = load_csv(Path(args.evaluation_csv))

    # CO negatives — completely-off predictions (case has keyword labels but wrong group)
    # These are the specific (case, wrong-label) pairs that fool the cosine similarity step.
    # Capped per case to avoid one prolific case dominating the training set.
    # Optionally merged from a second historical evaluation CSV to reduce cycle-to-cycle
    # oscillation caused by using only the most recent cycle's predictions.
    # When a rolling bank is provided, use it as the sole CO source.
    # This avoids double-counting: the bank already includes the previous cycle's
    # completely-off rows, which are also present in evaluation.csv.
    if args.co_neg_bank_csv and Path(args.co_neg_bank_csv).exists():
        co_eval_sources = [load_csv(Path(args.co_neg_bank_csv))]
        print(f"  Using CO bank ({args.co_neg_bank_csv})")
    else:
        co_eval_sources = [eval_rows]
        if args.co_neg_extra_csv:
            extra_path = Path(args.co_neg_extra_csv)
            if extra_path.exists():
                co_eval_sources.append(load_csv(extra_path))
                print(f"  Loaded extra CO negatives from {args.co_neg_extra_csv}")
            else:
                print(f"  Warning: --co-neg-extra-csv path not found: {args.co_neg_extra_csv}")

    co_case_count: dict[str, int] = {}
    for co_rows in co_eval_sources:
        for row in co_rows:
            if row["verdict"] != "completely_off":
                continue
            text = case_to_text.get(row["case_id"], "")
            if not text:
                continue
            if args.co_neg_per_case > 0:
                if co_case_count.get(row["case_id"], 0) >= args.co_neg_per_case:
                    continue
                co_case_count[row["case_id"]] = co_case_count.get(row["case_id"], 0) + 1
            out_rows.append({
                "case_id": row["case_id"],
                "merged_text": text,
                "label_term": row["predicted_term"],
                "label_group": row["predicted_group"],
                "target": 0,
                "source": "co_negative",
            })

    fp_case_covered: dict[str, set[str]] = {}  # case_id -> terms already added
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
        fp_case_covered.setdefault(row["case_id"], set()).add(row["predicted_term"])

    # FP extra negatives — additional random labels for each unique FP case
    # The hard negatives above only cover labels the pipeline already predicted.
    # Sampling extra labels teaches the classifier that FP cases have NO valid
    # label anywhere in the taxonomy, not just for the labels already seen.
    for cid, covered_terms in fp_case_covered.items():
        text = case_to_text.get(cid, "")
        if not text:
            continue
        candidates = [(term, group) for term, group in all_term_group if term not in covered_terms]
        sample = random.sample(candidates, min(args.fp_neg_per_case, len(candidates)))
        for term, group in sample:
            out_rows.append({
                "case_id": cid,
                "merged_text": text,
                "label_term": term,
                "label_group": group,
                "target": 0,
                "source": "fp_extra_negative",
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
    n_fp_extra = sum(1 for r in out_rows if r["source"] == "fp_extra_negative")
    n_co = sum(1 for r in out_rows if r["source"] == "co_negative")
    n_easy = sum(1 for r in out_rows if r["source"] == "easy_negative")
    print(f"Wrote {len(out_rows)} training pairs to {out_path}")
    print(f"  Positives:          {n_pos}")
    print(f"  Hard negatives:     {n_hard}")
    print(f"  FP extra negatives: {n_fp_extra}")
    print(f"  CO negatives:       {n_co}")
    print(f"  Easy negatives:     {n_easy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
