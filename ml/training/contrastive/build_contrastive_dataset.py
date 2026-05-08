"""Build (report_text, label_text) pairs for contrastive PetBERT fine-tuning.

Reads annotation CSV for confirmed (case, term) pairs and report.csv for the
full report text. Outputs a flat CSV of pairs that train_contrastive.py
consumes directly.

Label text format: "{term} {group}".
Report text format: TF-IDF-selected multi-column text matching the production pipeline.
"""

import argparse
import csv
import sys
from pathlib import Path

# Allow running as a script from any directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import config
from text_selection import get_selector, SOURCE_COLS as _TFIDF_SOURCE_COLS


def build_contrastive_pairs(
    *,
    reports_csv: str = config.REPORTS_CSV,
    annotation_csv: str = config.ANNOTATION_CSV,
    out_csv: str = config.CONTRASTIVE_PAIRS_CSV,
    tfidf_vectorizer_path: str = config.TFIDF_VECTORIZER_PATH,
    min_report_chars: int = 10,
    train_cases_txt: str = "",
) -> int:
    """Build and save contrastive pairs. Returns number of pairs written."""

    # --- Step 1: Load positive (case_id, term, group) pairs -----------------
    train_ids: set[str] | None = None
    if train_cases_txt and Path(train_cases_txt).exists():
        with open(train_cases_txt, encoding="utf-8") as f:
            train_ids = {line.strip() for line in f if line.strip()}
        print(f"  Train/test split active — restricting to {len(train_ids)} train cases.")

    print(f"Loading keyword annotations from {annotation_csv}...")
    # case_id -> set of (term, group) tuples, deduplicated
    case_to_labels: dict[str, set[tuple[str, str]]] = {}
    with open(annotation_csv, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("method", "") == "no_match":
                continue
            case_id = row["case_id"].strip()
            if train_ids is not None and case_id not in train_ids:
                continue
            term = row.get("matched_term", "").strip()
            group = row.get("matched_group", "").strip()
            if not case_id or not term or not group:
                continue
            case_to_labels.setdefault(case_id, set()).add((term, group))

    n_cases = len(case_to_labels)
    n_total = sum(len(v) for v in case_to_labels.values())
    print(f"  {n_cases} cases, {n_total} unique (case, label) pairs.")

    # --- Step 2: Join with report text --------------------------------------
    print(f"Loading report text from {reports_csv}...")
    selector = get_selector(tfidf_vectorizer_path)
    pairs: list[dict[str, str]] = []
    skipped_short = 0

    with open(reports_csv, encoding="latin-1") as f:
        reader = csv.DictReader(f)
        # Strip UTF-8 BOM artifacts that Excel-exported CSVs sometimes add
        if reader.fieldnames:
            reader.fieldnames = [
                c.lstrip("\ufeff").lstrip("ï»¿") for c in reader.fieldnames
            ]
        for row in reader:
            case_id = row.get("case_id", "").strip()
            if case_id not in case_to_labels:
                continue

            col_texts = {
                col: row.get(col, "").strip()
                for col in _TFIDF_SOURCE_COLS
            }
            report_text = selector.select(col_texts, max_tokens=512)

            if len(report_text) < min_report_chars:
                skipped_short += 1
                continue

            for term, group in case_to_labels[case_id]:
                pairs.append({
                    "case_id": case_id,
                    "report_text": report_text,
                    "label_text": f"{term} {group}",
                    "matched_term": term,
                    "matched_group": group,
                })

    if skipped_short:
        print(f"  Skipped {skipped_short} cases with fewer than {min_report_chars} report characters.")
    print(f"  Generated {len(pairs)} contrastive pairs.")

    # --- Step 3: Write CSV --------------------------------------------------
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["case_id", "report_text", "label_text", "matched_term", "matched_group"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pairs)

    print(f"Saved to {out_path}")
    return len(pairs)


def build_hard_neg_pairs(
    *,
    reports_csv: str = config.REPORTS_CSV,
    annotation_csv: str = config.ANNOTATION_CSV,
    co_bank_csv: str,
    out_csv: str = config.HARD_NEG_PAIRS_CSV,
    tfidf_vectorizer_path: str = config.TFIDF_VECTORIZER_PATH,
    min_report_chars: int = 10,
    train_cases_txt: str = "",
) -> int:
    """Build hard-negative triplets from the CO (wrong-group) feedback bank.

    For each case_id in the CO bank with verdict='completely_off', we look up
    the correct label(s) from annotation_csv and the report text from reports_csv,
    then emit a row for each (correct_label, wrong_label) combination.

    Output columns: case_id, report_text, correct_label_text, wrong_label_text

    Returns number of triplets written.
    """

    # --- Step 1: Load correct labels from annotation ------------------------
    train_ids: set[str] | None = None
    if train_cases_txt and Path(train_cases_txt).exists():
        with open(train_cases_txt, encoding="utf-8") as f:
            train_ids = {line.strip() for line in f if line.strip()}
        print(f"  Train/test split active — restricting to {len(train_ids)} train cases.")

    print(f"Loading keyword annotations from {annotation_csv}...")
    case_to_correct: dict[str, set[tuple[str, str]]] = {}
    with open(annotation_csv, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("method", "") == "no_match":
                continue
            case_id = row["case_id"].strip()
            if train_ids is not None and case_id not in train_ids:
                continue
            term = row.get("matched_term", "").strip()
            group = row.get("matched_group", "").strip()
            if not case_id or not term or not group:
                continue
            case_to_correct.setdefault(case_id, set()).add((term, group))

    # --- Step 2: Load wrong-group predictions from CO bank ------------------
    print(f"Loading CO bank from {co_bank_csv}...")
    case_to_wrong: dict[str, set[tuple[str, str]]] = {}
    with open(co_bank_csv, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("verdict", "") != "completely_off":
                continue
            case_id = row["case_id"].strip()
            wrong_term = row.get("predicted_term", "").strip()
            wrong_group = row.get("predicted_group", "").strip()
            if not case_id or not wrong_term or not wrong_group:
                continue
            # Only keep cases that also have a verified correct label
            if case_id not in case_to_correct:
                continue
            case_to_wrong.setdefault(case_id, set()).add((wrong_term, wrong_group))

    eligible = len(case_to_wrong)
    print(f"  {eligible} cases with both a verified correct label and a wrong-group prediction.")

    # --- Step 3: Join with report text and build triplets -------------------
    print(f"Loading report text from {reports_csv}...")
    selector = get_selector(tfidf_vectorizer_path)
    triplets: list[dict[str, str]] = []
    skipped_short = 0
    seen: set[tuple[str, str, str]] = set()  # dedup by (case_id, correct_label, wrong_label)

    with open(reports_csv, encoding="latin-1") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            reader.fieldnames = [
                c.lstrip("\ufeff").lstrip("ï»¿") for c in reader.fieldnames
            ]
        for row in reader:
            case_id = row.get("case_id", "").strip()
            if case_id not in case_to_wrong:
                continue

            col_texts = {
                col: row.get(col, "").strip()
                for col in _TFIDF_SOURCE_COLS
            }
            report_text = selector.select(col_texts, max_tokens=512)
            if len(report_text) < min_report_chars:
                skipped_short += 1
                continue

            for correct_term, correct_group in case_to_correct[case_id]:
                correct_label_text = f"{correct_term} {correct_group}"
                for wrong_term, wrong_group in case_to_wrong[case_id]:
                    wrong_label_text = f"{wrong_term} {wrong_group}"
                    # Skip if wrong == correct (in-bag false negative)
                    if wrong_label_text == correct_label_text:
                        continue
                    key = (case_id, correct_label_text, wrong_label_text)
                    if key in seen:
                        continue
                    seen.add(key)
                    triplets.append({
                        "case_id": case_id,
                        "report_text": report_text,
                        "correct_label_text": correct_label_text,
                        "wrong_label_text": wrong_label_text,
                    })

    if skipped_short:
        print(f"  Skipped {skipped_short} cases with fewer than {min_report_chars} report characters.")
    print(f"  Generated {len(triplets)} hard-negative triplets.")

    # --- Step 4: Write CSV --------------------------------------------------
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["case_id", "report_text", "correct_label_text", "wrong_label_text"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(triplets)

    print(f"Saved to {out_path}")
    return len(triplets)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build training data for contrastive PetBERT fine-tuning.",
    )
    parser.add_argument(
        "--mode",
        choices=["build-pairs", "build-hard-neg"],
        default="build-pairs",
        help=(
            "build-pairs: (report, label) positive pairs for InfoNCE (default). "
            "build-hard-neg: (report, correct_label, wrong_label) triplets from CO bank."
        ),
    )
    parser.add_argument("--reports-csv", default=config.REPORTS_CSV,
                        help=f"Path to report text CSV (default: {config.REPORTS_CSV})")
    parser.add_argument("--annotation-csv", default=config.ANNOTATION_CSV,
                        help="Path to annotation CSV")
    parser.add_argument("--co-bank-csv", default=None,
                        help="[build-hard-neg] Path to evaluation CO bank CSV "
                             "(e.g. ml/output/training/binary/evaluation_co_bank.csv)")
    parser.add_argument("--out-csv", default=None,
                        help="Output CSV path "
                             f"(default: {config.CONTRASTIVE_PAIRS_CSV} for build-pairs, "
                             f"{config.HARD_NEG_PAIRS_CSV} for build-hard-neg)")
    parser.add_argument("--train-cases", default="",
                        help="Path to train_cases.txt. When provided, only train cases are "
                             "included in the output. Generate with create_split.py.")
    args = parser.parse_args(argv)

    if args.mode == "build-pairs":
        out = args.out_csv or config.CONTRASTIVE_PAIRS_CSV
        n = build_contrastive_pairs(
            reports_csv=args.reports_csv,
            annotation_csv=args.annotation_csv,
            out_csv=out,
            train_cases_txt=args.train_cases,
        )
        print(f"Done — {n} pairs written.")

    else:  # build-hard-neg
        if not args.co_bank_csv:
            print("Error: --co-bank-csv is required for --mode build-hard-neg")
            return 1
        out = args.out_csv or config.HARD_NEG_PAIRS_CSV
        n = build_hard_neg_pairs(
            reports_csv=args.reports_csv,
            annotation_csv=args.annotation_csv,
            co_bank_csv=args.co_bank_csv,
            out_csv=out,
            train_cases_txt=args.train_cases,
        )
        print(f"Done — {n} triplets written.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
