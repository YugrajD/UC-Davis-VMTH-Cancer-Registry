"""Build (report_text, label_text) pairs for contrastive PetBERT fine-tuning.

Reads keyword_annotation.csv for keyword-confirmed (case, term) pairs and
report.csv for the full report text. Outputs a flat CSV of pairs that
train_contrastive.py consumes directly.

Label text format matches the pipeline exactly: "{term} {group}".
Report text format: non-empty columns joined as "[COL NAME] text".
"""

import argparse
import csv
import sys
from pathlib import Path

# Allow running as a script from any directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


DEFAULT_TEXT_COLS = (
    "FINAL COMMENT",
    "HISTOPATHOLOGICAL SUMMARY",
    "ANCILLARY TESTS",
)


def build_contrastive_pairs(
    *,
    reports_csv: str = "ml/data/report.csv",
    annotation_csv: str = "ml/output/annotation/keyword/keyword_annotation.csv",
    out_csv: str = "ml/data/contrastive_pairs.csv",
    text_cols: tuple[str, ...] = DEFAULT_TEXT_COLS,
    min_report_chars: int = 10,
) -> int:
    """Build and save contrastive pairs. Returns number of pairs written."""

    # --- Step 1: Load positive (case_id, term, group) pairs -----------------
    print(f"Loading keyword annotations from {annotation_csv}...")
    # case_id -> set of (term, group) tuples, deduplicated
    case_to_labels: dict[str, set[tuple[str, str]]] = {}
    with open(annotation_csv, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("method", "") == "no_match":
                continue
            case_id = row["case_id"].strip()
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
    pairs: list[dict[str, str]] = []
    skipped_short = 0

    with open(reports_csv, encoding="latin-1") as f:
        reader = csv.DictReader(f)
        # Strip UTF-8 BOM artifacts that Excel-exported CSVs sometimes add
        if reader.fieldnames:
            reader.fieldnames = [
                c.lstrip("\ufeff").lstrip("ï»¿") for c in reader.fieldnames
            ]
        available = set(reader.fieldnames or [])
        valid_cols = [c for c in text_cols if c in available]
        missing = [c for c in text_cols if c not in available]
        if missing:
            print(f"  Warning: columns not found in report CSV: {missing}")
        if not valid_cols:
            raise ValueError(f"None of {text_cols} exist in {reports_csv}.")

        for row in reader:
            case_id = row.get("case_id", "").strip()
            if case_id not in case_to_labels:
                continue

            # Concatenate non-empty columns with section markers
            parts = [
                f"[{col}] {row.get(col, '').strip()}"
                for col in valid_cols
                if row.get(col, "").strip()
            ]
            report_text = "\n\n".join(parts)

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build contrastive (report_text, label_text) pairs for PetBERT fine-tuning."
    )
    parser.add_argument("--reports-csv", default="ml/data/report.csv",
                        help="Path to report text CSV (default: ml/data/report.csv)")
    parser.add_argument("--annotation-csv",
                        default="ml/output/annotation/keyword/keyword_annotation.csv",
                        help="Path to keyword annotation CSV")
    parser.add_argument("--out-csv", default="ml/data/contrastive_pairs.csv",
                        help="Output CSV path (default: ml/data/contrastive_pairs.csv)")
    args = parser.parse_args(argv)

    n = build_contrastive_pairs(
        reports_csv=args.reports_csv,
        annotation_csv=args.annotation_csv,
        out_csv=args.out_csv,
    )
    print(f"Done — {n} pairs written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
