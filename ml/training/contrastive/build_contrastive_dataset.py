"""Build (report_text, label_text) pairs for contrastive PetBERT fine-tuning.

Reads annotation CSV for confirmed (case, term) pairs and report.csv for the
report text. Outputs a flat CSV of pairs that train_contrastive.py consumes
directly.

Pairs are per-section: one row per (case, label, section) where section_text is
non-empty. Section groups match SECTIONS_3 in scripts/run_embed_compare.py so
the backbone sees the same text distribution that concat_3 inference consumes.
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import config

_PER_SECTION_GROUPS: tuple[tuple[str, ...], ...] = (
    ("HISTOPATHOLOGICAL SUMMARY",),
    ("FINAL COMMENT", "COMMENT"),
    ("ANCILLARY TESTS",),
)


def _section_name(group: tuple[str, ...]) -> str:
    return group[0] if len(group) == 1 else "+".join(group)


def _load_case_labels(
    annotation_csv: str,
    train_ids: set[str] | None,
) -> dict[str, set[tuple[str, str]]]:
    """Return case_id → {(term, group), ...} from annotation CSV."""
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
    return case_to_labels


def _load_train_ids(train_cases_txt: str) -> set[str] | None:
    if not train_cases_txt or not Path(train_cases_txt).exists():
        return None
    with open(train_cases_txt, encoding="utf-8") as f:
        ids = {line.strip() for line in f if line.strip()}
    print(f"  Train/test split active — restricting to {len(ids)} train cases.")
    return ids


def _iter_report_rows(reports_csv: str):
    """Yield (case_id, row_dict) from reports CSV, stripping BOM artifacts."""
    with open(reports_csv, encoding="latin-1") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            reader.fieldnames = [
                c.lstrip("﻿").lstrip("ï»¿") for c in reader.fieldnames
            ]
        for row in reader:
            case_id = row.get("case_id", "").strip()
            if case_id:
                yield case_id, row


def _section_texts(row: dict) -> list[str]:
    """Return [HIST, FC+C, ANC] section texts joined per group."""
    out: list[str] = []
    for group in _PER_SECTION_GROUPS:
        parts = [(row.get(col, "") or "").strip() for col in group]
        out.append("\n".join(p for p in parts if p))
    return out


def build_contrastive_pairs(
    *,
    reports_csv: str = config.REPORTS_CSV,
    annotation_csv: str = config.ANNOTATION_CSV,
    out_csv: str = config.CONTRASTIVE_PAIRS_CSV,
    min_report_chars: int = 10,
    train_cases_txt: str = "",
) -> int:
    """Build per-section (report_text, label_text) pairs. Returns row count.

    Emits one row per (case, label, section) where the section has at least
    `min_report_chars` characters. Cases with all three sections empty are
    silently skipped (~0.25% of train cases historically).
    """
    train_ids = _load_train_ids(train_cases_txt)

    print(f"Loading annotations from {annotation_csv}...")
    case_to_labels = _load_case_labels(annotation_csv, train_ids)
    n_cases = len(case_to_labels)
    n_total_labels = sum(len(v) for v in case_to_labels.values())
    print(f"  {n_cases} cases, {n_total_labels} unique (case, label) pairs.")

    print(f"Loading report text from {reports_csv}...")
    pairs: list[dict[str, str]] = []
    cases_with_zero_sections = 0
    per_section_counts = [0] * len(_PER_SECTION_GROUPS)

    for case_id, row in _iter_report_rows(reports_csv):
        if case_id not in case_to_labels:
            continue

        kept = [
            (i, _section_name(_PER_SECTION_GROUPS[i]), t)
            for i, t in enumerate(_section_texts(row))
            if len(t) >= min_report_chars
        ]
        if not kept:
            cases_with_zero_sections += 1
            continue

        for term, group in case_to_labels[case_id]:
            label_text = f"{term} {group}"
            for sec_idx, sec_name, sec_text in kept:
                per_section_counts[sec_idx] += 1
                pairs.append({
                    "case_id": case_id,
                    "section": sec_name,
                    "report_text": sec_text,
                    "label_text": label_text,
                    "matched_term": term,
                    "matched_group": group,
                })

    if cases_with_zero_sections:
        print(
            f"  Skipped {cases_with_zero_sections} cases with all 3 sections "
            f"shorter than {min_report_chars} chars."
        )
    for i, sec_group in enumerate(_PER_SECTION_GROUPS):
        print(f"  Section {_section_name(sec_group)!r}: {per_section_counts[i]} pairs")
    print(f"  Generated {len(pairs)} per-section contrastive pairs.")

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["case_id", "section", "report_text", "label_text",
                  "matched_term", "matched_group"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pairs)

    print(f"Saved to {out_path}")
    return len(pairs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build per-section (report, label) pairs for contrastive PetBERT fine-tuning.",
    )
    parser.add_argument("--reports-csv", default=config.REPORTS_CSV)
    parser.add_argument("--annotation-csv", default=config.ANNOTATION_CSV)
    parser.add_argument("--out-csv", default=config.CONTRASTIVE_PAIRS_CSV,
                        help=f"Output CSV (default: {config.CONTRASTIVE_PAIRS_CSV})")
    parser.add_argument("--train-cases", default="",
                        help="Path to train_cases.txt to restrict cases.")
    args = parser.parse_args(argv)

    n = build_contrastive_pairs(
        reports_csv=args.reports_csv,
        annotation_csv=args.annotation_csv,
        out_csv=args.out_csv,
        train_cases_txt=args.train_cases,
    )
    print(f"Done — {n} per-section pairs written.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
