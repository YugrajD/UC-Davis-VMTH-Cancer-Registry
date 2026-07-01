"""Stratified-random gold-EVAL sample from test cases.

Draws ~n_cases cases from test_cases.txt, stratified by cancer group so that
every group with at least one test case gets at least one sampled case.  Adds
dup_frac * n_cases duplicate cases (re-listed with dup_pass=2) for
intra-annotator self-consistency, dispersed through the sheet so repeats are not
obvious.

Emits an Excel review workbook (see xlsx_io) — one row per diagnosis row of each
sampled case, with cascade predictions joined from annotation.csv and report
context joined from report.csv.  Also writes a batch-cases ledger (one case_id
per line) used by `check-split` / evaluation and to exclude already-sampled
cases from later batches.
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

from ICD_labels.taxonomy import load_labels_taxonomy

from .xlsx_io import write_review_workbook

# Columns from report.csv to include as reviewer context.
_REPORT_CONTEXT_COLS = ["HISTOPATHOLOGICAL SUMMARY", "FINAL COMMENT"]

# Blank human-input columns (ICD-O code is derived from group+term at ingest).
_HUMAN_COLS = ["verdict", "confirmed_term", "confirmed_group", "notes"]

# Cascade prediction columns joined from annotation.csv.
_CASCADE_COLS = [
    "cascade_matched_term",
    "cascade_matched_group",
    "cascade_matched_code",
    "cascade_method",
]


def _load_case_ids(txt_path: str) -> list[str]:
    with open(txt_path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _load_annotation(annotation_csv: str) -> dict[str, list[dict]]:
    """Return {case_id: [row, ...]} for every annotated case."""
    by_case: dict[str, list[dict]] = defaultdict(list)
    with open(annotation_csv, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            by_case[row["case_id"]].append(row)
    return by_case


def _load_report_context(reports_csv: str, case_ids: set[str]) -> dict[str, dict[str, str]]:
    """Return {case_id: {col: value}} for the requested context columns."""
    context: dict[str, dict[str, str]] = {}
    try:
        with open(reports_csv, encoding="latin-1") as f:
            for row in csv.DictReader(f):
                cid = row.get("case_id", "")
                if cid not in case_ids:
                    continue
                context[cid] = {col: row.get(col, "") for col in _REPORT_CONTEXT_COLS}
    except FileNotFoundError:
        # Report CSV may not be present in all environments; context columns will be empty.
        pass
    return context


def _assign_strata(annotation_by_case: dict[str, list[dict]],
                   test_ids: set[str]) -> dict[str, list[str]]:
    """Map group name → list of test case IDs in that group."""
    strata: dict[str, list[str]] = defaultdict(list)
    for case_id, rows in annotation_by_case.items():
        if case_id not in test_ids:
            continue
        groups = {r["matched_group"] for r in rows if r.get("matched_group", "").strip()}
        if not groups:
            groups = {"__no_group__"}
        for g in groups:
            strata[g].append(case_id)
    return strata


def _taxonomy_lists(labels_csv: str) -> tuple[list[str], list[str], list[tuple[str, str, str]]]:
    """Return (sorted unique groups, sorted unique terms, sorted (group, term, code))."""
    labels = load_labels_taxonomy(labels_csv)
    groups = sorted({lbl.group for lbl in labels})
    terms = sorted({lbl.term for lbl in labels})
    taxonomy_rows = sorted({(lbl.group, lbl.term, lbl.code) for lbl in labels})
    return groups, terms, taxonomy_rows


def sample(
    annotation_csv: str,
    test_cases_txt: str,
    reports_csv: str,
    labels_csv: str,
    out_xlsx: str,
    batch_cases_out: str,
    n_cases: int = 200,
    dup_frac: float = 0.08,
    seed: int = 42,
    exclude_cases: list[str] | None = None,
) -> None:
    """Draw a stratified sample of cases and write the review workbook + ledger."""
    rng = random.Random(seed)

    test_ids = set(_load_case_ids(test_cases_txt))
    annotation_by_case = _load_annotation(annotation_csv)

    # Exclude cases already used by earlier batches.
    excluded: set[str] = set()
    for path in (exclude_cases or []):
        excluded.update(_load_case_ids(path))

    # Only test cases that appear in annotation.csv have a prediction to review.
    test_ids_in_annotation = (test_ids & set(annotation_by_case.keys())) - excluded

    strata = _assign_strata(annotation_by_case, test_ids_in_annotation)

    # A multi-group case appears in several strata; track selections to dedupe.
    selected: list[str] = []
    selected_set: set[str] = set()

    # Phase 1 — force ≥1 case per group (rare groups may be fully included).
    for group, case_list in sorted(strata.items()):
        unique = [c for c in case_list if c not in selected_set]
        if not unique:
            continue
        pick = rng.choice(unique)
        selected.append(pick)
        selected_set.add(pick)

    # Phase 2 — fill up to n_cases from the remaining pool.
    remaining = n_cases - len(selected)
    if remaining > 0:
        pool = [c for c in test_ids_in_annotation if c not in selected_set]
        rng.shuffle(pool)
        extra = pool[:remaining]
        selected.extend(extra)
        selected_set.update(extra)

    # Phase 3 — pick dup_frac duplicates (re-listed for self-consistency).
    n_dups = max(1, round(len(selected) * dup_frac))
    dup_sources = rng.sample(selected, min(n_dups, len(selected)))

    # Build (case_id, dup_pass) blocks and shuffle so duplicates are dispersed.
    blocks = [(cid, 1) for cid in selected] + [(cid, 2) for cid in dup_sources]
    rng.shuffle(blocks)

    all_case_ids = selected_set | set(dup_sources)
    report_context = _load_report_context(reports_csv, all_case_ids)

    header = (
        ["case_id", "diagnosis_number", "diagnosis", "dup_pass"]
        + _CASCADE_COLS
        + _REPORT_CONTEXT_COLS
        + _HUMAN_COLS
    )

    def _make_rows(case_id: str, dup_pass: int) -> list[dict]:
        ctx = report_context.get(case_id, {})
        out = []
        for ann_row in annotation_by_case.get(case_id, []):
            row: dict = {
                "case_id": case_id,
                "diagnosis_number": ann_row.get("diagnosis_number", ""),
                "diagnosis": ann_row.get("diagnosis", ""),
                "dup_pass": dup_pass,
                "cascade_matched_term": ann_row.get("matched_term", ""),
                "cascade_matched_group": ann_row.get("matched_group", ""),
                "cascade_matched_code": ann_row.get("matched_code", ""),
                "cascade_method": ann_row.get("method", ""),
            }
            for col in _REPORT_CONTEXT_COLS:
                row[col] = ctx.get(col, "")
            for col in _HUMAN_COLS:
                row[col] = ""
            out.append(row)
        return out

    review_rows: list[dict] = []
    for case_id, dup_pass in blocks:
        review_rows.extend(_make_rows(case_id, dup_pass))

    groups, terms, taxonomy_rows = _taxonomy_lists(labels_csv)
    write_review_workbook(out_xlsx, header, review_rows, groups, terms, taxonomy_rows)

    # Ledger of the unique sampled case IDs (originals only — duplicates re-use them).
    ledger = Path(batch_cases_out)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text("\n".join(selected) + "\n", encoding="utf-8")

    print(f"Sampled {len(selected)} cases ({len(strata)} strata covered), "
          f"{len(dup_sources)} duplicate cases, {len(review_rows)} review rows.")
    if excluded:
        print(f"Excluded {len(excluded)} cases from earlier batches.")
    print(f"Review workbook: {out_xlsx}")
    print(f"Batch cases ledger: {batch_cases_out}")
