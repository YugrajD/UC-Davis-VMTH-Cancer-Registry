"""Verification harness for the LLM annotation pipeline.

Reproduces the 90-row stratified sample (seed=42) used in the noise audit and
prints a side-by-side before/after diff so changes from a re-run or a cleanup
pass can be inspected in one place.

Usage:
  python ml/annotation/llm_pipeline/audit.py \
    --before ml/output/annotation/llm_annotation.csv \
    --after  ml/output/annotation/llm_annotation_cleaned.csv

If --after is omitted, only the before sample is printed (useful for spot-checks
against a fresh re-run of the patched pipeline alone).
"""

from __future__ import annotations

import argparse

import pandas as pd


# Audit case IDs known to be flat-wrong in the original llm_annotation.csv.
# Each tuple: (case_id, diagnosis_number, short_label, expected_correction).
KNOWN_ERRORS = [
    ("CASE-44172",  6, "#49 negation",      "should be No Match (no evidence of neoplasia)"),
    ("CASE-28574",  1, "#66 histiocytoma",  "should be Histiocytoma, NOS (benign cutaneous Langerhans tumor)"),
    ("CASE-43123",  5, "#69 granuloma",     "should be No Match (inflammatory episcleral granuloma)"),
    ("CASE-4234",   1, "#71 mixed benign",  "should map to a benign /0 mixed mammary tumor variant"),
    ("CASE-2710",   1, "#73 mixed benign",  "should map to a benign /0 mixed mammary tumor variant"),
    ("CASE-24002", 1,  "#76 non-B/non-T",   "should be No Match or T/B-cell-agnostic NOS"),
]


def _stratified_sample(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for method, n in [("No Match", 30), ("Exact", 20), ("LLM", 20), ("Fuzzy", 10), ("Uncertain", 10)]:
        sub = df[df["method"] == method]
        if len(sub) >= n:
            parts.append(sub.sample(n=n, random_state=42))
        else:
            parts.append(sub)
    return pd.concat(parts, ignore_index=False)


def _row_summary(row: pd.Series) -> str:
    def _show(v):
        return "-" if pd.isna(v) or v == "" else str(v)
    term = _show(row.get("matched_term"))
    grp  = _show(row.get("matched_group"))
    code = _show(row.get("matched_code"))
    method = row.get("method", "?")
    return f"[{method}] {term} | {grp} | {code}"


def _print_known_errors(before: pd.DataFrame, after: pd.DataFrame | None) -> None:
    print("=" * 80)
    print("KNOWN AUDIT FAILURES - regression check")
    print("=" * 80)
    flipped = 0
    for case_id, diag_num, label, expected in KNOWN_ERRORS:
        b_row = before[(before["case_id"] == case_id) & (before["diagnosis_number"] == diag_num)]
        if b_row.empty:
            print(f"\n{label}  ({case_id} #{diag_num})")
            print("  NOT FOUND in --before file")
            continue
        b = b_row.iloc[0]
        print(f"\n{label}  ({case_id} #{diag_num})")
        print(f"  DIAG:     {str(b.get('diagnosis', ''))[:200]}")
        print(f"  Expected: {expected}")
        print(f"  Before:   {_row_summary(b)}")
        if after is not None:
            a_row = after[(after["case_id"] == case_id) & (after["diagnosis_number"] == diag_num)]
            if a_row.empty:
                print("  After:    NOT FOUND in --after file")
                continue
            a = a_row.iloc[0]
            print(f"  After:    {_row_summary(a)}")
            if str(b.get("matched_term", "")) != str(a.get("matched_term", "")) or \
               str(b.get("method", "")) != str(a.get("method", "")):
                flipped += 1
    if after is not None:
        print(f"\nFlipped: {flipped} of {len(KNOWN_ERRORS)} known errors")


def _print_stratified(before: pd.DataFrame, after: pd.DataFrame | None) -> None:
    print("\n" + "=" * 80)
    print("STRATIFIED SAMPLE (seed=42) - before vs. after")
    print("=" * 80)
    sample = _stratified_sample(before)
    for _, b in sample.iterrows():
        diag_num = b.get("diagnosis_number")
        case_id = b["case_id"]
        diag = str(b.get("diagnosis", ""))[:200]
        print(f"\n{case_id} #{diag_num}")
        print(f"  DIAG: {diag}")
        print(f"  Before: {_row_summary(b)}")
        if after is not None:
            mask = (after["case_id"] == case_id) & (after["diagnosis_number"] == diag_num)
            a_row = after[mask]
            if a_row.empty:
                print("  After:  NOT FOUND")
                continue
            a = a_row.iloc[0]
            tag = "" if str(b.get("matched_term")) == str(a.get("matched_term")) and str(b.get("method")) == str(a.get("method")) else " ← changed"
            print(f"  After:  {_row_summary(a)}{tag}")


def _print_aggregate(before: pd.DataFrame, after: pd.DataFrame | None) -> None:
    if after is None:
        return
    print("\n" + "=" * 80)
    print("AGGREGATE CHANGES")
    print("=" * 80)
    before_methods = before["method"].value_counts().to_dict()
    after_methods = after["method"].value_counts().to_dict()
    keys = sorted(set(before_methods) | set(after_methods))
    print(f"{'Method':<15} {'Before':>10} {'After':>10} {'Delta':>10}")
    for k in keys:
        b = before_methods.get(k, 0)
        a = after_methods.get(k, 0)
        print(f"{k:<15} {b:>10,} {a:>10,} {a-b:>+10,}")

    merged = before.merge(
        after[["case_id", "diagnosis_number", "matched_term", "matched_group", "method"]],
        on=["case_id", "diagnosis_number"],
        suffixes=("_before", "_after"),
        how="inner",
    )
    term_changed = merged["matched_term_before"].fillna("") != merged["matched_term_after"].fillna("")
    method_changed = merged["method_before"] != merged["method_after"]
    print(f"\nRows with term  changes: {term_changed.sum():,} / {len(merged):,}")
    print(f"Rows with method changes: {method_changed.sum():,} / {len(merged):,}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit harness for the LLM annotation pipeline.",
    )
    parser.add_argument("--before", required=True, help="Path to original llm_annotation.csv")
    parser.add_argument("--after", default=None, help="Optional path to cleaned/re-run annotation CSV")
    args = parser.parse_args()

    before = pd.read_csv(args.before)
    after = pd.read_csv(args.after) if args.after else None

    _print_known_errors(before, after)
    _print_stratified(before, after)
    _print_aggregate(before, after)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
