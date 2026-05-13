"""Apply the post-processing gates to an existing predictions CSV.

Copies the source predictions, runs the subtype and non-neoplastic gates
against the same reports CSV used at inference time, and writes a side-by-
side comparison of method/group/term distributions before vs after.

Skips the tissue-list filter, which requires re-embedding (it acts before
the embedder, not after).

Usage:
    python ml/scripts/apply_gates_to_predictions.py \
        --predictions ml/output/production/historical_archive/petbert_predictions.csv \
        --reports     ml/data/report_36yr.csv \
        --labels      ml/ICD_labels/labels.csv \
        --out-dir     ml/output/evaluation/gates_36yr
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import shutil
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _load_reports_minimal(reports_csv_path: str, id_col: str, text_cols: list[str]) -> dict:
    """Streaming-read of the reports CSV — avoids pulling pandas/torch deps."""
    out: dict[str, dict[str, str]] = {}
    with open(reports_csv_path, encoding="latin-1", newline="") as f:
        reader = csv.DictReader(f)
        # Strip BOM from fieldnames in place.
        reader.fieldnames = [
            (c or "").lstrip("﻿").lstrip("ï»¿") for c in (reader.fieldnames or [])
        ]
        for row in reader:
            cid = (row.get(id_col) or "").strip()
            if not cid:
                continue
            out[cid] = {c: (row.get(c) or "") for c in text_cols}
    return out


def _build_group_nos_index(labels_csv_path: str) -> dict[str, list[dict[str, str]]]:
    """Index every "X, NOS" label by its containing group; head = term minus ", NOS"."""
    out: dict[str, list[dict[str, str]]] = {}
    with open(labels_csv_path, encoding="utf-8-sig") as f:
        lines = f.readlines()
    header_idx = next(
        (i for i, line in enumerate(lines) if "Term" in line and "Group" in line), 0
    )
    reader = csv.DictReader(lines[header_idx:])
    for row in reader:
        term = (row.get("Term") or "").strip()
        group = (row.get("Group") or "").strip()
        code = (row.get("Vet-ICD-O-canine-1 code") or "").strip()
        if not term or not group:
            continue
        if not term.lower().endswith(", nos"):
            continue
        head = term[: -len(", NOS")].strip().lower()
        out.setdefault(group, []).append({"term": term, "head": head, "code": code})
    return out


def _strip_qualifiers(term: str, qualifier_canonicals: list[str], variants_table: dict) -> str:
    import re as _re

    out = term
    for canonical in qualifier_canonicals:
        for variant in variants_table.get(canonical, (canonical,)):
            out = _re.sub(rf"\b{_re.escape(variant)}\b", "", out, flags=_re.IGNORECASE)
    return _re.sub(r"\s+", " ", out).strip(" ,")


def _find_matched_nos(
    nos_index: dict[str, list[dict[str, str]]],
    group: str,
    predicted_term: str,
    missing: list[str],
    variants_table: dict,
) -> dict[str, str] | None:
    candidates = nos_index.get(group)
    if not candidates:
        return None
    stripped = _strip_qualifiers(predicted_term, missing, variants_table).lower()
    if not stripped:
        return None
    for c in candidates:
        if c["head"] == stripped:
            return c
    for c in candidates:
        if stripped in c["head"] or c["head"] in stripped:
            return c
    return None


def _summarize(rows: list[dict]) -> dict:
    return {
        "n_rows": len(rows),
        "method": dict(Counter(r.get("method", "") for r in rows).most_common(20)),
        "top_groups": dict(Counter(r.get("predicted_group", "") for r in rows).most_common(15)),
        "top_terms": dict(Counter(r.get("predicted_term", "") for r in rows).most_common(15)),
    }


def _format_comparison(before: dict, after: dict) -> str:
    lines = [f"rows: {before['n_rows']:,} (unchanged)\n"]

    lines.append("methods:")
    keys = set(before["method"]) | set(after["method"])
    for k in sorted(keys, key=lambda k: -(after["method"].get(k, 0) + before["method"].get(k, 0))):
        b = before["method"].get(k, 0)
        a = after["method"].get(k, 0)
        delta = a - b
        sign = "+" if delta > 0 else ""
        lines.append(f"  {k:<35} before={b:>7,}  after={a:>7,}  Δ {sign}{delta:,}")
    lines.append("")

    lines.append("top groups (by after-count):")
    for k, _ in Counter(after["top_groups"]).most_common(15):
        b = before["top_groups"].get(k, 0)
        a = after["top_groups"].get(k, 0)
        delta = a - b
        sign = "+" if delta > 0 else ""
        lines.append(f"  {k:<45} before={b:>7,}  after={a:>7,}  Δ {sign}{delta:,}")
    lines.append("")

    lines.append("top terms (by after-count):")
    for k, _ in Counter(after["top_terms"]).most_common(15):
        b = before["top_terms"].get(k, 0)
        a = after["top_terms"].get(k, 0)
        delta = a - b
        sign = "+" if delta > 0 else ""
        lines.append(f"  {k:<45} before={b:>7,}  after={a:>7,}  Δ {sign}{delta:,}")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True)
    p.add_argument("--reports", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--id-col", default="case_id")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    after_csv = out_dir / "petbert_predictions_with_gates.csv"
    diff_txt = out_dir / "diff_summary.txt"

    print(f"Copying predictions to {after_csv}")
    shutil.copyfile(args.predictions, after_csv)

    # Load text_filters/gates without importing the package (avoids torch).
    text_filters = _load_module(REPO_ROOT / "ml/production/petbert_pipeline/text_filters.py")
    ancillary_tests_support_neoplasia = text_filters.ancillary_tests_support_neoplasia
    final_comment_has_tumor_evidence = text_filters.final_comment_has_tumor_evidence
    has_non_neoplastic_primary_diagnosis = text_filters.has_non_neoplastic_primary_diagnosis
    looks_non_neoplastic = text_filters.looks_non_neoplastic
    qualifier_words_missing_from_text = text_filters.qualifier_words_missing_from_text
    qualifier_variants = text_filters._QUALIFIER_VARIANTS

    # Build the reports lookup ourselves (gates.py uses pandas; we avoid that).
    print(f"Loading reports from {args.reports} ...")
    text_cols = ["FINAL COMMENT", "HISTOPATHOLOGICAL SUMMARY", "ANCILLARY TESTS"]
    reports = _load_reports_minimal(args.reports, args.id_col, text_cols)
    print(f"  loaded {len(reports):,} reports")

    print(f"Building group→NOS index from {args.labels} ...")
    nos_index = _build_group_nos_index(args.labels)
    print(f"  {len(nos_index):,} groups have at least one NOS variant")

    # Read predictions.
    with open(after_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys()) if rows else []
    before_summary = _summarize(rows)

    # Subtype gate.
    n_demoted = 0
    n_skipped = 0
    for row in rows:
        term = row.get("predicted_term", "")
        group = row.get("predicted_group", "")
        if not term or not group or term == "Uncategorized":
            continue
        cid = row.get(args.id_col, "")
        report = reports.get(cid, {})
        source = " ".join(report.get(c, "") for c in text_cols)
        if not source.strip():
            continue
        missing = qualifier_words_missing_from_text(term, source)
        if not missing:
            continue
        nos = _find_matched_nos(nos_index, group, term, missing, qualifier_variants)
        if nos is None:
            n_skipped += 1
            continue
        if term == nos["term"]:
            continue
        row["predicted_term"] = nos["term"]
        row["predicted_code"] = nos["code"]
        m = row.get("method", "")
        row["method"] = f"{m}+nos_demote" if m else "nos_demote"
        n_demoted += 1
    print(
        f"Subtype gate: demoted {n_demoted:,} predictions to head-matched NOS variant "
        f"({n_skipped:,} skipped — no matching NOS in group)"
    )

    # Non-neoplastic gate.
    suppressed_ids: set[str] = set()
    n_final_comment_vetoed = 0
    n_ancillary_vetoed = 0
    for cid, cols in reports.items():
        fc = cols.get("FINAL COMMENT", "")
        hp = cols.get("HISTOPATHOLOGICAL SUMMARY", "")
        ancillary = cols.get("ANCILLARY TESTS", "")
        if not has_non_neoplastic_primary_diagnosis(fc):
            continue
        if final_comment_has_tumor_evidence(fc):
            n_final_comment_vetoed += 1
            continue
        if ancillary_tests_support_neoplasia(ancillary):
            n_ancillary_vetoed += 1
            continue
        if looks_non_neoplastic(fc, hp, ancillary):
            suppressed_ids.add(cid)
    n_suppressed = 0
    for row in rows:
        cid = row.get(args.id_col, "")
        if cid not in suppressed_ids or row.get("predicted_term") == "Non-neoplastic":
            continue
        row["predicted_term"] = "Non-neoplastic"
        row["predicted_group"] = "Non-neoplastic"
        row["predicted_code"] = ""
        m = row.get("method", "")
        row["method"] = f"{m}+non_neoplastic" if m else "non_neoplastic"
        n_suppressed += 1
    print(
        f"Non-neoplastic gate: flagged {len(suppressed_ids):,} cases, "
        f"suppressed {n_suppressed:,} prediction rows "
        f"({n_final_comment_vetoed:,} final-comment tumor-evidence vetoes, "
        f"{n_ancillary_vetoed:,} ancillary tumor-evidence vetoes)"
    )

    # Write back.
    with open(after_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    after_summary = _summarize(rows)
    diff = _format_comparison(before_summary, after_summary)
    diff_txt.write_text(diff)
    print()
    print(diff)
    print()
    print(f"Wrote: {after_csv}")
    print(f"Wrote: {diff_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
