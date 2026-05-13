"""Post-processing gates that mutate the predictions CSV in place.

Two independent gates, both run after the main categorization (and after
the kNN cascade if enabled):

    apply_subtype_gate
        If a predicted term contains a hallucinated qualifier word
        (e.g., "Microcystic meningioma" when the report doesn't mention
        "microcystic"), demote to the matching NOS variant in the same
        group — the one that shares the predicted term's head noun
        ("Microcystic meningioma" -> "Meningioma, NOS"). When no head-
        matched NOS exists we leave the prediction alone rather than
        risk a malignancy flip (e.g., "Periosteal osteosarcoma" must
        not become "Osteoma, NOS").

    apply_non_neoplastic_gate
        If the report's primary diagnosis looks non-neoplastic
        (hyperplasia, cyst, -itis, degeneration) with no competing tumor
        diagnosis, replace the prediction with "Non-neoplastic".

Both gates read the predictions CSV produced by ``write_predictions_csv``
and write back to the same path, mirroring the contract used by
``cascade.apply_cascade``.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict

import pandas as pd

from .text_filters import (
    _QUALIFIER_VARIANTS,
    ancillary_tests_support_neoplasia,
    final_comment_has_tumor_evidence,
    has_non_neoplastic_primary_diagnosis,
    looks_non_neoplastic,
    qualifier_words_missing_from_text,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_predictions(path: str) -> tuple[list[dict], list[str]]:
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys()) if rows else []
    return rows, fieldnames


def _write_predictions(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_reports(reports_csv_path: str, id_col: str, text_cols: list[str]) -> dict[str, dict[str, str]]:
    """Map case_id -> {col: text}. Used so gates can inspect the source text."""
    df = pd.read_csv(reports_csv_path, encoding="latin-1", low_memory=False)
    df.columns = [c.lstrip("﻿").lstrip("ï»¿") for c in df.columns]
    keep = [id_col] + [c for c in text_cols if c in df.columns]
    df = df[keep].fillna("")
    return {
        str(r[id_col]).strip(): {c: str(r.get(c, "")) for c in text_cols}
        for _, r in df.iterrows()
    }


def _build_group_nos_index(labels_csv_path: str) -> dict[str, list[dict[str, str]]]:
    """Index every "X, NOS" label by its containing group.

    Returns: group -> [{"term": "Osteosarcoma, NOS", "head": "osteosarcoma", "code": "..."}, ...].
    The ``head`` is the predicted-term-side string we match against (term
    minus the trailing ``, NOS``). Subtype demotion finds the entry whose
    head matches the predicted term after stripping its missing qualifiers.
    """
    out: dict[str, list[dict[str, str]]] = defaultdict(list)
    with open(labels_csv_path, encoding="utf-8-sig") as f:
        lines = f.readlines()
    header_idx = next(
        (i for i, line in enumerate(lines) if "Term" in line and "Group" in line),
        0,
    )
    reader = csv.DictReader(lines[header_idx:])
    for row in reader:
        term = row.get("Term", "").strip()
        group = row.get("Group", "").strip()
        code = row.get("Vet-ICD-O-canine-1 code", "").strip()
        if not term or not group:
            continue
        if not term.lower().endswith(", nos"):
            continue
        head = term[: -len(", NOS")].strip().lower()
        out[group].append({"term": term, "head": head, "code": code})
    return out


def _strip_qualifiers(term: str, qualifier_canonicals: list[str]) -> str:
    """Remove the listed qualifier words (and their variants) from ``term``."""
    out = term
    for canonical in qualifier_canonicals:
        for variant in _QUALIFIER_VARIANTS.get(canonical, (canonical,)):
            out = re.sub(rf"\b{re.escape(variant)}\b", "", out, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", out).strip(" ,")


def _find_matched_nos(
    nos_index: dict[str, list[dict[str, str]]],
    group: str,
    predicted_term: str,
    missing_qualifiers: list[str],
) -> dict[str, str] | None:
    """Find the NOS entry in ``group`` whose head matches the qualifier-stripped term."""
    candidates = nos_index.get(group)
    if not candidates:
        return None
    stripped = _strip_qualifiers(predicted_term, missing_qualifiers).lower()
    if not stripped:
        return None
    # Prefer exact head match; fall back to head-substring match.
    for c in candidates:
        if c["head"] == stripped:
            return c
    for c in candidates:
        if stripped in c["head"] or c["head"] in stripped:
            return c
    return None


# ---------------------------------------------------------------------------
# Subtype-qualifier gate
# ---------------------------------------------------------------------------

def apply_subtype_gate(
    predictions_csv: str,
    reports_csv_path: str,
    labels_csv_path: str,
    id_col: str = "case_id",
    text_cols: tuple[str, ...] = ("FINAL COMMENT", "HISTOPATHOLOGICAL SUMMARY", "ANCILLARY TESTS"),
) -> None:
    """Demote hallucinated subtype predictions to the group's NOS variant.

    For each row, if the predicted term contains a recognized qualifier word
    (e.g. "Microcystic", "Surface", "Infiltrative") that does not appear in
    the case's report text, the term/code are replaced with the group's NOS
    entry and ``method`` is suffixed with ``+nos_demote``.
    """
    rows, fieldnames = _load_predictions(predictions_csv)
    if not rows:
        return
    reports = _load_reports(reports_csv_path, id_col, list(text_cols))
    nos_index = _build_group_nos_index(labels_csv_path)

    n_demoted = 0
    n_skipped_no_match = 0
    for row in rows:
        term = row.get("predicted_term", "")
        group = row.get("predicted_group", "")
        if not term or not group or term == "Uncategorized":
            continue
        cid = row.get(id_col, "")
        report = reports.get(cid, {})
        source = " ".join(report.get(c, "") for c in text_cols)
        if not source.strip():
            continue
        missing = qualifier_words_missing_from_text(term, source)
        if not missing:
            continue
        nos = _find_matched_nos(nos_index, group, term, missing)
        if nos is None:
            # No head-matched NOS variant: leave the prediction alone rather
            # than risk demoting a malignant tumor to a benign NOS.
            n_skipped_no_match += 1
            continue
        if term == nos["term"]:
            continue
        row["predicted_term"] = nos["term"]
        row["predicted_code"] = nos["code"]
        existing_method = row.get("method", "")
        row["method"] = f"{existing_method}+nos_demote" if existing_method else "nos_demote"
        n_demoted += 1

    _write_predictions(predictions_csv, rows, fieldnames)
    print(
        f"Subtype gate: demoted {n_demoted} predictions to head-matched NOS variant "
        f"({n_skipped_no_match} skipped — no matching NOS in group)"
    )


# ---------------------------------------------------------------------------
# Non-neoplastic gate
# ---------------------------------------------------------------------------

NON_NEOPLASTIC_TERM = "Non-neoplastic"
NON_NEOPLASTIC_GROUP = "Non-neoplastic"


def apply_non_neoplastic_gate(
    predictions_csv: str,
    reports_csv_path: str,
    id_col: str = "case_id",
    final_comment_col: str = "FINAL COMMENT",
    hp_summary_col: str = "HISTOPATHOLOGICAL SUMMARY",
    ancillary_tests_col: str = "ANCILLARY TESTS",
) -> None:
    """Suppress cancer predictions when the report's primary diagnosis is non-neoplastic.

    Replaces predicted_term / predicted_group / predicted_code for affected
    rows and suffixes ``method`` with ``+non_neoplastic`` so downstream
    consumers can audit the gate's decisions.
    """
    rows, fieldnames = _load_predictions(predictions_csv)
    if not rows:
        return
    reports = _load_reports(
        reports_csv_path, id_col, [final_comment_col, hp_summary_col, ancillary_tests_col]
    )

    suppressed_ids: set[str] = set()
    n_final_comment_vetoed = 0
    n_ancillary_vetoed = 0
    for cid, cols in reports.items():
        fc = cols.get(final_comment_col, "")
        hp = cols.get(hp_summary_col, "")
        ancillary = cols.get(ancillary_tests_col, "")
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
        cid = row.get(id_col, "")
        if cid not in suppressed_ids:
            continue
        if row.get("predicted_term") == NON_NEOPLASTIC_TERM:
            continue
        row["predicted_term"] = NON_NEOPLASTIC_TERM
        row["predicted_group"] = NON_NEOPLASTIC_GROUP
        row["predicted_code"] = ""
        existing_method = row.get("method", "")
        row["method"] = (
            f"{existing_method}+non_neoplastic" if existing_method else "non_neoplastic"
        )
        n_suppressed += 1

    _write_predictions(predictions_csv, rows, fieldnames)
    print(
        f"Non-neoplastic gate: suppressed {n_suppressed} predictions across "
        f"{len(suppressed_ids)} flagged cases "
        f"({n_final_comment_vetoed} final-comment tumor-evidence vetoes, "
        f"{n_ancillary_vetoed} ancillary tumor-evidence vetoes)"
    )
