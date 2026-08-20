"""Ingest a filled review CSV into the gold annotation store.

Reads the filled review CSV, validates verdict values, resolves confirmed
labels, and writes GOLD_ANNOTATION_CSV.

Schema of GOLD_ANNOTATION_CSV:
  case_id, diagnosis_number, diagnosis,
  matched_term, matched_group, matched_code, matched_keyword, method, confidence,
  tier, verified_by, verified_date, provenance,
  decision_stage, sample_stratum, sample_weight

Mapping rules:
  verdict=correct   → copy cascade_matched_{term,group,code} into matched_*
  verdict=wrong     → use confirmed_{term,group}; matched_code derived from the
                      taxonomy via (group, term), or (term) alone if unambiguous
  verdict=no_cancer / uncertain → matched_term/group/code left empty

The review CSV has no dropdowns, so a `wrong` row whose confirmed term/group is
not in the taxonomy is a validation error rather than a silently code-less gold
row.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

from ICD_labels.taxonomy import load_labels_taxonomy

from .csv_io import read_review_rows

VALID_VERDICTS = frozenset({"correct", "wrong", "no_cancer", "uncertain"})

_GOLD_FIELDNAMES = [
    "case_id", "diagnosis_number", "diagnosis",
    "matched_term", "matched_group", "matched_code",
    "matched_keyword", "method", "confidence",
    "tier", "verified_by", "verified_date", "provenance",
    # Carried from the review CSV: the audit deliberately over-samples the hard
    # strata, so without these no rate computed off this store can be weighted
    # back to the Tier-3 population.
    "decision_stage", "sample_stratum", "sample_weight",
]


def _taxonomy_index(labels_csv: str):
    """Build lookup maps for resolving a confirmed (group, term) to its ICD-O code."""
    by_group_term: dict[tuple[str, str], str] = {}
    by_term: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for lbl in load_labels_taxonomy(labels_csv):
        by_group_term[(lbl.group.lower(), lbl.term.lower())] = lbl.code
        by_term[lbl.term.lower()].append((lbl.group, lbl.code))
    return by_group_term, by_term


def _resolve_wrong(group: str, term: str, by_group_term, by_term) -> tuple[str, str, str]:
    """Return (matched_group, matched_code, error) for a corrected label."""
    gt = (group.lower(), term.lower())
    if gt in by_group_term:
        return group, by_group_term[gt], ""
    # Group left blank or not matching: fall back to term lookup.
    cands = by_term.get(term.lower(), [])
    if not cands:
        return "", "", f"confirmed_term {term!r} is not in the taxonomy"
    groups = sorted({g for g, _ in cands})
    if len(groups) == 1:
        return cands[0][0], cands[0][1], ""   # unambiguous term → backfill group + code
    return "", "", (f"confirmed_term {term!r} does not belong to confirmed_group "
                    f"{group!r}; valid groups for it: {groups}")


def ingest(
    review_csv: str,
    out_csv: str,
    verified_by: str,
    labels_csv: str,
    provenance: str = "round0",
) -> None:
    """Read a filled review CSV and write the gold annotation store."""
    today = date.today().isoformat()
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    by_group_term, by_term = _taxonomy_index(labels_csv)

    errors: list[str] = []
    out_rows: list[dict] = []

    for idx, row in enumerate(read_review_rows(review_csv), start=2):  # +1 header
        verdict = row.get("verdict", "").strip().lower()
        cid = row.get("case_id", "")
        if not verdict:
            errors.append(f"Row {idx}: verdict is blank (case_id={cid})")
            continue
        if verdict not in VALID_VERDICTS:
            errors.append(
                f"Row {idx}: unknown verdict {verdict!r} (case_id={cid}). "
                f"Allowed: {sorted(VALID_VERDICTS)}"
            )
            continue

        if verdict == "correct":
            matched_term = row.get("cascade_matched_term", "")
            matched_group = row.get("cascade_matched_group", "")
            matched_code = row.get("cascade_matched_code", "")
        elif verdict == "wrong":
            matched_term = row.get("confirmed_term", "").strip()
            if not matched_term:
                errors.append(f"Row {idx}: verdict=wrong but confirmed_term is empty (case_id={cid})")
                continue
            matched_group, matched_code, err = _resolve_wrong(
                row.get("confirmed_group", "").strip(), matched_term, by_group_term, by_term)
            if err:
                errors.append(f"Row {idx}: {err} (case_id={cid})")
                continue
        else:  # no_cancer or uncertain
            matched_term = matched_group = matched_code = ""

        out_rows.append({
            "case_id": cid,
            "diagnosis_number": row.get("diagnosis_number", ""),
            "diagnosis": row.get("diagnosis", ""),
            "matched_term": matched_term,
            "matched_group": matched_group,
            "matched_code": matched_code,
            "matched_keyword": "",
            "method": row.get("cascade_method", ""),
            "confidence": "",
            "tier": "gold",
            "verified_by": verified_by,
            "verified_date": today,
            "provenance": provenance,
            "decision_stage": row.get("decision_stage", ""),
            "sample_stratum": row.get("sample_stratum", ""),
            "sample_weight": row.get("sample_weight", ""),
        })

    if errors:
        print("ERROR — ingestion aborted due to validation failures:", file=sys.stderr)
        for msg in errors:
            print(f"  {msg}", file=sys.stderr)
        sys.exit(1)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_GOLD_FIELDNAMES)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Ingested {len(out_rows)} rows into gold store: {out_csv}")
