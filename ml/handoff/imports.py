"""Imports: files the backend developer exports from the cloud, landed locally.

Two inbox kinds (ml-rewrite-plan.md Artefacts, "Handoff"):

- **Pending diagnoses** (``pending_diagnoses_<export>.csv``): new diagnosis
  text awaiting the local cascade (icd-mapping-strategy.md 2.1). Every import
  is keyed by a mandatory ``export_id``; a later export's rows for a case_id
  replace that case's earlier rows in the cumulative landing table
  (``config.HANDOFF_PENDING_DIAGNOSES_CSV``) — the same "later export
  replaces the earlier one" rule the gold store applies per case. This module
  owns that merge; there is no other public store for pending diagnoses to
  delegate to.
- **Gold** (``gold_<export>.csv``): case-level specialist review. Each row
  carries an ``origin``, or leaves it blank for a case that was on an audit
  list, whose recorded origin is filled in. Storage, origin validation and the
  per-case replace rule all belong to ``manual_audit.gold.ingest_gold``
  already — this module only lands a raw copy for the audit trail and calls
  that public API, per CLAUDE.md ("Don't reach through modules... call the
  public interface") and this WP's scope ("delegate storage to
  manual_audit.gold's public ingest API — don't duplicate gold-store logic").

Every raw import is also copied into ``config.HANDOFF_INBOX_DIR`` under its
export-stamped filename with a ``contracts.write_sidecar`` manifest
(schema_version, sha256) — an audit trail of exactly what the cloud sent and
when, independent of what the merge/ingest step did with it. Both csvs are
read as ``utf-8`` (handoff files are the rewrite's own outputs, not the raw
latin-1 legacy corpus) with ``dtype=str, keep_default_na=False``, matching the
rest of the rewrite's own-store convention.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import config
import io_utils
from diagnosis_mapping.silver import DIAG_NUM_COL, ID_COL
from handoff import contracts
from manual_audit import sheets
from manual_audit import audit_list
from manual_audit.gold import ingest_gold

READ_KWARGS = dict(encoding="utf-8", dtype=str, keep_default_na=False)


class HandoffImportError(Exception):
    """An inbox file was refused before it ever reached its target store."""


def _land_copy(rows: pd.DataFrame, filename: str, *, kind: str, schema_version: int) -> Path:
    path = config.HANDOFF_INBOX_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    io_utils.write_csv(rows, path)
    contracts.write_sidecar(path, kind=kind, schema_version=schema_version)
    return path


def import_pending_diagnoses(
    csv_path: str | Path,
    export_id: str,
    *,
    merged_csv: str | Path | None = None,
) -> dict:
    """Land ``csv_path`` (a ``pending_diagnoses_<export_id>.csv``) and merge it
    into the cumulative pending-diagnoses landing table, replacing any earlier
    rows for the same case_id. Returns import + merge counts (never text)."""
    if not export_id:
        raise HandoffImportError("import_pending_diagnoses requires a non-empty export_id")

    rows = io_utils.read_csv(csv_path, **READ_KWARGS)
    missing = set(contracts.PENDING_DIAGNOSES_REQUIRED_COLUMNS) - set(rows.columns)
    if missing:
        raise HandoffImportError(f"{csv_path}: missing required column(s) {sorted(missing)}")
    if DIAG_NUM_COL not in rows.columns:
        rows = rows.copy()
        rows[DIAG_NUM_COL] = "1"
    rows = rows[contracts.PENDING_DIAGNOSES_COLUMNS]

    blank_case = rows[ID_COL].str.strip() == ""
    if blank_case.any():
        raise HandoffImportError(f"{csv_path}: {int(blank_case.sum())} row(s) have a blank {ID_COL}")

    _land_copy(
        rows, f"pending_diagnoses_{export_id}.csv",
        kind=contracts.PENDING_DIAGNOSES_KIND, schema_version=contracts.PENDING_DIAGNOSES_SCHEMA_VERSION,
    )

    merged_path = Path(merged_csv) if merged_csv is not None else config.HANDOFF_PENDING_DIAGNOSES_CSV
    existing = sheets.read_existing_store(merged_path, contracts.PENDING_DIAGNOSES_COLUMNS)

    new_cases = set(rows[ID_COL])
    replaced_cases = new_cases & set(existing[ID_COL]) if len(existing) else set()
    kept = existing[~existing[ID_COL].isin(new_cases)] if len(existing) else existing
    merged = pd.concat([kept, rows], ignore_index=True)

    merged_path.parent.mkdir(parents=True, exist_ok=True)
    io_utils.write_csv(merged, merged_path)

    return {
        "export_id": export_id,
        "imported_rows": len(rows),
        "imported_cases": rows[ID_COL].nunique(),
        "replaced_cases": len(replaced_cases),
        "merged_total_rows": len(merged),
        "merged_total_cases": merged[ID_COL].nunique(),
    }


def import_gold(
    csv_path: str | Path,
    export_id: str,
    *,
    reviewer: str,
    labels_csv: str | Path | None = None,
    upload_period: str = "",
    slice_rate: str = "",
    eval_batch_ledger_csv: str | Path | None = None,
) -> dict:
    """Land ``csv_path`` (a ``gold_<export_id>.csv``) for the audit trail, then
    delegate to ``manual_audit.gold.ingest_gold`` for validation and storage
    (per-case replace). A row with a blank or absent ``origin`` takes the origin
    its case was listed under (``manual_audit.audit_list``); a case never listed
    must carry one, and a given origin must match the listed one."""
    if not export_id:
        raise HandoffImportError("import_gold requires a non-empty export_id")

    rows = io_utils.read_csv(csv_path, **READ_KWARGS)
    missing = set(contracts.GOLD_IMPORT_REQUIRED_COLUMNS) - set(rows.columns)
    if missing:
        raise HandoffImportError(f"{csv_path}: missing required column(s) {sorted(missing)}")

    with_origin = rows.assign(origin=rows["origin"].str.strip() if "origin" in rows.columns else "")
    listed = audit_list.origin_of(with_origin["case_id"])
    blank = with_origin["origin"] == ""
    unlisted = sorted(set(with_origin.loc[blank, "case_id"]) - set(listed))
    if unlisted:
        raise HandoffImportError(f"{len(unlisted)} case(s) have no origin and were never on an audit list: "
                                 f"{unlisted[:5]}")
    conflicting = sorted({c for c, o in zip(with_origin["case_id"], with_origin["origin"])
                          if o and c in listed and listed[c] != o})
    if conflicting:
        raise HandoffImportError(f"{len(conflicting)} case(s) carry an origin other than the one they were listed "
                                 f"under: {conflicting[:5]}")
    with_origin.loc[blank, "origin"] = with_origin.loc[blank, "case_id"].map(listed)

    result = ingest_gold(
        with_origin, reviewer=reviewer, source_path=csv_path, labels_csv=labels_csv,
        batch_or_export_id=export_id, upload_period=upload_period, slice_rate=slice_rate,
        eval_batch_ledger_csv=eval_batch_ledger_csv,
    )

    _land_copy(
        rows, f"gold_{export_id}.csv",
        kind=contracts.GOLD_IMPORT_KIND, schema_version=contracts.GOLD_IMPORT_SCHEMA_VERSION,
    )

    return {**result, "export_id": export_id}
