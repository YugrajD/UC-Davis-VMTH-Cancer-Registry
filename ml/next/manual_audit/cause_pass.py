"""Cause pass: on misses only, does the method's own input support the gold code?

Takes a verdict table of misses — ``case_id, gold_code, method, source_version``,
one row per gold code a method (``silver`` or ``bronze``) failed to reach —
and turns it into a sheet the specialist fills in with one judgement per row:
does the diagnosis line (for silver) or the report text (for bronze) actually
support the gold code? "Yes" is a method error, fixable in that method; "no" is
an input gap, not fixable there (icd-mapping-strategy.md, "Measuring
accuracy"). Answers are ingested into ``config.CAUSE_STORE_CSV``.

This module does not build the verdict table itself — that comes from scoring
gold-eval (``evaluation/gold_eval.py``, a later work package) — it only turns
misses into a sheet and ingests the answers.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

import config
import io_utils
from manual_audit import sheets

VALID_METHODS = frozenset({"silver", "bronze"})
VALID_INPUT_SUPPORTS = frozenset({"yes", "no"})

CAUSE_STORE_FIELDS = ["case_id", "gold_code", "method", "source_version", "input_supports", "reviewer", "reviewed_at"]

# The one fill-in column on the misses sheet.
_FILL_IN_COLUMN = "input_supports"
_SHEET_COLUMNS = ["case_id", "gold_code", "method", "source_version", _FILL_IN_COLUMN]


class CausePassError(Exception):
    """A cause-pass sheet build or ingest was refused."""


def build_misses_sheet(verdicts: pd.DataFrame, out_csv: str | Path) -> dict:
    """Write the misses sheet from a verdict table of missed gold codes.

    ``verdicts`` must have columns ``case_id, gold_code, method, source_version``;
    ``method`` must be ``silver`` or ``bronze``. Adds one blank ``input_supports``
    fill-in column — the sheet is not blind (it already names the miss being
    diagnosed), it just has nothing else for the specialist to fill in.
    """
    for column in ("case_id", "gold_code", "method", "source_version"):
        if column not in verdicts.columns:
            raise CausePassError(f"verdict table is missing required column {column!r}")
    bad_method = sorted(set(verdicts.loc[~verdicts["method"].isin(VALID_METHODS), "method"]))
    if bad_method:
        raise CausePassError(f"verdict table has method(s) outside {sorted(VALID_METHODS)}: {bad_method}")

    rows = [
        {**{c: str(record[c]) for c in _SHEET_COLUMNS if c != _FILL_IN_COLUMN}, _FILL_IN_COLUMN: ""}
        for record in verdicts.to_dict("records")
    ]
    sheets.write_csv(out_csv, _SHEET_COLUMNS, rows)
    return {"rows": len(rows)}


def ingest_cause_sheet(
    filled_csv: str | Path,
    *,
    reviewer: str,
    out_csv: str | Path | None = None,
) -> dict:
    """Merge a filled misses sheet into the cause store.

    Validates every ``input_supports`` is ``yes``/``no`` (case-insensitive);
    aborts the whole ingest, writing nothing, if any row fails. Cumulative and
    keyed on (case_id, gold_code, method): re-ingesting replaces those rows.
    """
    out_csv = out_csv if out_csv is not None else config.CAUSE_STORE_CSV
    rows = sheets.read_csv(filled_csv)
    errors: list[str] = []
    out_rows: list[dict] = []
    today = date.today().isoformat()

    for idx, row in enumerate(rows, start=2):  # +1 header
        answer = (row.get(_FILL_IN_COLUMN) or "").strip().lower()
        if answer not in VALID_INPUT_SUPPORTS:
            errors.append(
                f"Row {idx}: input_supports={row.get(_FILL_IN_COLUMN)!r} is not one of "
                f"{sorted(VALID_INPUT_SUPPORTS)} (case_id={row.get('case_id')})"
            )
            continue
        out_rows.append({
            "case_id": row.get("case_id", ""),
            "gold_code": row.get("gold_code", ""),
            "method": row.get("method", ""),
            "source_version": row.get("source_version", ""),
            "input_supports": answer,
            "reviewer": reviewer,
            "reviewed_at": today,
        })

    if errors:
        raise CausePassError("cause-sheet ingest aborted due to validation failures:\n" + "\n".join(errors))

    out_path = Path(out_csv)
    try:
        existing = sheets.read_existing_store(out_path, CAUSE_STORE_FIELDS)
    except sheets.StoreSchemaError as error:
        raise CausePassError(str(error)) from error
    store = {(r["case_id"], r["gold_code"], r["method"]): r for r in existing.to_dict("records")}
    added = replaced = 0
    for row in out_rows:
        key = (row["case_id"], row["gold_code"], row["method"])
        if key in store:
            replaced += 1
        else:
            added += 1
        store[key] = row

    out_path.parent.mkdir(parents=True, exist_ok=True)
    io_utils.write_csv(pd.DataFrame(store.values(), columns=CAUSE_STORE_FIELDS), out_path)
    return {"ingested": len(out_rows), "added": added, "replaced": replaced, "total_rows": len(store)}
