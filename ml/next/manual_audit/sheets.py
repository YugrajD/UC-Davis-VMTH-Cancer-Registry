"""Read/write review CSVs and their sidecars — the sheet layer every audit tool uses.

Carries over ``ml/annotation/gold/csv_io.py``'s behaviour: reviewer-facing sheets
are written ``utf-8-sig`` (opens cleanly in Excel/LibreOffice without mangling
accents, and a double-click never mis-renders the BOM as a stray character in
the first header cell), sidecar key/reference tables are plain
``csv.DictReader``/``DictWriter`` round trips, and none of this module knows
anything about sampling, stratification or ingest validation — it only moves
rows in and out of files.

**No sheet writer here has a parameter through which a prediction, a silver
code or a confidence could be passed.** ``write_case_sheet`` goes furthest:
given only case ids and the names of blank fill-in columns, there is no
argument slot for a value to occupy in the first place. This is a structural
property of the CSV this module writes, independent of how the case is
*reviewed* — the eval-batch case sheet (``eval_batch.py``) carries no
prediction column even though the specialist looks each case up in the
registry app, which does show its predictions (a 2026-09-26 user decision;
review is knowingly non-blind there, tracked via the ledger's ``review_mode``).

The Tier-3 audit sheet (``manual_audit/tier3_audit.py``) is the other sheet
that shows a cascade answer, by its original row-level design (see that
module's docstring) — it uses the generic ``write_csv``/``read_csv`` here like
everything else, it just chooses to put a ``Predicted Match`` column in its
header.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

import io_utils

# utf-8-sig: a double-click still opens cleanly in Excel without mangling accents.
ENCODING = "utf-8-sig"


class StoreSchemaError(Exception):
    """An existing cumulative store's columns don't match what the caller expects."""


def write_csv(path: str | Path, header: list[str], rows: list[dict]) -> None:
    """Write a reviewer-facing sheet or sidecar table."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding=ENCODING) as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: str | Path) -> list[dict]:
    """Return a filled sheet or sidecar table as a list of header-keyed dicts (values as str)."""
    with open(path, encoding=ENCODING) as file:
        return [
            {key: ("" if value is None else str(value)) for key, value in row.items()}
            for row in csv.DictReader(file)
        ]


def read_key_rows(path: str | Path, key_column: str = "row_id") -> dict[str, dict]:
    """Return a key/reference sidecar indexed by ``key_column`` (default ``row_id``)."""
    return {row[key_column]: row for row in read_csv(path)}


def write_instructions(path: str | Path, text: str) -> None:
    """Write a reviewer-facing instructions sidecar (plain Markdown, not a CSV)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8", newline="\n")


def write_case_sheet(
    path: str | Path,
    case_ids: list[str],
    pointer_of: dict[str, str],
    fill_in_columns: list[str],
    pointer_column: str = "record_pointer",
) -> None:
    """Write a case-level sheet with only ``case_id`` + a record pointer, plus blank fill-ins.

    There is no parameter here through which a prediction, a silver code or a
    confidence could be threaded onto the page: the only per-case value written
    is ``pointer_of[case_id]``, and every other cell in every fill-in column is
    the empty string. Used by ``eval_batch.py`` for the case-level eval batch.
    """
    header = ["case_id", pointer_column, *fill_in_columns]
    rows = [
        {"case_id": case_id, pointer_column: pointer_of[case_id], **{c: "" for c in fill_in_columns}}
        for case_id in case_ids
    ]
    write_csv(path, header, rows)


def read_existing_store(path: str | Path, expected_columns: list[str]) -> pd.DataFrame:
    """Load a cumulative store CSV for merging, refusing one with unexpected columns.

    Carries over the legacy gold-ingest guard (``ml/annotation/gold/ingest.py``):
    a store whose columns don't match the schema an ingester is about to write
    is a sign of a schema change or the wrong file, and silently merging into
    it would produce a store mixing old and new columns. The caller must move
    it aside (or pass a fresh output path) rather than have ingest guess.
    """
    path = Path(path)
    if not path.is_file():
        return pd.DataFrame(columns=expected_columns)
    df = io_utils.read_csv(path, encoding="utf-8", dtype=str, keep_default_na=False)
    unknown = set(df.columns) - set(expected_columns)
    if unknown:
        raise StoreSchemaError(
            f"existing store {path} has unexpected column(s) {sorted(unknown)}; refusing to merge "
            f"into it — move it aside, or pass a fresh output path to start over."
        )
    return df
