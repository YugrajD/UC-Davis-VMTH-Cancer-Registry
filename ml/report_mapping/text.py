"""The whole report as one text: every text column of report.csv, each non-empty one under its heading.

Nothing is dropped or regrouped. A missing column counts as empty, so a report with only some of the
columns (an upload) is read the same way as a report.csv row.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import config
import io_utils

if TYPE_CHECKING:
    import pandas as pd

# Bump when the text fed to the encoder changes (columns, order, headings, cleaning): a generation records it
# and refuses to load against different text (generation.verify_spec).
TEXT_SPEC_VERSION = 1

# Fixed order = report.csv column order.
REPORT_COLUMNS: tuple[str, ...] = (
    "CLINICAL ABSTRACT", "GROSS DESCRIPTION", "HISTOPATHOLOGICAL SUMMARY", "COMMENT",
    "FINAL COMMENT", "ANCILLARY TESTS", "ADDENDUM", "ADDITIONAL INFORMATION",
)


def clean_text(value: object) -> str:
    """Normalize a raw cell value to a stripped string ("" for missing/NaN)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def build_report_text(row: pd.Series) -> str:
    parts = []
    for column in REPORT_COLUMNS:
        text = clean_text(row.get(column))
        if text:
            parts.append(f"{column}: {text}")
    return "\n".join(parts)


def build_report_texts(reports: pd.DataFrame) -> list[str]:
    """One text per report row, in row order."""
    return [build_report_text(row) for _, row in reports.iterrows()]


def has_text(row: pd.Series) -> bool:
    """A case has text iff any of the 8 columns is non-empty; only such cases get prediction rows."""
    return any(clean_text(row.get(column)) for column in REPORT_COLUMNS)


def read_reports() -> pd.DataFrame:
    """report.csv (latin-1) with ``case_id`` cleaned. One row is one case."""
    reports = io_utils.read_csv(config.REPORT_CSV, encoding="latin-1")
    reports["case_id"] = reports["case_id"].map(clean_text)
    return reports
