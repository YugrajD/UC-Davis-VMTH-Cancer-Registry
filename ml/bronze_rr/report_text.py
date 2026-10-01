"""The whole report as one text: every text column of report.csv, each non-empty one under its heading.

Unlike report_mapping/sections.py (three PetBERT sections), nothing is dropped or regrouped here.
The retriever and the gate read identical text.
"""

from __future__ import annotations

import pandas as pd

from report_mapping import sections
from report_mapping.sections import clean_text

# Fixed order = report.csv column order.
REPORT_COLUMNS: tuple[str, ...] = (
    "CLINICAL ABSTRACT", "GROSS DESCRIPTION", "HISTOPATHOLOGICAL SUMMARY", "COMMENT",
    "FINAL COMMENT", "ANCILLARY TESTS", "ADDENDUM", "ADDITIONAL INFORMATION",
)


def build_report_text(row: pd.Series) -> str:
    parts = []
    for column in REPORT_COLUMNS:
        text = clean_text(row[column])
        if text:
            parts.append(f"{column}: {text}")
    return "\n".join(parts)


def build_report_texts(reports: pd.DataFrame) -> dict[str, str]:
    """{case_id: report text}; one report.csv row is one case."""
    texts = reports.apply(build_report_text, axis=1)
    return dict(zip(reports["case_id"].map(clean_text), texts))


def cases_with_text(reports: pd.DataFrame) -> set[str]:
    """Cases production would write rows for: the merged text of the three PetBERT sections is non-empty
    (report_mapping.inference.stages writes no row for an "empty" case). A case that only has e.g. a
    CLINICAL ABSTRACT is empty here too, so row counts stay comparable with the report_mapping baseline."""
    merged = sections.merged_texts(sections.build_section_frame(reports))
    return {clean_text(c) for c, text in zip(reports["case_id"], merged) if text}
