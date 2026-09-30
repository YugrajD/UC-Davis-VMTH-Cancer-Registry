"""The whole report as one text: every text column of report.csv, each non-empty one under its heading.

Unlike report_mapping/sections.py (three PetBERT sections), nothing is dropped or regrouped here.
Used by both the retriever (ModernBERT) and the OpenJev reranker so they read identical text.
"""

from __future__ import annotations

import pandas as pd

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
