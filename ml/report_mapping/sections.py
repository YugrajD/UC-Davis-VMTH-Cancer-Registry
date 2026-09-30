"""The single concat-3 section spec and report-text builder.

Legacy duplicated this spec in two places — ``production/petbert_pipeline/
pipeline.py:54-58`` (``CONCAT_3_SECTIONS``) and ``training/contrastive/
build_contrastive_dataset.py:21-25`` (``_PER_SECTION_GROUPS``). ``tests/
test_sections.py`` imports both by path and asserts they equal
``CONCAT_3_SECTIONS`` below — see that test for the comparison result. They
agree: both are exactly the three groups below, in the same order.

Three per-row synthetic sections, each fed to PetBERT independently and mean-
pooled, then concatenated into a single 2304-dim (3 x 768) case embedding:

  __sec_0__ = HISTOPATHOLOGICAL SUMMARY
  __sec_1__ = FINAL COMMENT + "\\n" + COMMENT
  __sec_2__ = ANCILLARY TESTS

``SECTION_SPEC_VERSION`` is part of a generation's embedding fingerprint
(report_mapping/model/generation.py): bump it whenever this spec changes
(section grouping or field selection) so stale classifiers refuse to load
silently against differently-built embeddings (CLAUDE.md, "Embedding &
Classifier Versioning").
"""

from __future__ import annotations

import math
from typing import Iterable

import pandas as pd

SECTION_SPEC_VERSION = 1

CONCAT_3_SECTIONS: tuple[tuple[str, ...], ...] = (
    ("HISTOPATHOLOGICAL SUMMARY",),
    ("FINAL COMMENT", "COMMENT"),
    ("ANCILLARY TESTS",),
)

SECTION_COLUMNS: tuple[str, ...] = tuple(f"__sec_{i}__" for i in range(len(CONCAT_3_SECTIONS)))

# Alias under which the 2304-dim per-row concat of the three section
# embeddings is stored (embedding_cache.py, training NPZs) — hardcoded by
# downstream trainers, so it must stay stable.
CONCAT_3_KEY = "concat_3"


def clean_text(value: object) -> str:
    """Normalize a raw cell value to a stripped string ("" for missing/NaN)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def build_section_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``dataframe`` with the three ``__sec_N__`` columns added.

    Each section column joins its source column(s) with "\\n" (a single-column
    section is unaffected by the join). Missing cells become "". Raises if a
    source column is absent.
    """
    dataframe = dataframe.copy()
    for i, group in enumerate(CONCAT_3_SECTIONS):
        missing = [c for c in group if c not in dataframe.columns]
        if missing:
            raise ValueError(
                f"Missing source columns for section {i} ({group!r}): {missing!r}. "
                f"Available: {dataframe.columns.tolist()}"
            )
        dataframe[SECTION_COLUMNS[i]] = (
            dataframe[list(group)].fillna("").astype(str).agg("\n".join, axis=1)
        )
    return dataframe


def section_texts(dataframe: pd.DataFrame) -> dict[str, list[str]]:
    """{section_column: [cleaned text, ...]} for each of the three section columns.

    ``dataframe`` must already have the section columns (``build_section_frame``).
    """
    return {col: dataframe[col].map(clean_text).tolist() for col in SECTION_COLUMNS}


def merge_report_columns(row: pd.Series, columns: Iterable[str] = SECTION_COLUMNS) -> str:
    """Merge named section columns into one ``[COLUMN] text`` string for keyword filters.

    Each non-empty column is prefixed with its own name so the report text fed
    to keyword_correction.py distinguishes sections; empty cells are skipped.
    """
    parts = []
    for col in columns:
        val = clean_text(row.get(col, ""))
        if val:
            parts.append(f"[{col}] {val}")
    return " ".join(parts)


def merged_texts(dataframe: pd.DataFrame) -> list[str]:
    """Per-row merged text (``merge_report_columns`` over every row), used for
    keyword correction and the Lipoma rescue."""
    return dataframe.apply(merge_report_columns, axis=1).tolist()
