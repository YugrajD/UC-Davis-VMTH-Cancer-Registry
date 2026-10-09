"""report_mapping/text.py: the whole report as one text, the has-text rule, and report.csv reading."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import config
from report_mapping import text

from . import fixtures as fx

ALL_COLUMNS = ["CLINICAL ABSTRACT", "GROSS DESCRIPTION", "HISTOPATHOLOGICAL SUMMARY", "COMMENT",
               "FINAL COMMENT", "ANCILLARY TESTS", "ADDENDUM", "ADDITIONAL INFORMATION"]


def _reference_text(row: pd.Series) -> str:
    """The retrieve-and-rerank experiment's builder (bronze_rr.report_text.build_report_text), restated."""
    parts = []
    for column in ALL_COLUMNS:
        value = row[column]
        cell = "" if value is None or (isinstance(value, float) and math.isnan(value)) else str(value).strip()
        if cell:
            parts.append(f"{column}: {cell}")
    return "\n".join(parts)


def test_columns_are_the_eight_report_columns_in_report_csv_order():
    assert list(text.REPORT_COLUMNS) == ALL_COLUMNS == fx.REPORT_COLUMNS[1:]


def test_text_is_byte_identical_to_the_experiment_builder_for_rows_with_all_columns():
    rows = pd.DataFrame([
        dict(zip(ALL_COLUMNS, ["  abstract  ", "gross", np.nan, "comment\nwith a newline", None, "  ", 42, "extra"])),
        dict(zip(ALL_COLUMNS, [""] * 8)),
        dict(zip(ALL_COLUMNS, ["a", "b", "c", "d", "e", "f", "g", "h"])),
    ])
    for _, row in rows.iterrows():
        assert text.build_report_text(row) == _reference_text(row)
    assert text.build_report_text(rows.iloc[0]) == (
        "CLINICAL ABSTRACT: abstract\nGROSS DESCRIPTION: gross\nCOMMENT: comment\nwith a newline\n"
        "ADDENDUM: 42\nADDITIONAL INFORMATION: extra")
    assert text.build_report_text(rows.iloc[1]) == ""


def test_a_missing_column_counts_as_empty():
    row = pd.Series({"case_id": "c", "COMMENT": "x", "CLINICAL ABSTRACT": "y"})
    assert text.build_report_text(row) == "CLINICAL ABSTRACT: y\nCOMMENT: x"
    assert text.build_report_text(pd.Series({"case_id": "c"})) == ""


def test_build_report_texts_is_one_text_per_row_in_row_order_and_keeps_repeated_case_ids():
    reports = pd.DataFrame({"case_id": ["a", "a", "b"], "COMMENT": ["one", "two", ""]})
    assert text.build_report_texts(reports) == ["COMMENT: one", "COMMENT: two", ""]


def test_a_case_has_text_iff_any_of_the_eight_columns_is_non_empty():
    def row(**cells):
        return pd.Series({"case_id": "c", **cells})

    assert text.has_text(row(**{"CLINICAL ABSTRACT": "x"}))        # the PetBERT rule dropped this case
    assert text.has_text(row(**{"ADDITIONAL INFORMATION": "x"}))
    assert text.has_text(row(COMMENT="x"))
    assert not text.has_text(row(COMMENT="   ", ADDENDUM=np.nan))
    assert not text.has_text(row())


def test_clean_text():
    assert text.clean_text(None) == "" and text.clean_text(float("nan")) == "" and text.clean_text("  a ") == "a"
    assert text.clean_text(7) == "7"


def test_read_reports_reads_latin1_and_cleans_the_case_id(tmp_path, monkeypatch):
    path = tmp_path / "report.csv"
    path.write_bytes("case_id,COMMENT\n  C1 ,caf\xe9\nC2,\n".encode("latin-1"))
    monkeypatch.setattr(config, "REPORT_CSV", path)
    reports = text.read_reports()
    assert reports["case_id"].tolist() == ["C1", "C2"]
    assert text.build_report_texts(reports)[0] == "COMMENT: caf\xe9"
