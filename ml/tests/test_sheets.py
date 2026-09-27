"""manual_audit/sheets.py: generic CSV round trip, the case-sheet writer, the store-schema guard."""

from __future__ import annotations

import pytest

from manual_audit import sheets


def test_write_read_round_trip(tmp_path):
    path = tmp_path / "sheet.csv"
    header = ["row_id", "value"]
    rows = [{"row_id": "1", "value": "a"}, {"row_id": "2", "value": ""}]
    sheets.write_csv(path, header, rows)
    assert sheets.read_csv(path) == rows


def test_read_key_rows_indexes_on_row_id(tmp_path):
    path = tmp_path / "key.csv"
    sheets.write_csv(path, ["row_id", "case_id"], [{"row_id": "7", "case_id": "CASE-0007"}])
    keyed = sheets.read_key_rows(path)
    assert keyed == {"7": {"row_id": "7", "case_id": "CASE-0007"}}


def test_write_instructions_writes_plain_text(tmp_path):
    path = tmp_path / "instructions.md"
    sheets.write_instructions(path, "# Hello\n")
    assert path.read_text(encoding="utf-8") == "# Hello\n"


def test_case_sheet_has_no_prediction_column(tmp_path):
    path = tmp_path / "sheet.csv"
    case_ids = ["CASE-0001", "CASE-0002"]
    sheets.write_case_sheet(
        path, case_ids, {c: c for c in case_ids}, ["term_1", "term_2", "no_cancer", "reviewer", "notes"],
    )
    rows = sheets.read_csv(path)
    header = set(rows[0].keys())
    # Nothing that could carry a prediction, silver code, or confidence exists
    # in the header at all -- there was never a parameter to pass one through.
    for forbidden in ("prediction", "predicted", "silver", "bronze", "confidence", "matched_term", "matched_group"):
        assert not any(forbidden in col.lower() for col in header)
    assert header == {"case_id", "record_pointer", "term_1", "term_2", "no_cancer", "reviewer", "notes"}
    for row in rows:
        assert row["record_pointer"] == row["case_id"]
        for blank_col in ("term_1", "term_2", "no_cancer", "reviewer", "notes"):
            assert row[blank_col] == ""


def test_read_existing_store_missing_file_returns_empty_frame(tmp_path):
    df = sheets.read_existing_store(tmp_path / "nope.csv", ["a", "b"])
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 0


def test_read_existing_store_refuses_unknown_columns(tmp_path):
    import io_utils
    import pandas as pd

    path = tmp_path / "store.csv"
    io_utils.write_csv(pd.DataFrame([{"a": "1", "unexpected": "x"}]), path)
    with pytest.raises(sheets.StoreSchemaError, match="unexpected"):
        sheets.read_existing_store(path, ["a", "b"])
