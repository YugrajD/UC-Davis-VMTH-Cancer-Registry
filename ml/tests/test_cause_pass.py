"""manual_audit/cause_pass.py: build the misses sheet, ingest input_supports answers."""

from __future__ import annotations

import pandas as pd
import pytest

from manual_audit import cause_pass, sheets


def _verdicts():
    return pd.DataFrame([
        {"case_id": "CASE-A", "gold_code": "1001/3", "method": "silver", "source_version": "silver-0-legacy"},
        {"case_id": "CASE-B", "gold_code": "2001/3", "method": "bronze", "source_version": "report_mapping-gen0"},
    ])


def test_build_misses_sheet_has_one_blank_fill_in_column(tmp_path):
    out = tmp_path / "misses.csv"
    result = cause_pass.build_misses_sheet(_verdicts(), out_csv=out)
    assert result == {"rows": 2}
    rows = sheets.read_csv(out)
    assert set(rows[0].keys()) == {"case_id", "gold_code", "method", "source_version", "input_supports"}
    assert all(row["input_supports"] == "" for row in rows)
    assert rows[0]["method"] == "silver" and rows[1]["method"] == "bronze"


def test_build_misses_sheet_refuses_unknown_method(tmp_path):
    bad = _verdicts()
    bad.loc[0, "method"] = "gold"
    with pytest.raises(cause_pass.CausePassError, match="gold"):
        cause_pass.build_misses_sheet(bad, out_csv=tmp_path / "misses.csv")


def test_ingest_cause_sheet_round_trip(tmp_path):
    misses_csv = tmp_path / "misses.csv"
    cause_pass.build_misses_sheet(_verdicts(), out_csv=misses_csv)
    rows = sheets.read_csv(misses_csv)
    rows[0]["input_supports"] = "Yes"
    rows[1]["input_supports"] = "no"
    sheets.write_csv(misses_csv, list(rows[0].keys()), rows)

    out = tmp_path / "cause_store.csv"
    result = cause_pass.ingest_cause_sheet(misses_csv, reviewer="Dr. Test", out_csv=out)
    assert result == {"ingested": 2, "added": 2, "replaced": 0, "total_rows": 2}
    store = sheets.read_csv(out)
    assert {r["case_id"]: r["input_supports"] for r in store} == {"CASE-A": "yes", "CASE-B": "no"}
    assert all(r["reviewer"] == "Dr. Test" for r in store)


def test_ingest_cause_sheet_refuses_invalid_answer(tmp_path):
    misses_csv = tmp_path / "misses.csv"
    cause_pass.build_misses_sheet(_verdicts(), out_csv=misses_csv)
    rows = sheets.read_csv(misses_csv)
    rows[0]["input_supports"] = "maybe"
    rows[1]["input_supports"] = "no"
    sheets.write_csv(misses_csv, list(rows[0].keys()), rows)

    out = tmp_path / "cause_store.csv"
    with pytest.raises(cause_pass.CausePassError, match="maybe"):
        cause_pass.ingest_cause_sheet(misses_csv, reviewer="Dr. Test", out_csv=out)
    assert not out.exists()


def test_ingest_refuses_unknown_columns_in_existing_store(tmp_path):
    import io_utils

    misses_csv = tmp_path / "misses.csv"
    cause_pass.build_misses_sheet(_verdicts(), out_csv=misses_csv)
    rows = sheets.read_csv(misses_csv)
    rows[0]["input_supports"] = "yes"
    rows[1]["input_supports"] = "no"
    sheets.write_csv(misses_csv, list(rows[0].keys()), rows)

    out = tmp_path / "cause_store.csv"
    io_utils.write_csv(pd.DataFrame([{"case_id": "X", "unexpected_column": "y"}]), out)
    with pytest.raises(cause_pass.CausePassError, match="unexpected"):
        cause_pass.ingest_cause_sheet(misses_csv, reviewer="Dr. Test", out_csv=out)


def test_reingest_replaces_not_duplicates(tmp_path):
    misses_csv = tmp_path / "misses.csv"
    cause_pass.build_misses_sheet(_verdicts(), out_csv=misses_csv)
    rows = sheets.read_csv(misses_csv)
    rows[0]["input_supports"] = "yes"
    rows[1]["input_supports"] = "no"
    sheets.write_csv(misses_csv, list(rows[0].keys()), rows)
    out = tmp_path / "cause_store.csv"
    cause_pass.ingest_cause_sheet(misses_csv, reviewer="Dr. Test", out_csv=out)

    rows[0]["input_supports"] = "no"
    sheets.write_csv(misses_csv, list(rows[0].keys()), rows)
    result = cause_pass.ingest_cause_sheet(misses_csv, reviewer="Dr. Test", out_csv=out)
    assert result == {"ingested": 2, "added": 0, "replaced": 2, "total_rows": 2}
    store = sheets.read_csv(out)
    assert len(store) == 2
    assert {r["case_id"]: r["input_supports"] for r in store}["CASE-A"] == "no"
