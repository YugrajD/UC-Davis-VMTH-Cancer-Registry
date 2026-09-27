"""handoff/imports.py: pending-diagnoses landing + merge, and gold import delegation."""

from __future__ import annotations

import csv

import pandas as pd
import pytest

import config
import io_utils
from handoff import imports as handoff_imports
from manual_audit.gold import GoldIngestError

from . import fixtures as fx


@pytest.fixture
def handoff_root(monkeypatch, tmp_path):
    fx.point_handoff_config_at(monkeypatch, tmp_path)
    return tmp_path


def _write_pending_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["case_id", "diagnosis_number", "diagnosis"])
        writer.writerows(rows)
    return path


def test_import_pending_diagnoses_lands_and_merges(handoff_root):
    csv_path = _write_pending_csv(handoff_root / "export1.csv", [("CASE-A", 1, "MAST CELL TUMOR")])
    result = handoff_imports.import_pending_diagnoses(csv_path, "export-1")

    assert result == {
        "export_id": "export-1", "imported_rows": 1, "imported_cases": 1,
        "replaced_cases": 0, "merged_total_rows": 1, "merged_total_cases": 1,
    }
    landed = config.HANDOFF_INBOX_DIR / "pending_diagnoses_export-1.csv"
    assert landed.is_file()
    assert (config.HANDOFF_INBOX_DIR / "pending_diagnoses_export-1.csv.manifest.json").is_file()
    merged = io_utils.read_csv(config.HANDOFF_PENDING_DIAGNOSES_CSV, encoding="utf-8", dtype=str, keep_default_na=False)
    assert list(merged["case_id"]) == ["CASE-A"]


def test_import_pending_diagnoses_later_export_replaces_case(handoff_root):
    csv_path1 = _write_pending_csv(handoff_root / "export1.csv", [("CASE-A", 1, "MAST CELL TUMOR")])
    handoff_imports.import_pending_diagnoses(csv_path1, "export-1")

    csv_path2 = _write_pending_csv(handoff_root / "export2.csv", [("CASE-A", 1, "CORRECTED DIAGNOSIS"), ("CASE-B", 1, "FIBROSARCOMA")])
    result = handoff_imports.import_pending_diagnoses(csv_path2, "export-2")

    assert result["replaced_cases"] == 1
    assert result["merged_total_cases"] == 2
    merged = io_utils.read_csv(config.HANDOFF_PENDING_DIAGNOSES_CSV, encoding="utf-8", dtype=str, keep_default_na=False)
    row_a = merged[merged["case_id"] == "CASE-A"].iloc[0]
    assert row_a["diagnosis"] == "CORRECTED DIAGNOSIS"  # old CASE-A row is gone, not duplicated
    assert len(merged) == 2


def test_import_pending_diagnoses_requires_export_id(handoff_root):
    csv_path = _write_pending_csv(handoff_root / "export1.csv", [("CASE-A", 1, "MAST CELL TUMOR")])
    with pytest.raises(handoff_imports.HandoffImportError, match="export_id"):
        handoff_imports.import_pending_diagnoses(csv_path, "")


def test_import_pending_diagnoses_missing_column_refused(handoff_root):
    csv_path = handoff_root / "bad.csv"
    csv_path.write_text("case_id\nCASE-A\n", encoding="utf-8")
    with pytest.raises(handoff_imports.HandoffImportError, match="diagnosis"):
        handoff_imports.import_pending_diagnoses(csv_path, "export-1")


def test_import_pending_diagnoses_blank_case_id_refused(handoff_root):
    csv_path = _write_pending_csv(handoff_root / "bad.csv", [("", 1, "MAST CELL TUMOR")])
    with pytest.raises(handoff_imports.HandoffImportError, match="case_id"):
        handoff_imports.import_pending_diagnoses(csv_path, "export-1")


def _write_gold_csv(path, rows):
    df = pd.DataFrame(rows)
    io_utils.write_csv(df, path)
    return path


def test_import_gold_delegates_to_ingest_gold(monkeypatch, handoff_root, labels_csv):
    monkeypatch.setattr(config, "LABELS_CSV", labels_csv)  # ingest_gold defaults to config.LABELS_CSV
    ledger = handoff_root / "ledger.csv"
    io_utils.write_csv(pd.DataFrame({"case_id": ["CASE-A"]}), ledger)
    monkeypatch.setattr(config, "EVAL_BATCH_LEDGER_CSV", ledger)

    csv_path = _write_gold_csv(handoff_root / "gold_export1.csv", [
        {"case_id": "CASE-A", "term": "Mast cell tumor, malignant", "origin": "eval_batch"},
    ])
    result = handoff_imports.import_gold(csv_path, "gold-export-1", reviewer="Dr. Test")

    assert result["added_cases"] == 1
    assert result["export_id"] == "gold-export-1"
    landed = config.HANDOFF_INBOX_DIR / "gold_gold-export-1.csv"
    assert landed.is_file()
    assert (config.HANDOFF_INBOX_DIR / "gold_gold-export-1.csv.manifest.json").is_file()


def test_import_gold_requires_export_id(handoff_root, labels_csv):
    csv_path = _write_gold_csv(handoff_root / "gold_export1.csv", [
        {"case_id": "CASE-A", "term": "Mast cell tumor, malignant", "origin": "eval_batch"},
    ])
    with pytest.raises(handoff_imports.HandoffImportError, match="export_id"):
        handoff_imports.import_gold(csv_path, "", reviewer="Dr. Test")


def test_import_gold_missing_origin_column_refused_before_ingest(handoff_root):
    csv_path = handoff_root / "gold_export1.csv"
    csv_path.write_text("case_id,term\nCASE-A,Mast cell tumor\n", encoding="utf-8")
    with pytest.raises(handoff_imports.HandoffImportError, match="origin"):
        handoff_imports.import_gold(csv_path, "gold-export-1", reviewer="Dr. Test")
    # Refused before landing: no raw copy for a rejected import.
    assert not (config.HANDOFF_INBOX_DIR / "gold_gold-export-1.csv").exists()


def test_import_gold_invalid_origin_propagates_and_is_not_landed(monkeypatch, handoff_root, labels_csv):
    monkeypatch.setattr(config, "LABELS_CSV", labels_csv)
    csv_path = _write_gold_csv(handoff_root / "gold_export1.csv", [
        {"case_id": "CASE-A", "term": "Mast cell tumor, malignant", "origin": "manual"},
    ])
    with pytest.raises(GoldIngestError, match="origin"):
        handoff_imports.import_gold(csv_path, "gold-export-1", reviewer="Dr. Test")
    assert not (config.HANDOFF_INBOX_DIR / "gold_gold-export-1.csv").exists()
