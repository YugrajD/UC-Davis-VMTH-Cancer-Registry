"""handoff/contracts.py: the sidecar (schema_version, sha256) round trip."""

from __future__ import annotations

import pytest

from handoff import contracts


def test_write_and_verify_sidecar_round_trip(tmp_path):
    csv_path = tmp_path / "silver_codes_s1.csv"
    csv_path.write_text("case_id,code\nCASE-A,1001/3\n", encoding="utf-8")

    sidecar_path = contracts.write_sidecar(csv_path, kind=contracts.SILVER_EXPORT_KIND, schema_version=1)
    assert sidecar_path.is_file()

    sidecar = contracts.verify_sidecar(csv_path, expected_kind=contracts.SILVER_EXPORT_KIND)
    assert sidecar["schema_version"] == 1
    assert sidecar["kind"] == contracts.SILVER_EXPORT_KIND
    assert "sha256" in sidecar and "written_at" in sidecar


def test_verify_sidecar_detects_tampering(tmp_path):
    csv_path = tmp_path / "combined_predictions_run1.csv"
    csv_path.write_text("case_id,code\nCASE-A,1001/3\n", encoding="utf-8")
    contracts.write_sidecar(csv_path, kind=contracts.COMBINED_PREDICTIONS_EXPORT_KIND, schema_version=1)

    csv_path.write_text("case_id,code\nCASE-A,9999/0\n", encoding="utf-8")  # changed after the sidecar was written
    with pytest.raises(contracts.SidecarError, match="sha256"):
        contracts.verify_sidecar(csv_path)


def test_verify_sidecar_rejects_wrong_kind(tmp_path):
    csv_path = tmp_path / "review_queue_run1.csv"
    csv_path.write_text("case_id,reason\nCASE-A,vague_silver\n", encoding="utf-8")
    contracts.write_sidecar(csv_path, kind=contracts.REVIEW_QUEUE_EXPORT_KIND, schema_version=1)

    with pytest.raises(contracts.SidecarError, match="kind"):
        contracts.verify_sidecar(csv_path, expected_kind=contracts.COMBINED_PREDICTIONS_EXPORT_KIND)


def test_read_sidecar_missing_raises(tmp_path):
    csv_path = tmp_path / "no_sidecar.csv"
    csv_path.write_text("case_id\nCASE-A\n", encoding="utf-8")
    with pytest.raises(contracts.SidecarError, match="missing"):
        contracts.read_sidecar(csv_path)


def test_silver_export_columns_drop_diagnosis_text():
    from diagnosis_mapping.silver import TEXT_COL

    assert TEXT_COL not in contracts.SILVER_EXPORT_COLUMNS
    assert "silver_generation" in contracts.SILVER_EXPORT_COLUMNS
