"""handoff/exports.py: silver/coding exports and the report-mapping bundle."""

from __future__ import annotations

import shutil
import tarfile
from pathlib import Path

import pandas as pd
import pytest

import config
import io_utils
from coding.combine import COMBINED_PREDICTIONS_COLUMNS
from coding.queue import REVIEW_QUEUE_COLUMNS
from generations.manifest import read_manifest, update_manifest
from handoff import contracts, exports
from report_mapping.generation import GenerationError

from . import fixtures as fx


@pytest.fixture
def handoff_root(monkeypatch, tmp_path):
    fx.point_handoff_config_at(monkeypatch, tmp_path)
    return tmp_path


def _make_stamped_silver_generation(silver_id: str, rows: list[tuple]) -> str:
    """Like fx.make_silver_generation, but with the ``silver_generation`` column
    every real silver.run() output carries (fx.make_silver_generation itself
    omits it — it's a generic helper other WPs' tests don't need it for)."""
    df = pd.DataFrame(rows, columns=fx.ANNOTATION_COLUMNS)
    df["silver_generation"] = silver_id
    directory = config.SILVER_DIR / silver_id
    directory.mkdir(parents=True, exist_ok=True)
    io_utils.write_csv(df, directory / "annotation.csv")
    from generations.manifest import write_manifest
    write_manifest(directory, {"silver_id": silver_id, "llm_enabled": True})
    return silver_id


def test_export_silver_drops_diagnosis_text(handoff_root):
    _make_stamped_silver_generation("s1", [
        ("CASE-A", 1, "MAST CELL TUMOR TEXT", "Mast cell tumor, malignant", "Mast Cell Tumors",
         "1001/3", "mast cell tumor", "Exact", 1.0, "tier1_exact"),
    ])

    out_path = exports.export_silver("s1")

    assert out_path == config.HANDOFF_OUTBOX_DIR / "silver_codes_s1.csv"
    df = io_utils.read_csv(out_path, encoding="utf-8", dtype=str, keep_default_na=False)
    assert list(df.columns) == contracts.SILVER_EXPORT_COLUMNS
    assert "diagnosis" not in df.columns
    assert df.iloc[0]["matched_code"] == "1001/3"
    sidecar = contracts.verify_sidecar(out_path, expected_kind=contracts.SILVER_EXPORT_KIND)
    assert sidecar["schema_version"] == contracts.SILVER_EXPORT_SCHEMA_VERSION


def test_export_coding_writes_both_files(handoff_root):
    combined = pd.DataFrame([{c: "" for c in COMBINED_PREDICTIONS_COLUMNS}])
    combined.loc[0, ["case_id", "code"]] = ["CASE-A", "1001/3"]
    config.COMBINED_PREDICTIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    io_utils.write_csv(combined, config.COMBINED_PREDICTIONS_CSV)

    queue = pd.DataFrame([{c: "" for c in REVIEW_QUEUE_COLUMNS}])
    queue.loc[0, ["case_id", "reason"]] = ["CASE-B", "vague_silver"]
    config.REVIEW_QUEUE_CSV.parent.mkdir(parents=True, exist_ok=True)
    io_utils.write_csv(queue, config.REVIEW_QUEUE_CSV)

    result = exports.export_coding("run1")

    assert result["combined_predictions_path"] == config.HANDOFF_OUTBOX_DIR / "combined_predictions_run1.csv"
    assert result["review_queue_path"] == config.HANDOFF_OUTBOX_DIR / "review_queue_run1.csv"
    assert result["combined_rows"] == 1 and result["review_queue_rows"] == 1
    contracts.verify_sidecar(result["combined_predictions_path"], expected_kind=contracts.COMBINED_PREDICTIONS_EXPORT_KIND)
    contracts.verify_sidecar(result["review_queue_path"], expected_kind=contracts.REVIEW_QUEUE_EXPORT_KIND)


def test_export_coding_missing_column_refused(handoff_root):
    bad = pd.DataFrame([{"case_id": "CASE-A"}])  # missing every other required column
    config.COMBINED_PREDICTIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    io_utils.write_csv(bad, config.COMBINED_PREDICTIONS_CSV)
    config.REVIEW_QUEUE_CSV.parent.mkdir(parents=True, exist_ok=True)
    io_utils.write_csv(pd.DataFrame([{c: "" for c in REVIEW_QUEUE_COLUMNS}]), config.REVIEW_QUEUE_CSV)

    with pytest.raises(exports.HandoffExportError, match="missing required column"):
        exports.export_coding("run1")


def _current(tiny_generation_dir, generation_id="gen-bundle"):
    shutil.copytree(tiny_generation_dir, config.REPORT_MAPPING_CURRENT_DIR)
    update_manifest(config.REPORT_MAPPING_CURRENT_DIR, {"generation_id": generation_id})
    return config.REPORT_MAPPING_CURRENT_DIR


def test_export_bundle_and_verify_round_trip(handoff_root, tiny_generation_dir):
    _current(tiny_generation_dir)

    result = exports.export_bundle("current")

    assert result["tarball_path"].is_file()
    assert result["sha256_path"].is_file()
    assert result["generation_id"] == "gen-bundle"
    with tarfile.open(result["tarball_path"]) as tar:
        names = {Path(name).relative_to("current").as_posix() for name in tar.getnames() if name != "current"}
    assert {"manifest.json", "head.pt", "gate_head.pt", "terms.json", "term_info.json", "thresholds.json",
            "encoder/model.safetensors"} <= names

    verify_result = exports.verify_bundle(result["tarball_path"])
    assert verify_result == {"sha256_verified": True, "generation_id": "gen-bundle"}


def test_export_bundle_refuses_a_tampered_generation(handoff_root, tiny_generation_dir):
    directory = _current(tiny_generation_dir)
    (directory / "thresholds.json").write_text('{"tampered": true}', encoding="utf-8")

    with pytest.raises(GenerationError, match="does not match"):
        exports.export_bundle("current")
    assert not config.HANDOFF_BUNDLES_DIR.exists()


def test_export_bundle_refuses_an_uncalibrated_or_other_architecture_generation(handoff_root, tiny_generation_dir):
    directory = _current(tiny_generation_dir)
    update_manifest(directory, {"calibration": {"status": "pending"}})
    with pytest.raises(GenerationError, match="calibration.status"):
        exports.export_bundle("current")
    update_manifest(directory, {"calibration": read_manifest(tiny_generation_dir)["calibration"],
                                "architecture": {"name": "petbert-four-stage"}})
    with pytest.raises(GenerationError, match="architecture"):
        exports.export_bundle("current")


def test_verify_bundle_detects_checksum_mismatch(handoff_root, tiny_generation_dir):
    _current(tiny_generation_dir)
    result = exports.export_bundle("current")
    result["tarball_path"].write_bytes(b"corrupted")

    with pytest.raises(exports.HandoffExportError, match="sha256"):
        exports.verify_bundle(result["tarball_path"])
