"""silver.py: cascade orchestration, versioned silver generations (write-once,
manifest verification), and the legacy import.
"""

from __future__ import annotations

import pytest

import config
import io_utils
from generations.manifest import ManifestError, read_manifest
from taxonomy.taxonomy import load_labels_taxonomy
from diagnosis_mapping import llm_client, silver

from . import fixtures as fx


@pytest.fixture
def dm_env(tmp_path, monkeypatch):
    fx.point_diagnosis_mapping_config_at(monkeypatch, tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# run_cascade: orchestration over a diagnoses-shaped dataframe
# ---------------------------------------------------------------------------


def test_run_cascade_produces_expected_stages(diagnoses_csv, labels_csv):
    df = io_utils.read_csv(diagnoses_csv)
    labels = load_labels_taxonomy(str(labels_csv))
    out_df, counters = silver.run_cascade(df, labels, silver.CascadeConfig(llm_enabled=False))

    assert list(out_df.columns) == silver.ANNOTATION_COLUMNS
    stages = dict(zip(out_df["case_id"], out_df["decision_stage"]))
    methods = dict(zip(out_df["case_id"], out_df["method"]))
    # From fixtures.make_diagnoses_csv: exact keyword hits for the taxonomy
    # terms, "NO EVIDENCE OF NEOPLASIA" has no cancer signal at all.
    assert stages["CASE-0001"] == "tier1_exact" and methods["CASE-0001"] == "Exact"  # MAST CELL TUMOR
    assert stages["CASE-0002"] == "no_signal" and methods["CASE-0002"] == "No Match"  # NO EVIDENCE OF NEOPLASIA
    assert stages["CASE-0003"] == "tier1_exact"  # MAST CELL SARCOMA
    assert stages["CASE-0004"] == "tier1_exact"  # FIBROSARCOMA
    assert stages["CASE-0005"] == "tier1_exact"  # SQUAMOUS CELL CARCINOMA
    assert stages["CASE-0007"] == "tier1_exact"  # MAST CELL TUMOR BENIGN
    assert counters["tier3_calls"] == 0  # llm_enabled=False: no tier3-eligible rows here


# ---------------------------------------------------------------------------
# run(): write a fresh silver generation
# ---------------------------------------------------------------------------


def test_run_writes_annotation_csv_and_manifest(dm_env, diagnoses_csv, labels_csv):
    manifest = silver.run("silver-1", diagnoses_csv=diagnoses_csv, labels_csv=labels_csv, llm_enabled=False)

    directory = silver.silver_dir("silver-1")
    assert (directory / "annotation.csv").is_file()
    assert manifest["silver_id"] == "silver-1"
    assert manifest["llm_enabled"] is False
    assert manifest["cleanup_enabled"] is False
    assert manifest["input_csv_sha256"] == read_manifest(directory)["input_csv_sha256"]
    assert sum(manifest["decision_stage_counts"].values()) == 8  # fixtures.CASE_IDS has 8 rows
    assert "cascade_constants_sha256" in manifest
    # Manifest is written last: annotation.csv is listed and verifies clean.
    assert "annotation.csv" in manifest["files"]


def test_run_refuses_to_overwrite_existing_silver_id(dm_env, diagnoses_csv, labels_csv):
    silver.run("silver-1", diagnoses_csv=diagnoses_csv, labels_csv=labels_csv, llm_enabled=False)
    with pytest.raises(FileExistsError):
        silver.run("silver-1", diagnoses_csv=diagnoses_csv, labels_csv=labels_csv, llm_enabled=False)


def test_load_silver_round_trips(dm_env, diagnoses_csv, labels_csv):
    silver.run("silver-1", diagnoses_csv=diagnoses_csv, labels_csv=labels_csv, llm_enabled=False)
    loaded = silver.load_silver("silver-1", allow_no_llm=True)
    assert set(loaded["case_id"]) == {f"CASE-{i:04d}" for i in range(1, 9)}
    assert (loaded["silver_generation"] == "silver-1").all()


def test_load_silver_detects_tampering(dm_env, diagnoses_csv, labels_csv):
    silver.run("silver-1", diagnoses_csv=diagnoses_csv, labels_csv=labels_csv, llm_enabled=False)
    directory = silver.silver_dir("silver-1")
    with open(directory / "annotation.csv", "a", encoding="utf-8") as file:
        file.write("CASE-9999,1,tampered,,,,,,No Match,0.0,no_signal,silver-1\n")
    with pytest.raises(ManifestError):
        silver.load_silver("silver-1", allow_no_llm=True)


def test_load_silver_refuses_no_llm_generation_by_default(dm_env, diagnoses_csv, labels_csv):
    silver.run("silver-1", diagnoses_csv=diagnoses_csv, labels_csv=labels_csv, llm_enabled=False)
    with pytest.raises(silver.NoLLMGenerationError):
        silver.load_silver("silver-1")


def test_load_silver_allows_no_llm_generation_with_flag(dm_env, diagnoses_csv, labels_csv):
    silver.run("silver-1", diagnoses_csv=diagnoses_csv, labels_csv=labels_csv, llm_enabled=False)
    loaded = silver.load_silver("silver-1", allow_no_llm=True)
    assert len(loaded) == 8


def test_load_silver_allows_llm_enabled_generation_by_default(dm_env, diagnoses_csv, labels_csv, monkeypatch):
    # No tier3-eligible rows in this fixture, but llm_enabled=True is recorded
    # regardless, so the default (allow_no_llm=False) must not refuse it.
    monkeypatch.setattr(llm_client, "chat", lambda *a, **k: "no match")
    silver.run("silver-1", diagnoses_csv=diagnoses_csv, labels_csv=labels_csv, llm_enabled=True)
    loaded = silver.load_silver("silver-1")
    assert len(loaded) == 8


def test_run_with_cleanup_records_cleanup_config(dm_env, diagnoses_csv, labels_csv, monkeypatch):
    monkeypatch.setattr(llm_client, "chat", lambda *a, **k: "CORRECT")
    manifest = silver.run(
        "silver-1", diagnoses_csv=diagnoses_csv, labels_csv=labels_csv, llm_enabled=False,
        cleanup_enabled=True, cleanup_models=["model-a", "model-b"],
    )
    assert manifest["cleanup_enabled"] is True
    assert manifest["cleanup_models"] == ["model-a", "model-b"]
    loaded = silver.load_silver("silver-1", allow_no_llm=True)
    # Every confirmed (Exact) row was voted "CORRECT" by both mocked models, so
    # the matched codes from the cascade must be unchanged after cleanup.
    exact_rows = loaded[loaded["method"] == "Exact"]
    assert len(exact_rows) > 0
    assert (exact_rows["matched_code"] != "").all()


# ---------------------------------------------------------------------------
# import_legacy()
# ---------------------------------------------------------------------------


def test_import_legacy_adds_silver_generation_column_only(dm_env):
    config.LEGACY_ANNOTATION_CSV.parent.mkdir(parents=True, exist_ok=True)
    fx.make_annotation_csv(config.LEGACY_ANNOTATION_CSV)
    original = io_utils.read_csv(config.LEGACY_ANNOTATION_CSV, encoding="utf-8", dtype=str, keep_default_na=False)

    manifest = silver.import_legacy()

    assert manifest["silver_id"] == silver.LEGACY_SILVER_ID
    assert "unknown" in manifest["cascade_version"]
    loaded = silver.load_silver(silver.LEGACY_SILVER_ID)
    assert (loaded["silver_generation"] == silver.LEGACY_SILVER_ID).all()
    assert loaded.drop(columns=["silver_generation"]).equals(original)
    assert manifest["decision_stage_counts"] == original["decision_stage"].value_counts().to_dict()


def test_import_legacy_refuses_to_overwrite(dm_env):
    config.LEGACY_ANNOTATION_CSV.parent.mkdir(parents=True, exist_ok=True)
    fx.make_annotation_csv(config.LEGACY_ANNOTATION_CSV)
    silver.import_legacy()
    with pytest.raises(FileExistsError):
        silver.import_legacy()
