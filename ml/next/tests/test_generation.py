"""report_mapping/model/generation.py: bundle load/save, fingerprint refusal."""

from __future__ import annotations

from pathlib import Path

import pytest

from report_mapping.model.generation import (
    GenerationError,
    compute_embedding_fingerprint,
    generation_paths,
    load_generation,
    resolve_generation_dir,
    safe_filename,
)


def test_safe_filename():
    assert safe_filename("Mast Cell Tumors") == "mast_cell_tumors"
    assert safe_filename("Blood/vessel, tumors!!") == "blood_vessel_tumors"


def test_resolve_generation_dir(monkeypatch):
    import config

    monkeypatch.setattr(config, "REPORT_MAPPING_CURRENT_DIR", Path("/x/current"))
    monkeypatch.setattr(config, "REPORT_MAPPING_CANDIDATE_DIR", Path("/x/candidate"))
    assert resolve_generation_dir("current") == Path("/x/current")
    assert resolve_generation_dir("candidate") == Path("/x/candidate")
    assert resolve_generation_dir("/some/dir") == Path("/some/dir")


def test_load_generation_round_trip(report_mapping_bundle: Path):
    gen = load_generation(report_mapping_bundle)
    assert gen.generation_id == "test-gen"
    assert set(gen.group_names) == {"Neoplasms, NOS", "Mast Cell Tumors", "Uncommon"}
    assert gen.uncommon_groups == frozenset({"Rare Sarcomas", "Rare Carcinomas"})
    assert set(gen.label_presence_heads) == set(gen.group_names)
    assert gen.thresholds["case_presence_gate"] == 0.5
    assert len(gen.taxonomy_labels) > 0


def test_load_generation_default_uses_config_current(report_mapping_bundle: Path, monkeypatch):
    import config

    monkeypatch.setattr(config, "REPORT_MAPPING_CURRENT_DIR", report_mapping_bundle)
    gen = load_generation()  # no arg -> config.REPORT_MAPPING_CURRENT_DIR, resolved at call time
    assert gen.generation_id == "test-gen"


def test_fingerprint_mismatch_refuses_to_load(report_mapping_bundle: Path):
    # Simulate the code's section-spec version changing since this generation was
    # written: the files (and their sha256s) are untouched -- verify_manifest
    # passes -- but the manifest's recorded fingerprint no longer matches what
    # compute_embedding_fingerprint() gets from the *current* code + the
    # generation's own (unchanged) bundled backbone.
    import json

    from generations.manifest import MANIFEST_NAME

    manifest_path = report_mapping_bundle / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["embedding_fingerprint"]["section_spec_version"] = 999
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(GenerationError, match="fingerprint"):
        load_generation(report_mapping_bundle)


def test_manifest_tamper_outside_petbert_also_refuses(report_mapping_bundle: Path):
    # A file changed anywhere in the bundle fails verify_manifest first (a
    # stricter, earlier check than the fingerprint comparison).
    from report_mapping.model.generation import GenerationError as _GE  # noqa: F401
    from generations.manifest import ManifestError

    paths = generation_paths(report_mapping_bundle)
    paths.thresholds_json.write_text('{"case_presence_gate": 0.9}', encoding="utf-8")
    with pytest.raises(ManifestError):
        load_generation(report_mapping_bundle)


def test_compute_embedding_fingerprint_stable(report_mapping_bundle: Path):
    paths = generation_paths(report_mapping_bundle)
    first = compute_embedding_fingerprint(paths.petbert_dir)
    second = compute_embedding_fingerprint(paths.petbert_dir)
    assert first == second
    assert set(first) == {"backbone_sha256", "section_spec_version", "max_length"}


# ---------------------------------------------------------------------------
# calibration.status: a freshly trained candidate's thresholds are a
# placeholder copied from its parent, so loading it for real prediction must
# be refused until it's calibrated (scripts/train.py writes "pending";
# calibrate.py writes "calibrated"; report_mapping_bundle is "calibrated" by
# default, like a normal ready-to-use "current" generation).
# ---------------------------------------------------------------------------


def _set_calibration_status(bundle: Path, status: str | None) -> None:
    import json

    from generations.manifest import MANIFEST_NAME

    manifest_path = bundle / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if status is None:
        manifest.pop("calibration", None)
    else:
        manifest["calibration"] = {"status": status}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_load_generation_refuses_pending_calibration(report_mapping_bundle: Path):
    _set_calibration_status(report_mapping_bundle, "pending")
    with pytest.raises(GenerationError, match="calibration"):
        load_generation(report_mapping_bundle)


def test_load_generation_refuses_missing_calibration_block(report_mapping_bundle: Path):
    _set_calibration_status(report_mapping_bundle, None)
    with pytest.raises(GenerationError, match="calibration"):
        load_generation(report_mapping_bundle)


def test_load_generation_allow_uncalibrated_bypasses_the_refusal(report_mapping_bundle: Path):
    _set_calibration_status(report_mapping_bundle, "pending")
    gen = load_generation(report_mapping_bundle, allow_uncalibrated=True)
    assert gen.generation_id == "test-gen"


def test_load_generation_accepts_calibrated_status(report_mapping_bundle: Path):
    # report_mapping_bundle's default -- confirms the fixture itself is calibrated.
    gen = load_generation(report_mapping_bundle)
    assert gen.generation_id == "test-gen"
