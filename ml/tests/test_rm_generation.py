"""report_mapping/generation.py: layout, manifest, and the spec check every loader runs."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

import config
from generations.manifest import read_manifest, verify_manifest
from report_mapping import generation
from report_mapping.generation import GenerationError

from . import fixtures as fx


def _edit_manifest(directory: Path, edit) -> None:
    """Change manifest.json fields (the manifest does not hash itself)."""
    manifest = read_manifest(directory)
    edit(manifest)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


@pytest.fixture
def gen(tmp_path, tiny_modernbert_dir):
    return fx.build_tiny_generation(tmp_path / "gen", tiny_modernbert_dir)


def test_generation_id_is_gen_and_the_utc_time():
    assert generation.new_generation_id(datetime(2026, 10, 9, 3, 15, 0, tzinfo=timezone.utc)) == "gen-20261009T031500Z"
    assert re.fullmatch(r"gen-\d{8}T\d{6}Z", generation.new_generation_id())


def test_a_trained_generation_has_the_planned_manifest(tmp_path, tiny_modernbert_dir):
    pending = fx.build_tiny_generation(tmp_path / "p", tiny_modernbert_dir, calibrated=False)
    manifest = read_manifest(pending)
    assert manifest["kind"] == "report_mapping_generation" and manifest["schema_version"] == 2
    assert re.fullmatch(r"gen-\d{8}T\d{6}Z", manifest["generation_id"]) and manifest["status"] == "candidate"
    assert manifest["architecture"] == {"name": "joint-modernbert-v1", "text_spec_version": 1, "pooling": "mean",
                                        "heads": ["term", "gate"], "decision_rule": "gate-top1-rank2-v1"}
    assert manifest["model"]["n_terms"] == 4 and manifest["model"]["max_length"] == 64
    assert manifest["calibration"] == {"status": "pending"}
    assert {"created_at", "git_sha", "files"} <= set(manifest)
    assert set(manifest["files"]) >= {"encoder/model.safetensors", "head.pt", "gate_head.pt", "terms.json",
                                      "term_info.json", "encoder/config.json"}
    assert "thresholds.json" not in manifest["files"]    # no thresholds are fitted at training time


def test_weights_sha256_covers_the_encoder_weights_both_heads_and_the_terms(gen):
    digest = hashlib.sha256()
    for name in ("encoder/model.safetensors", "head.pt", "gate_head.pt", "terms.json"):
        digest.update(f"{name}:{hashlib.sha256((gen / name).read_bytes()).hexdigest()}\n".encode())
    assert generation.weights_sha256(gen) == digest.hexdigest() == read_manifest(gen)["model"]["weights_sha256"]


def test_a_calibrated_generation_passes_the_spec_check_and_binds_the_thresholds_to_its_weights(gen):
    manifest = generation.verify_spec(gen)
    assert manifest["calibration"]["status"] == "calibrated"
    assert manifest["calibration"]["weights_sha256"] == manifest["model"]["weights_sha256"]
    assert "thresholds.json" in manifest["files"]
    assert generation.read_thresholds(gen) == fx.TINY_THRESHOLDS
    assert generation.read_terms(gen) == fx.TINY_TERMS
    assert generation.read_term_info(gen)["Mast cell tumor, benign"] == ("Mast Cell Tumors", "1002/0")


def test_calibrating_keeps_the_training_time_and_code_version(tmp_path, tiny_modernbert_dir):
    pending = fx.build_tiny_generation(tmp_path / "p", tiny_modernbert_dir, calibrated=False)
    before = read_manifest(pending)
    generation.write_calibration(pending, fx.TINY_THRESHOLDS, {"partition": "calibration"})
    after = read_manifest(pending)
    assert (after["created_at"], after["git_sha"], after["generation_id"]) == (
        before["created_at"], before["git_sha"], before["generation_id"])
    verify_manifest(pending)


def test_a_tampered_file_is_refused(gen):
    (gen / "head.pt").write_bytes(b"tampered")
    with pytest.raises(GenerationError, match="head.pt"):
        generation.verify_spec(gen)


def test_a_file_the_manifest_does_not_list_is_refused(gen):
    (gen / "stray.txt").write_text("x", encoding="utf-8")
    with pytest.raises(GenerationError, match="stray.txt"):
        generation.verify_spec(gen)


def test_a_text_spec_the_code_does_not_have_is_refused(gen):
    _edit_manifest(gen, lambda m: m["architecture"].update(text_spec_version=99))
    with pytest.raises(GenerationError, match="architecture"):
        generation.verify_spec(gen)


def test_another_architecture_name_is_refused(gen):
    _edit_manifest(gen, lambda m: m["architecture"].update(name="other-v2"))
    with pytest.raises(GenerationError, match="architecture"):
        generation.verify_spec(gen)


def test_an_uncalibrated_generation_is_refused_unless_calibrate_itself_loads_it(tmp_path, tiny_modernbert_dir):
    pending = fx.build_tiny_generation(tmp_path / "p", tiny_modernbert_dir, calibrated=False)
    with pytest.raises(GenerationError, match="calibrate"):
        generation.verify_spec(pending)
    assert generation.verify_spec(pending, require_calibrated=False)["calibration"]["status"] == "pending"


def test_thresholds_calibrated_for_other_weights_are_refused(gen):
    _edit_manifest(gen, lambda m: m["calibration"].update(weights_sha256="0" * 64))
    with pytest.raises(GenerationError, match="other weights"):
        generation.verify_spec(gen)


def test_a_manifest_weights_hash_that_does_not_match_the_files_is_refused(gen):
    _edit_manifest(gen, lambda m: m["model"].update(weights_sha256="1" * 64))
    with pytest.raises(GenerationError, match="weights_sha256"):
        generation.verify_spec(gen)


def test_a_term_count_that_does_not_match_terms_json_is_refused(gen):
    _edit_manifest(gen, lambda m: m["model"].update(n_terms=3))
    with pytest.raises(GenerationError, match="n_terms"):
        generation.verify_spec(gen)


def test_a_directory_that_is_not_a_generation_is_refused(tmp_path):
    (tmp_path / "x").mkdir()
    with pytest.raises(GenerationError):
        generation.verify_spec(tmp_path / "x")


def test_read_thresholds_names_a_missing_key(gen):
    (gen / "thresholds.json").write_text(json.dumps({"gate_t": 0.5}), encoding="utf-8")
    with pytest.raises(GenerationError, match="review_cut"):
        generation.read_thresholds(gen)


def test_resolve_generation_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REPORT_MAPPING_CURRENT_DIR", tmp_path / "cur")
    monkeypatch.setattr(config, "REPORT_MAPPING_CANDIDATE_DIR", tmp_path / "cand")
    assert generation.resolve_generation_dir("current") == tmp_path / "cur"
    assert generation.resolve_generation_dir("candidate") == tmp_path / "cand"
    assert generation.resolve_generation_dir(str(tmp_path / "elsewhere")) == tmp_path / "elsewhere"


def test_generation_module_does_not_import_torch():
    code = "import sys, report_mapping.generation; sys.exit(int('torch' in sys.modules))"
    result = subprocess.run([sys.executable, "-c", code], cwd=config.ML_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
