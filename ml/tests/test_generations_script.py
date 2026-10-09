"""scripts/generations.py (status, recalibrate) and report_mapping.generation.copy_for_recalibration."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

import config
from generations.manifest import read_manifest, update_manifest, verify_manifest
from report_mapping import generation

from . import fixtures as fx

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "generations.py"
_spec = importlib.util.spec_from_file_location("ml_next_scripts_generations_tests", _PATH)
script = importlib.util.module_from_spec(_spec)
sys.modules.setdefault(_spec.name, script)
_spec.loader.exec_module(script)


@pytest.fixture
def world(tmp_path, monkeypatch, tiny_generation_dir):
    """The training world with a calibrated current/ (a copy of the tiny generation, trained on the world's split)."""
    labels, split_id = fx.make_training_world(tmp_path, monkeypatch)
    shutil.copytree(tiny_generation_dir, config.REPORT_MAPPING_CURRENT_DIR)
    update_manifest(config.REPORT_MAPPING_CURRENT_DIR, {"generation_id": "gen-A", "status": "current",
                                                       "parents": {"split_id": split_id, "silver_id": "silver-A"}})
    return labels, split_id


# ---- copy_for_recalibration ----

def test_copy_for_recalibration_is_a_new_pending_generation_with_the_same_weights(world, tmp_path):
    source = config.REPORT_MAPPING_CURRENT_DIR
    update_manifest(source, {"adoption": {"reason": "documentation/history/x.md", "adopted_at": "2026-10-09"}})
    before = read_manifest(source)

    manifest = generation.copy_for_recalibration(source, tmp_path / "copy")

    assert manifest["generation_id"] != "gen-A" and manifest["recalibrated_from"] == "gen-A"
    assert manifest["status"] == "candidate" and manifest["calibration"] == {"status": "pending"}
    assert manifest["model"] == before["model"] and manifest["architecture"] == before["architecture"]
    assert manifest["parents"] == before["parents"] and "adoption" not in manifest
    assert not (tmp_path / "copy" / "thresholds.json").exists()
    for name in generation.WEIGHT_FILES:
        assert (tmp_path / "copy" / name).read_bytes() == (source / name).read_bytes()
    assert (tmp_path / "copy" / "encoder" / "config.json").is_file()
    assert read_manifest(source) == before            # the source is untouched
    verify_manifest(tmp_path / "copy")
    assert generation.verify_spec(tmp_path / "copy", require_calibrated=False)["generation_id"] == manifest["generation_id"]
    with pytest.raises(generation.GenerationError, match="calibrate"):
        generation.verify_spec(tmp_path / "copy")


def test_copy_for_recalibration_refuses_an_existing_destination_and_an_unverified_source(world, tmp_path):
    source = config.REPORT_MAPPING_CURRENT_DIR
    (tmp_path / "taken").mkdir()
    with pytest.raises(generation.GenerationError, match="already exists"):
        generation.copy_for_recalibration(source, tmp_path / "taken")
    update_manifest(source, {"calibration": {"status": "pending"}})
    with pytest.raises(generation.GenerationError, match="calibration.status"):
        generation.copy_for_recalibration(source, tmp_path / "copy")
    assert not (tmp_path / "copy").exists()


# ---- recalibrate ----

def test_recalibrate_makes_a_new_calibrated_candidate_and_leaves_current_alone(world, capsys):
    labels, split_id = world
    before = read_manifest(config.REPORT_MAPPING_CURRENT_DIR)

    assert script.main(["recalibrate", "--labels", labels, "--split", split_id, "--device", "cpu",
                        "--max-fp-share", "1.0"]) == 0

    candidate = generation.verify_spec(config.REPORT_MAPPING_CANDIDATE_DIR)   # calibrated and bound to its weights
    assert candidate["generation_id"] != "gen-A" and candidate["recalibrated_from"] == "gen-A"
    assert candidate["model"]["weights_sha256"] == before["model"]["weights_sha256"]
    assert candidate["calibration"]["max_fp_share"] == 1.0
    assert read_manifest(config.REPORT_MAPPING_CURRENT_DIR) == before
    out = capsys.readouterr().out
    assert "copied gen-A" in out and "calibrated: gate_t=" in out


def test_recalibrate_defaults_to_the_config_cap_and_review_share(world, monkeypatch):
    labels, _ = world
    seen = []

    def spy(*args):
        seen.append(args)
        return {"values": {"gate_t": 0.5}, "partition_gs_share": 0.0}

    monkeypatch.setattr(script, "calibrate", spy)
    assert script.main(["recalibrate", "--labels", labels, "--device", "cpu"]) == 0
    assert seen == [(config.REPORT_MAPPING_CANDIDATE_DIR, labels, config.DEFAULT_SPLIT_ID, config.FP_CAP,
                     config.REVIEW_TARGET_SHARE, "cpu")]


def test_recalibrate_refuses_to_overwrite_an_existing_candidate(world):
    labels, split_id = world
    config.REPORT_MAPPING_CANDIDATE_DIR.mkdir(parents=True)
    with pytest.raises(generation.GenerationError, match="already exists"):
        script.main(["recalibrate", "--labels", labels, "--split", split_id, "--device", "cpu"])


# ---- status ----

def test_status_lists_current_candidate_and_archive_with_architecture_and_adoption(world, tiny_generation_dir, capsys):
    candidate = config.REPORT_MAPPING_CANDIDATE_DIR
    shutil.copytree(tiny_generation_dir, candidate)
    update_manifest(candidate, {"generation_id": "gen-B", "calibration": {"status": "pending"}})
    archived = config.ARCHIVE_ROOT / "2026-10-09_joint-modernbert"
    shutil.copytree(tiny_generation_dir, archived)
    update_manifest(archived, {"generation_id": "gen-0", "status": "archived", "architecture": {"name": "petbert-four-stage"},
                               "adoption": {"reason": "documentation/history/decisions/0009-x.md"}})
    (config.ARCHIVE_ROOT / "2026-05-01_no-manifest").mkdir()    # not a generation: skipped

    assert script.main(["status"]) == 0

    lines = capsys.readouterr().out.splitlines()
    assert any(line.startswith("current: gen-A") and "architecture=joint-modernbert-v1" in line
               and "calibration=calibrated" in line for line in lines)
    assert any(line.startswith("candidate: gen-B") and "calibration=pending" in line for line in lines)
    assert "archive: 1 generation(s)" in lines
    assert any("2026-10-09_joint-modernbert: gen-0" in line and "architecture=petbert-four-stage" in line
               and "adopted_for=documentation/history/decisions/0009-x.md" in line for line in lines)
    assert not any("trigger" in line for line in lines)      # no --silver: no triggers


def test_status_without_any_generation(tmp_path, monkeypatch, capsys):
    fx.point_promotion_config_at(monkeypatch, tmp_path)
    assert script.main(["status"]) == 0
    out = capsys.readouterr().out
    assert "current: none" in out and "candidate: none" in out and "archive: 0 generation(s)" in out


def test_status_with_silver_reports_the_retraining_triggers(world, capsys):
    _, split_id = world
    fx.make_two_way_split_generation("promo-split", ["P1"], ["P2"])
    assert script.main(["status", "--silver", "silver-B", "--split", "promo-split"]) == 0
    out = capsys.readouterr().out
    assert "trigger new_silver_lineage   MET" in out and "Retrain: yes" in out
