"""scripts/train.py: the thin entry point.

End-to-end on synthetic data: ``--stage heads`` (gate, then group, then
label-presence) produces a complete candidate generation whose manifest
verifies and that ``report_mapping.model.generation.load_generation`` can
load — the manifest is the deliverable's own acceptance test ("the candidate
manifest is complete, and load_generation loads the candidate").
"""

from __future__ import annotations

import importlib.util
import re
import shutil
import sys
from pathlib import Path

import pytest

from generations.guards import GuardViolation
from report_mapping.model.generation import GenerationError, load_generation

from . import fixtures as fx

_TRAIN_PY = Path(__file__).resolve().parents[1] / "scripts" / "train.py"
_spec = importlib.util.spec_from_file_location("ml_next_scripts_train", _TRAIN_PY)
train_script = importlib.util.module_from_spec(_spec)
sys.modules.setdefault(_spec.name, train_script)
_spec.loader.exec_module(train_script)


@pytest.fixture
def training_env(tmp_path, monkeypatch, tiny_bert_dir):
    fx.point_training_config_at(monkeypatch, tmp_path)
    import config

    config.REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fx.make_training_reports_csv(config.REPORT_CSV)
    fx.make_labels_csv(config.LABELS_CSV)
    ann_path = tmp_path / "annotation.csv"
    fx.make_training_annotation_csv(ann_path)
    fx.make_two_way_split_generation("all-train", train_ids=fx.TRAINING_CASE_IDS, test_ids=[])

    from report_mapping.model.generation import generation_paths
    current_petbert = generation_paths(config.REPORT_MAPPING_CURRENT_DIR).petbert_dir
    current_petbert.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(tiny_bert_dir, current_petbert)
    # A minimal "current" generation thresholds.json so train.py's manifest
    # step has a parent placeholder to copy.
    current_thresholds = generation_paths(config.REPORT_MAPPING_CURRENT_DIR).thresholds_json
    current_thresholds.parent.mkdir(parents=True, exist_ok=True)
    current_thresholds.write_text('{"case_presence_gate": 0.5}', encoding="utf-8")

    return ann_path, config


def test_heads_stage_produces_a_loadable_candidate(training_env):
    ann_path, config = training_env

    rc = train_script.main([
        "--stage", "heads", "--labels", str(ann_path), "--split", "all-train",
        "--seed", "42", "--device", "cpu", "--out", "candidate", "--local-only",
    ])
    assert rc == 0

    candidate_dir = config.REPORT_MAPPING_CANDIDATE_DIR
    manifest_path = candidate_dir / "manifest.json"
    assert manifest_path.is_file()

    # A freshly trained candidate is uncalibrated: real loading must be refused by default...
    with pytest.raises(GenerationError, match="calibration"):
        load_generation(candidate_dir)

    # ...but allow_uncalibrated=True (e.g. for inspection, or calibrate.py's own load) works.
    gen = load_generation(candidate_dir, allow_uncalibrated=True)
    assert gen.group_names == ["Uncommon"]
    assert "Uncommon" in gen.label_presence_heads
    assert gen.thresholds == {"case_presence_gate": 0.5}  # copied from the parent as a placeholder
    assert gen.manifest["calibration"]["status"] == "pending"
    assert gen.manifest["status"] == "candidate"
    assert gen.manifest["parents"]["split_id"] == "all-train"
    # What the triggers read: a plain CSV with no silver_generation column trained on no gold.
    assert gen.manifest["parents"]["silver_id"] is None
    assert gen.manifest["parents"]["gold_train_codes"] == 0
    assert re.fullmatch(r"gen-\d{8}T\d{6}Z", gen.generation_id)


def test_backbone_then_heads_stage_reuses_candidate_petbert(training_env):
    ann_path, config = training_env
    from report_mapping.model.generation import generation_paths

    current_petbert = generation_paths(config.REPORT_MAPPING_CURRENT_DIR).petbert_dir
    rc = train_script.main([
        "--stage", "backbone", "--labels", str(ann_path), "--split", "all-train",
        "--seed", "42", "--device", "cpu", "--out", "candidate", "--local-only",
        "--model", str(current_petbert),
    ])
    assert rc == 0
    candidate_petbert = generation_paths(config.REPORT_MAPPING_CANDIDATE_DIR).petbert_dir
    assert candidate_petbert.is_dir()
    fingerprint_before = candidate_petbert.stat().st_mtime

    rc = train_script.main([
        "--stage", "heads", "--labels", str(ann_path), "--split", "all-train",
        "--seed", "42", "--device", "cpu", "--out", "candidate", "--local-only",
    ])
    assert rc == 0
    # heads-only training must reuse the backbone the "backbone" stage just
    # wrote, not overwrite it with a fresh copy of the current generation's.
    assert candidate_petbert.stat().st_mtime == fingerprint_before

    gen = load_generation(config.REPORT_MAPPING_CANDIDATE_DIR, allow_uncalibrated=True)
    assert gen.group_names == ["Uncommon"]


def test_heads_stage_refuses_when_a_split_guard_fails(tmp_path, monkeypatch, tiny_bert_dir):
    fx.point_training_config_at(monkeypatch, tmp_path)
    import config

    config.REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fx.make_training_reports_csv(config.REPORT_CSV)
    fx.make_labels_csv(config.LABELS_CSV)
    ann_path = tmp_path / "annotation.csv"
    fx.make_training_annotation_csv(ann_path)
    # A corrupted split: one case sits in both train and test -- guards.check_all's
    # disjointness check must refuse before any data loading or training happens
    # (ml-rewrite-plan.md: "split.py check runs inside train ... and refuses on
    # any violation").
    fx.make_two_way_split_generation("corrupt-split", train_ids=fx.TRAINING_CASE_IDS,
                                      test_ids=[fx.TRAINING_CASE_IDS[0]])

    from report_mapping.model.generation import generation_paths
    current_petbert = generation_paths(config.REPORT_MAPPING_CURRENT_DIR).petbert_dir
    current_petbert.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(tiny_bert_dir, current_petbert)

    with pytest.raises(GuardViolation):
        train_script.main([
            "--stage", "heads", "--labels", str(ann_path), "--split", "corrupt-split",
            "--seed", "42", "--device", "cpu", "--out", "candidate", "--local-only",
        ])

    # Refused before anything was written.
    assert not (config.REPORT_MAPPING_CANDIDATE_DIR / "manifest.json").exists()
