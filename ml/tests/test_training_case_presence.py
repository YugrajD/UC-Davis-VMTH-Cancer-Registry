"""report_mapping/training/case_presence.py: the Stage 1 gate trainer.

Tiny synthetic run: loss falls over training, and the same seed on CPU
reproduces identical weights.
"""

from __future__ import annotations

from pathlib import Path

import torch

from report_mapping.model.generation import generation_paths
from report_mapping.model.heads import CasePresenceClassifier
from report_mapping.training import case_presence

from . import fixtures as fx


def _training_env(tmp_path: Path, monkeypatch, tiny_bert_dir: Path) -> tuple:
    fx.point_training_config_at(monkeypatch, tmp_path)
    import config

    config.REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fx.make_training_reports_csv(config.REPORT_CSV)
    fx.make_labels_csv(config.LABELS_CSV)
    ann_path = tmp_path / "annotation.csv"
    fx.make_training_annotation_csv(ann_path)
    fx.make_two_way_split_generation("all-train", train_ids=fx.TRAINING_CASE_IDS, test_ids=[])

    import pandas as pd
    labels = pd.read_csv(ann_path, dtype=str, keep_default_na=False)

    current_petbert = generation_paths(config.REPORT_MAPPING_CURRENT_DIR).petbert_dir
    current_petbert.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copytree(tiny_bert_dir, current_petbert)

    return labels, config


def test_gate_train_loss_falls_and_writes_checkpoint(tmp_path, monkeypatch, tiny_bert_dir):
    labels, config = _training_env(tmp_path, monkeypatch, tiny_bert_dir)

    out_dir = tmp_path / "candidate1"
    result = case_presence.train(labels, "all-train", seed=42, device="cpu", local_only=True, out_dir=out_dir)

    assert result["n_cases"] == 30
    assert result["n_cancer"] == 15
    assert result["best_score"] > -1.0
    assert result["final_loss"] < result["initial_loss"]
    checkpoint = generation_paths(out_dir).case_presence_pt
    assert checkpoint.is_file()
    model = CasePresenceClassifier.load(checkpoint)
    assert model.emb_dim == fx.TINY_EMB_DIM * 3  # concat-3


def test_gate_train_excludes_a_train_case_absent_from_labels(tmp_path, monkeypatch, tiny_bert_dir):
    labels, config = _training_env(tmp_path, monkeypatch, tiny_bert_dir)
    # Drop one case's row entirely -- simulating a vague-without-gold case
    # that coding.corrected removes from the labels table (rather than
    # recording it as confirmed non-cancer). It still has a report/embedding
    # and sits in train, so without labels.training_case_ids it would
    # silently become a confirmed-non-cancer negative.
    vague_case = fx.TRAINING_CASE_IDS[-1]
    labels_without_vague = labels[labels["case_id"] != vague_case].reset_index(drop=True)

    out_dir = tmp_path / "candidate_vague"
    result = case_presence.train(labels_without_vague, "all-train", seed=42, device="cpu",
                                  local_only=True, out_dir=out_dir)
    assert result["n_cases"] == 29
    assert result["n_cancer"] == 15


def test_gate_train_same_seed_reproducible_on_cpu(tmp_path, monkeypatch, tiny_bert_dir):
    labels, config = _training_env(tmp_path, monkeypatch, tiny_bert_dir)

    out_dir_a = tmp_path / "candidate_a"
    out_dir_b = tmp_path / "candidate_b"
    case_presence.train(labels, "all-train", seed=7, device="cpu", local_only=True, out_dir=out_dir_a)
    case_presence.train(labels, "all-train", seed=7, device="cpu", local_only=True, out_dir=out_dir_b)

    model_a = CasePresenceClassifier.load(generation_paths(out_dir_a).case_presence_pt)
    model_b = CasePresenceClassifier.load(generation_paths(out_dir_b).case_presence_pt)
    for (name_a, param_a), (name_b, param_b) in zip(model_a.state_dict().items(), model_b.state_dict().items()):
        assert name_a == name_b
        torch.testing.assert_close(param_a, param_b)
