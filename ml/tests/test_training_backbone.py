"""report_mapping/training/backbone.py: the contrastive PetBERT backbone trainer.

Tiny synthetic run: InfoNCE loss falls over training, and the same seed on
CPU reproduces identical backbone weights.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForMaskedLM

from report_mapping.model.generation import generation_paths
from report_mapping.training import backbone

from . import fixtures as fx


def _training_env(tmp_path: Path, monkeypatch):
    fx.point_training_config_at(monkeypatch, tmp_path)
    import config

    config.REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fx.make_training_reports_csv(config.REPORT_CSV)
    fx.make_labels_csv(config.LABELS_CSV)
    ann_path = tmp_path / "annotation.csv"
    fx.make_training_annotation_csv(ann_path)
    fx.make_two_way_split_generation("all-train", train_ids=fx.TRAINING_CASE_IDS, test_ids=[])
    labels = pd.read_csv(ann_path, dtype=str, keep_default_na=False)
    report_frame = pd.read_csv(config.REPORT_CSV, encoding="utf-8")
    return labels, report_frame


def test_backbone_train_loss_falls(tmp_path, monkeypatch, tiny_bert_dir):
    labels, report_frame = _training_env(tmp_path, monkeypatch)

    out_dir = tmp_path / "candidate1"
    result = backbone.train(
        labels, "all-train", seed=42, device="cpu", model_name=str(tiny_bert_dir),
        local_only=True, out_dir=out_dir, report_frame=report_frame,
    )
    assert result["n_pairs"] > 0
    assert result["final_loss"] < result["initial_loss"]
    petbert_dir = generation_paths(out_dir).petbert_dir
    assert petbert_dir.is_dir()
    assert (petbert_dir / "config.json").is_file()


def test_backbone_train_same_seed_reproducible_on_cpu(tmp_path, monkeypatch, tiny_bert_dir):
    labels, report_frame = _training_env(tmp_path, monkeypatch)

    out_dir_a = tmp_path / "candidate_a"
    out_dir_b = tmp_path / "candidate_b"
    backbone.train(labels, "all-train", seed=7, device="cpu", model_name=str(tiny_bert_dir),
                    local_only=True, out_dir=out_dir_a, report_frame=report_frame)
    backbone.train(labels, "all-train", seed=7, device="cpu", model_name=str(tiny_bert_dir),
                    local_only=True, out_dir=out_dir_b, report_frame=report_frame)

    model_a = AutoModelForMaskedLM.from_pretrained(str(generation_paths(out_dir_a).petbert_dir))
    model_b = AutoModelForMaskedLM.from_pretrained(str(generation_paths(out_dir_b).petbert_dir))
    for (name_a, param_a), (name_b, param_b) in zip(model_a.state_dict().items(), model_b.state_dict().items()):
        assert name_a == name_b
        torch.testing.assert_close(param_a, param_b)
