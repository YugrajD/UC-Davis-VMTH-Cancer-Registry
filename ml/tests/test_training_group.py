"""report_mapping/training/group.py: the Stage 2 GroupClassifier trainer.

Tiny synthetic run: loss falls (val macro F1 improves from the initial -1.0
sentinel), the same seed on CPU reproduces identical weights, and every
synthetic group (all below the production uncommon_threshold=200) merges
into the shared "Uncommon" head with "Neoplasms, NOS"-style forcing available
via recipe.GROUP.excluded_groups.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import torch

from report_mapping.model.generation import generation_paths
from report_mapping.model.heads import GroupClassifier
from report_mapping.training import group

from . import fixtures as fx


def _training_env(tmp_path: Path, monkeypatch, tiny_bert_dir: Path):
    fx.point_training_config_at(monkeypatch, tmp_path)
    import config

    config.REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fx.make_training_reports_csv(config.REPORT_CSV)
    fx.make_labels_csv(config.LABELS_CSV)
    ann_path = tmp_path / "annotation.csv"
    fx.make_training_annotation_csv(ann_path)
    fx.make_two_way_split_generation("all-train", train_ids=fx.TRAINING_CASE_IDS, test_ids=[])
    labels = pd.read_csv(ann_path, dtype=str, keep_default_na=False)

    current_petbert = generation_paths(config.REPORT_MAPPING_CURRENT_DIR).petbert_dir
    current_petbert.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(tiny_bert_dir, current_petbert)
    return labels, config


def test_group_train_merges_everything_into_uncommon(tmp_path, monkeypatch, tiny_bert_dir):
    labels, config = _training_env(tmp_path, monkeypatch, tiny_bert_dir)

    out_dir = tmp_path / "candidate1"
    result = group.train(labels, "all-train", seed=42, device="cpu", local_only=True, out_dir=out_dir)

    # All 3 synthetic groups (5 cases each) are far below the production
    # uncommon_threshold=200, so they all merge into one "Uncommon" head.
    assert result["group_names"] == ["Uncommon"]
    assert sorted(result["uncommon_groups"]) == ["Mast Cell Tumors", "Rare Carcinomas", "Rare Sarcomas"]
    assert result["best_f1"] >= 0.0
    assert result["n_cases"] == 30
    assert result["final_loss"] < result["initial_loss"]

    checkpoint = generation_paths(out_dir).group_pt
    assert checkpoint.is_file()
    model, group_names = GroupClassifier.load(checkpoint)
    assert group_names == ["Uncommon"]
    assert model.num_groups == 1

    uncommon_txt = generation_paths(out_dir).uncommon_groups_txt
    assert uncommon_txt.is_file()
    lines = {line.strip() for line in uncommon_txt.read_text(encoding="utf-8").splitlines() if line.strip()}
    assert lines == {"Mast Cell Tumors", "Rare Carcinomas", "Rare Sarcomas"}


def test_group_train_excludes_a_train_case_absent_from_labels(tmp_path, monkeypatch, tiny_bert_dir):
    labels, config = _training_env(tmp_path, monkeypatch, tiny_bert_dir)
    # Same vague-without-gold scenario as case_presence's equivalent test: a
    # case with an embedding, in train, but no row at all in the labels
    # table must not silently become a confirmed-non-cancer negative.
    vague_case = fx.TRAINING_CASE_IDS[-1]
    labels_without_vague = labels[labels["case_id"] != vague_case].reset_index(drop=True)

    out_dir = tmp_path / "candidate_vague"
    result = group.train(labels_without_vague, "all-train", seed=42, device="cpu", local_only=True, out_dir=out_dir)
    assert result["n_cases"] == 29


def test_group_train_same_seed_reproducible_on_cpu(tmp_path, monkeypatch, tiny_bert_dir):
    labels, config = _training_env(tmp_path, monkeypatch, tiny_bert_dir)

    out_dir_a = tmp_path / "candidate_a"
    out_dir_b = tmp_path / "candidate_b"
    group.train(labels, "all-train", seed=7, device="cpu", local_only=True, out_dir=out_dir_a)
    group.train(labels, "all-train", seed=7, device="cpu", local_only=True, out_dir=out_dir_b)

    model_a, names_a = GroupClassifier.load(generation_paths(out_dir_a).group_pt)
    model_b, names_b = GroupClassifier.load(generation_paths(out_dir_b).group_pt)
    assert names_a == names_b
    for (name_a, param_a), (name_b, param_b) in zip(model_a.state_dict().items(), model_b.state_dict().items()):
        assert name_a == name_b
        torch.testing.assert_close(param_a, param_b)
