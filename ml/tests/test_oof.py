"""report_mapping/training/oof.py: k-fold heads-only out-of-fold predictions."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from report_mapping.model.generation import generation_paths
from report_mapping.training import oof

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
    return labels


def test_case_presence_oof_covers_every_train_case(tmp_path, monkeypatch, tiny_bert_dir):
    labels = _training_env(tmp_path, monkeypatch, tiny_bert_dir)

    result = oof.run_case_presence_oof(labels, "all-train", seed=42, device="cpu", k=3, local_only=True)

    assert len(result.case_ids) == 30
    assert result.probs.shape == (30,)
    assert result.targets.shape == (30,)
    assert int(result.targets.sum()) == 15
    assert ((result.probs >= 0.0) & (result.probs <= 1.0)).all()
    # Every case got a real fold prediction (no leftover zero-initialised slot
    # from a case that fell through the k-fold partition).
    assert (result.probs != 0.0).any()


def test_group_oof_covers_every_train_case(tmp_path, monkeypatch, tiny_bert_dir):
    labels = _training_env(tmp_path, monkeypatch, tiny_bert_dir)

    result = oof.run_group_oof(labels, "all-train", seed=42, device="cpu", k=3, local_only=True)

    assert len(result.case_ids) == 30
    assert result.group_names == ["Uncommon"]
    assert result.probs.shape == (30, 1)
    assert result.targets.shape == (30, 1)
    assert int(result.targets.sum()) == 15
    assert ((result.probs >= 0.0) & (result.probs <= 1.0)).all()
