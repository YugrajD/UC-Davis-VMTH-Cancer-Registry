"""report_mapping/training/label_presence.py: per-group LabelPresenceClassifier
trainers.

Tests both the single-group core (``train_one_group``, directly against a
named common group — the path a group that *wasn't* merged into Uncommon
would take) and the composite ``train`` orchestration (which, on this
synthetic taxonomy, trains only the merged "Uncommon" head, since every
group falls below the production uncommon threshold — see
test_training_group.py). Loss falls (best score beats the -1.0 sentinel) and
the same seed on CPU reproduces identical weights.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import torch

from report_mapping.model.generation import generation_paths, safe_filename
from report_mapping.model.heads import LabelPresenceClassifier
from report_mapping.training import embeddings as embeddings_mod
from report_mapping.training import label_presence
from report_mapping.training import labels as labels_mod
from report_mapping.training.case_presence import default_backbone_dir
from taxonomy.taxonomy import load_labels_taxonomy

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


def test_train_one_group_named_group_loss_falls(tmp_path, monkeypatch, tiny_bert_dir):
    labels, config = _training_env(tmp_path, monkeypatch, tiny_bert_dir)
    labels_train = labels_mod.select_train(labels, "all-train")
    taxonomy_labels = load_labels_taxonomy(str(config.LABELS_CSV))
    cache = embeddings_mod.get_or_build(
        default_backbone_dir(), str(config.LABELS_CSV), taxonomy_labels, local_only=True, device=torch.device("cpu"),
    )
    # "Mast Cell Tumors" alone has only 5 cases / 15 pairs -- the production
    # 25-epoch recipe barely moves the loss on that little data within this
    # test's tight tolerance. More epochs (same recipe otherwise) gives the
    # tiny dataset a fair chance to show a clear, non-flaky improvement; the
    # composite test below already covers the production epoch count on the
    # larger (15-positive) Uncommon pool.
    import dataclasses

    from report_mapping.training import recipe
    monkeypatch.setattr(recipe, "LABEL_PRESENCE", dataclasses.replace(recipe.LABEL_PRESENCE, epochs=150))

    out_pt = tmp_path / "mast_cell_tumors.pt"
    result = label_presence.train_one_group(
        labels_train, "Mast Cell Tumors", taxonomy_labels, cache, seed=42, device=torch.device("cpu"), out_path=out_pt,
    )
    assert result["best_score"] >= 0.0
    assert result["final_loss"] < result["initial_loss"]
    assert out_pt.is_file()
    model = LabelPresenceClassifier.load(out_pt)
    assert model.n_cols == 3 and model.col_pair_mode is True and model.col_combine == "learned"


def test_train_one_group_same_seed_reproducible_on_cpu(tmp_path, monkeypatch, tiny_bert_dir):
    labels, config = _training_env(tmp_path, monkeypatch, tiny_bert_dir)
    labels_train = labels_mod.select_train(labels, "all-train")
    taxonomy_labels = load_labels_taxonomy(str(config.LABELS_CSV))
    cache = embeddings_mod.get_or_build(
        default_backbone_dir(), str(config.LABELS_CSV), taxonomy_labels, local_only=True, device=torch.device("cpu"),
    )

    out_a, out_b = tmp_path / "a.pt", tmp_path / "b.pt"
    label_presence.train_one_group(labels_train, "Mast Cell Tumors", taxonomy_labels, cache,
                                    seed=7, device=torch.device("cpu"), out_path=out_a)
    label_presence.train_one_group(labels_train, "Mast Cell Tumors", taxonomy_labels, cache,
                                    seed=7, device=torch.device("cpu"), out_path=out_b)

    model_a, model_b = LabelPresenceClassifier.load(out_a), LabelPresenceClassifier.load(out_b)
    for (na, pa), (nb, pb) in zip(model_a.state_dict().items(), model_b.state_dict().items()):
        assert na == nb
        torch.testing.assert_close(pa, pb)


def test_train_composite_trains_only_uncommon_head(tmp_path, monkeypatch, tiny_bert_dir):
    labels, config = _training_env(tmp_path, monkeypatch, tiny_bert_dir)

    result = label_presence.train(
        labels, "all-train", seed=42, device="cpu", local_only=True, out_dir=tmp_path / "candidate1",
        group_names=["Uncommon"], uncommon_groups=["Mast Cell Tumors", "Rare Sarcomas", "Rare Carcinomas"],
    )
    uncommon_result = result["results"]["Uncommon"]
    assert uncommon_result["final_loss"] < uncommon_result["initial_loss"]
    assert result["trained"] + result["skipped"] == 1

    out_pt = generation_paths(tmp_path / "candidate1").label_presence_dir / f"{safe_filename('Uncommon')}.pt"
    assert out_pt.is_file()


def test_train_composite_defaults_read_group_checkpoint(tmp_path, monkeypatch, tiny_bert_dir):
    # No explicit group_names/uncommon_groups -> reads them from this
    # candidate's own checkpoints/{group_classifier_best.pt,uncommon_groups.txt},
    # the normal `--stage heads` path (group training already ran this cycle).
    labels, config = _training_env(tmp_path, monkeypatch, tiny_bert_dir)
    from report_mapping.training import group as group_mod

    out_dir = tmp_path / "candidate1"
    group_mod.train(labels, "all-train", seed=42, device="cpu", local_only=True, out_dir=out_dir)

    result = label_presence.train(labels, "all-train", seed=42, device="cpu", local_only=True, out_dir=out_dir)
    assert "Uncommon" in result["scores"]
    assert result["results"]["Uncommon"]["final_loss"] < result["results"]["Uncommon"]["initial_loss"]


def test_train_one_group_lp_pairs_identical_across_different_outer_seeds(tmp_path, monkeypatch, tiny_bert_dir):
    # LP negative sampling always uses recipe.LP_PAIR_SAMPLING_SEED (42),
    # independent of the run's own --seed (recipe.py's docstring; the plan's
    # L3 parity note). Spy on labels_mod.label_presence_pairs to capture the
    # actual pairs frame train_one_group builds under two different outer
    # seeds and confirm they're byte-identical.
    labels, config = _training_env(tmp_path, monkeypatch, tiny_bert_dir)
    labels_train = labels_mod.select_train(labels, "all-train")
    taxonomy_labels = load_labels_taxonomy(str(config.LABELS_CSV))
    cache = embeddings_mod.get_or_build(
        default_backbone_dir(), str(config.LABELS_CSV), taxonomy_labels, local_only=True, device=torch.device("cpu"),
    )

    captured = []
    original = labels_mod.label_presence_pairs

    def spy(*args, **kwargs):
        pairs = original(*args, **kwargs)
        captured.append(pairs)
        return pairs

    monkeypatch.setattr(label_presence.labels_mod, "label_presence_pairs", spy)

    label_presence.train_one_group(labels_train, "Mast Cell Tumors", taxonomy_labels, cache,
                                    seed=1, device=torch.device("cpu"), out_path=tmp_path / "seed1.pt")
    label_presence.train_one_group(labels_train, "Mast Cell Tumors", taxonomy_labels, cache,
                                    seed=2, device=torch.device("cpu"), out_path=tmp_path / "seed2.pt")

    assert len(captured) == 2
    pd.testing.assert_frame_equal(captured[0].reset_index(drop=True), captured[1].reset_index(drop=True))
