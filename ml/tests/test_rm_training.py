"""report_mapping/training: label helpers, data, loop, train, calibrate (thresholds + review cut), crossfit.
Synthetic data and a tiny offline ModernBERT only."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import config
from evaluation import verdicts
from generations.guards import GuardViolation
from generations.manifest import read_manifest, verify_manifest
from generations.splits import load_split
from report_mapping import generation, text
from report_mapping.inference import decide, scoring
from report_mapping.inference.decide import NEVER, Thresholds
from report_mapping.inference.predict import load_generation
from report_mapping.inference.scoring import RankedScores
from report_mapping.model import JointClassifier, load_joint, load_tokenizer, new_encoder
from report_mapping.training import calibrate as cal
from report_mapping.training import labels as labels_mod
from report_mapping.training import loop, recipe as recipe_mod
from report_mapping.training import train as train_mod
from report_mapping.training.crossfit import COLUMNS, crossfit
from report_mapping.training.data import assign_folds, load_training_data, split_holdout

from . import fixtures as fx

# --- a small world: reports, labels, a three-way split, the taxonomy --------------------------------------------------

IDS = fx.TRAINING_CASE_IDS                       # 0..14 cancer, 15..29 non-cancer in the fixture's own labels
TRAIN = IDS[0:10] + IDS[15:25]
CALIBRATION = IDS[10:13] + IDS[25:28]
TEST = IDS[13:15] + IDS[28:30]
T4 = ["Mast cell tumor, malignant", "Mast cell tumor, benign", "Mast cell sarcoma", "Fibrosarcoma, NOS"]
SINGLETON = "Squamous cell carcinoma, NOS"
TERM_ROW = {term: (group, code) for group, entries in fx.TAXONOMY_GROUPS.items() for code, term in entries}
NO_LABEL_ROW = IDS[24]       # in train, has a report, but no row in the labels table
NO_TEXT = IDS[23]            # in train, labelled, but every report column is empty


def _terms_of(i: int) -> list[str]:
    terms = {2: [T4[2], T4[3]], 6: [T4[2], T4[0]], 9: [SINGLETON]}
    return terms.get(i, [T4[i % 4]])


def _write_world(root, monkeypatch) -> SimpleNamespace:
    fx.point_training_config_at(monkeypatch, root)
    (root / "data").mkdir()
    fx.make_labels_csv(root / "labels.csv")
    fx.make_training_reports_csv(root / "data" / "report.csv")
    reports = pd.read_csv(root / "data" / "report.csv", dtype=str, keep_default_na=False)
    reports.loc[reports["case_id"] == NO_TEXT, reports.columns[1:]] = ""
    reports.to_csv(root / "data" / "report.csv", index=False)

    rows = []
    for i, case_id in enumerate(IDS):
        if case_id == NO_LABEL_ROW:
            continue
        if i < 15:
            rows += [(case_id, t, TERM_ROW[t][0], TERM_ROW[t][1], "silver-test") for t in _terms_of(i)]
        else:
            rows.append((case_id, "", "", "", "silver-test"))
    labels = root / "labels_table.csv"
    pd.DataFrame(rows, columns=[*labels_mod.REQUIRED_COLUMNS, "silver_generation"]).to_csv(labels, index=False)
    fx.make_three_way_split_generation("tiny-split", TRAIN, CALIBRATION, TEST)
    return SimpleNamespace(root=root, labels=str(labels), split="tiny-split")


@pytest.fixture
def world(tmp_path, monkeypatch):
    return _write_world(tmp_path, monkeypatch)


TINY = recipe_mod.Recipe(epochs=1, max_length=64, batch_size=4, token_budget=256, lr=1e-3, holdout_fraction=0.2)


# --- recipe -----------------------------------------------------------------------------------------------------

def test_recipe_holds_the_values_the_frozen_model_was_trained_with():
    r = recipe_mod.Recipe()
    assert (r.epochs, r.max_length, r.batch_size, r.token_budget) == (3, 4096, 32, 16384)
    assert (r.lr, r.head_lr, r.weight_decay, r.warmup_frac) == (5e-5, 1e-3, 0.01, 0.1)
    assert (r.grad_clip, r.gate_weight, r.holdout_fraction, r.holdout_seed) == (1.0, 1.0, 0.05, 1234)
    assert r.gradient_checkpointing is True and r.seed == recipe_mod.PRODUCTION_SEED == 42
    with pytest.raises(Exception):
        r.epochs = 4


def test_seed_all_makes_torch_and_numpy_draws_repeat():
    recipe_mod.seed_all(7)
    first = (torch.rand(3), np.random.rand(3))
    recipe_mod.seed_all(7)
    assert torch.equal(first[0], torch.rand(3)) and np.array_equal(first[1], np.random.rand(3))


# --- labels (moved helpers) ---------------------------------------------------------------------------------------

def _labels_frame(rows):
    return pd.DataFrame(rows, columns=["case_id", "matched_term", "matched_group", "matched_code"])


def test_lineage_of_a_corrected_table_counts_gold_codes_only():
    df = _labels_frame([("C1", "T1", "G1", "8000/3"), ("C2", "", "", ""), ("C3", "T2", "G1", "8010/3")])
    df["label_source"] = ["gold", "gold", "silver"]  # C2 is a gold NO_CANCER row: no code
    df["silver_generation"] = "silver-7"
    df["gold_snapshot"] = "abc123"
    assert labels_mod.lineage(df, "corrected.csv") == {
        "silver_id": "silver-7", "gold_train_snapshot": "abc123", "gold_train_codes": 1}


def test_lineage_of_a_plain_table_is_unknown_silver_and_no_gold():
    df = _labels_frame([("C1", "T1", "G1", "8000/3")])
    assert labels_mod.lineage(df, "plain.csv") == {"silver_id": None, "gold_train_snapshot": None, "gold_train_codes": 0}


def test_lineage_refuses_a_table_mixing_silver_generations():
    df = _labels_frame([("C1", "T1", "G1", "8000/3"), ("C2", "T1", "G1", "8000/3")])
    df["silver_generation"] = ["silver-1", "silver-2"]
    with pytest.raises(ValueError, match="mixes 2 silver_generation"):
        labels_mod.lineage(df, "x")


def test_load_labels_table_from_csv_path_and_missing_columns(tmp_path):
    path = fx.make_training_annotation_csv(tmp_path / "annotation.csv")
    df = labels_mod.load_labels_table(str(path))
    assert len(df) == 30 and set(labels_mod.REQUIRED_COLUMNS) <= set(df.columns)
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"case_id": ["A"]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="missing column"):
        labels_mod.load_labels_table(str(bad))


def test_select_train_filters_to_the_train_partition(tmp_path, monkeypatch):
    fx.point_generations_config_at(monkeypatch, tmp_path)
    fx.make_two_way_split_generation("s1", train_ids=["A", "B"], test_ids=["C"])
    labels = _labels_frame([("A", "Term A", "Group A", "1"), ("B", "", "", ""), ("C", "Term C", "Group C", "3")])
    assert set(labels_mod.select_train(labels, "s1")["case_id"]) == {"A", "B"}


def test_select_train_refuses_when_split_partitions_overlap(tmp_path, monkeypatch):
    fx.point_generations_config_at(monkeypatch, tmp_path)
    fx.make_two_way_split_generation("corrupt", train_ids=["A", "B"], test_ids=["B", "C"])   # "B" in both
    with pytest.raises(GuardViolation):
        labels_mod.select_train(_labels_frame([("A", "", "", ""), ("B", "Term B", "Group B", "2")]), "corrupt")


def test_training_case_ids_excludes_a_train_case_absent_from_labels_and_keeps_report_order(tmp_path, monkeypatch):
    fx.point_generations_config_at(monkeypatch, tmp_path)
    fx.make_two_way_split_generation("s2", train_ids=["A", "B", "C"], test_ids=["D"])
    split = load_split("s2")
    # "C" has a report and is in train, but no row at all in labels_train (e.g. a vague-without-gold case dropped
    # by coding.corrected): it must not silently become a confirmed-non-cancer negative.
    labels_train = _labels_frame([("A", "Term A", "Group A", "1"), ("B", "", "", "")])
    assert labels_mod.training_case_ids(labels_train, ["B", "A", "C", "D"], split) == ["B", "A"]


def test_gate_targets_and_cancer_case_ids():
    labels = _labels_frame([("A", "Term A", "Group A", "1"), ("A", "", "", ""), ("B", "", "", ""),
                            ("C", "Term C", "Group C", "3")])
    assert labels_mod.cancer_case_ids(labels) == {"A", "C"}
    np.testing.assert_array_equal(labels_mod.gate_targets(labels, ["A", "B", "C", "D"]), [1.0, 0.0, 1.0, 0.0])


# --- data -----------------------------------------------------------------------------------------------------------

def test_training_data_is_the_train_cases_with_text_and_a_label_row(world):
    data = load_training_data(labels_mod.load_labels_table(world.labels), world.split)
    expected = [c for c in TRAIN if c not in (NO_LABEL_ROW, NO_TEXT)]
    assert data.case_ids == expected and len(data.texts) == len(expected)
    assert not set(data.case_ids) & (set(CALIBRATION) | set(TEST))
    assert all(data.texts) and data.texts[0].startswith("HISTOPATHOLOGICAL SUMMARY: ")
    reports = text.read_reports().set_index("case_id")
    assert data.texts[0] == text.build_report_text(reports.loc[data.case_ids[0]])


def test_vocab_is_the_sorted_train_terms_and_targets_are_multi_hot_on_cancer_cases(world):
    data = load_training_data(labels_mod.load_labels_table(world.labels), world.split)
    assert data.vocab == sorted([*T4, SINGLETON])
    assert data.term_targets.shape == (len(data.case_ids), 5) and data.term_targets.dtype == np.float32
    for row, case_id in enumerate(data.case_ids):
        i = IDS.index(case_id)
        expected = set(_terms_of(i)) if i < 15 else set()
        assert {data.vocab[j] for j in np.flatnonzero(data.term_targets[row])} == expected
        assert data.gate_targets[row] == (1.0 if i < 15 else 0.0)


def test_limit_keeps_the_first_cases_but_the_whole_vocabulary(world):
    table = labels_mod.load_labels_table(world.labels)
    full, small = load_training_data(table, world.split), load_training_data(table, world.split, limit=3)
    assert small.case_ids == full.case_ids[:3] and small.vocab == full.vocab and small.term_targets.shape == (3, 5)


def test_a_labelled_case_in_the_calibration_partition_never_reaches_training(world):
    table = labels_mod.load_labels_table(world.labels)
    assert set(table["case_id"]) & set(CALIBRATION)          # the table does hold calibration labels
    assert not set(load_training_data(table, world.split).case_ids) & set(CALIBRATION)


def test_holdout_is_stratified_deterministic_and_a_partition():
    targets = np.array([0.0] * 100 + [1.0] * 40)
    train_idx, held_idx = split_holdout(targets, 0.05, 1234)
    assert (train_idx, held_idx) == split_holdout(targets, 0.05, 1234)
    assert split_holdout(targets, 0.05, 1)[1] != held_idx
    assert sorted(train_idx + held_idx) == list(range(140)) and not set(train_idx) & set(held_idx)
    assert targets[held_idx].sum() == 2 and (targets[held_idx] == 0).sum() == 5       # 5% of each class
    assert held_idx == sorted(held_idx) and train_idx == sorted(train_idx)


def test_folds_are_stratified_balanced_and_deterministic():
    targets = np.array([0.0] * 31 + [1.0] * 17)
    folds = assign_folds(targets, 5, seed=42)
    assert np.array_equal(folds, assign_folds(targets, 5, seed=42)) and not np.array_equal(folds, assign_folds(targets, 5, seed=43))
    assert sorted(set(folds.tolist())) == [0, 1, 2, 3, 4]
    for fold in range(5):
        assert 3 <= targets[folds == fold].sum() <= 4 and 6 <= (targets[folds == fold] == 0).sum() <= 7


# --- the joint loss, metrics, loop ---------------------------------------------------------------------------------

def _loss_inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    cancer = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    term_logits, gate_logits = torch.randn(6, 4, generator=g), torch.randn(6, generator=g)
    term_targets = (torch.rand(6, 4, generator=g) < 0.4).float()
    return term_logits, gate_logits, term_targets, cancer


def test_joint_loss_is_term_bce_over_cancer_cases_plus_gate_bce_over_all_cases():
    term_logits, gate_logits, term_targets, cancer = _loss_inputs()
    term, gate_loss = loop.step_loss(term_logits, gate_logits, term_targets, cancer, cancer, 3, 6)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    by_hand_term = sum(bce(term_logits[i], term_targets[i], reduction="sum") for i in (0, 2, 5)) / 3
    by_hand_gate = sum(bce(gate_logits[i], cancer[i]) for i in range(6)) / 6
    assert term.item() == pytest.approx(by_hand_term.item()) and gate_loss.item() == pytest.approx(by_hand_gate.item())


def test_non_cancer_cases_contribute_no_term_loss_or_gradient_but_do_train_the_gate():
    term_logits, gate_logits, term_targets, cancer = _loss_inputs()
    base, _ = loop.step_loss(term_logits, gate_logits, term_targets, cancer, cancer, 3, 6)
    scrambled = term_targets.clone()
    scrambled[cancer == 0] = 1.0 - scrambled[cancer == 0]            # whatever a non-cancer row holds, it is ignored
    assert loop.step_loss(term_logits, gate_logits, scrambled, cancer, cancer, 3, 6)[0].item() == base.item()

    term_logits, gate_logits = term_logits.clone().requires_grad_(), gate_logits.clone().requires_grad_()
    term, _ = loop.step_loss(term_logits, gate_logits, term_targets, cancer, cancer, 3, 6)
    term.backward()
    assert (term_logits.grad[cancer == 0] == 0).all() and (term_logits.grad[cancer == 1] != 0).any()
    assert gate_logits.grad is None                                   # the term loss does not touch the gate head
    term_logits.grad = None
    _, gate_loss = loop.step_loss(term_logits, gate_logits, term_targets, cancer, cancer, 3, 6)
    gate_loss.backward()
    assert (gate_logits.grad != 0).all() and term_logits.grad is None


def test_a_step_without_cancer_cases_has_zero_term_loss():
    term_logits, gate_logits, term_targets, _ = _loss_inputs()
    none = torch.zeros(6)
    term, gate_loss = loop.step_loss(term_logits, gate_logits, term_targets * 0, none, none, 0, 6)
    assert term.item() == 0.0 and gate_loss.item() > 0


def test_accumulating_micro_batches_equals_the_whole_step_loss_and_gradient():
    term_logits, gate_logits, term_targets, cancer = _loss_inputs()

    def run(micros):
        t, g = term_logits.clone().requires_grad_(), gate_logits.clone().requires_grad_()
        term_sum = gate_sum = 0.0
        for rows in micros:
            term, gate_loss = loop.step_loss(t[rows], g[rows], term_targets[rows], cancer[rows], cancer[rows], 3, 6)
            (term + 0.7 * gate_loss).backward()
            term_sum, gate_sum = term_sum + term.item(), gate_sum + gate_loss.item()
        return term_sum, gate_sum, t.grad, g.grad

    whole = run([[0, 1, 2, 3, 4, 5]])
    split = run([[0, 1, 2], [3, 4], [5]])
    assert split[0] == pytest.approx(whole[0]) and split[1] == pytest.approx(whole[1])
    assert torch.allclose(split[2], whole[2], atol=1e-7) and torch.allclose(split[3], whole[3], atol=1e-7)


def test_roc_auc():
    assert train_mod.roc_auc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == 1.0
    assert train_mod.roc_auc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([0, 0, 1, 1])) == 0.0
    assert train_mod.roc_auc(np.array([0.5, 0.5, 0.5, 0.5]), np.array([0, 1, 0, 1])) == 0.5
    scores, pos = np.array([0.1, 0.4, 0.35, 0.8]), np.array([0, 0, 1, 1])
    assert train_mod.roc_auc(scores, pos) == pytest.approx(0.75)
    assert np.isnan(train_mod.roc_auc(scores, np.zeros(4)))


def test_heldout_metrics_gate_auc_loss_term_top1_and_pair_recall_at_20():
    base = np.arange(30, 0, -1, dtype=np.float32)               # term index i has rank i
    term_logits = np.stack([base, base, base])
    gate_logits, targets = np.array([3.0, 1.0, -2.0], dtype=np.float32), np.array([1.0, 1.0, 0.0], dtype=np.float32)
    gold = [frozenset({0, 19}), frozenset({5, 20}), frozenset()]       # two cancer cases, then a non-cancer case
    m = train_mod.heldout_metrics(term_logits, gate_logits, targets, gold)
    assert m["heldout_term_top1"] == 0.5 and m["heldout_term_pair_recall_at_20"] == 0.75   # 3 of 4 gold pairs in the top 20
    assert m["heldout_gate_auc"] == 1.0
    expected = np.mean(np.logaddexp(0, gate_logits) - targets * gate_logits)
    assert m["heldout_gate_loss"] == pytest.approx(float(expected), abs=1e-6)


def test_heldout_metrics_with_no_cancer_case_held_out_is_nan_not_an_error():
    m = train_mod.heldout_metrics(np.zeros((2, 3), np.float32), np.zeros(2, np.float32), np.zeros(2, np.float32),
                                  [frozenset(), frozenset()])
    assert np.isnan(m["heldout_term_pair_recall_at_20"]) and np.isnan(m["heldout_gate_auc"])


def _encoded(tiny_modernbert_dir, n=16):
    tokenizer = load_tokenizer(str(tiny_modernbert_dir))
    texts = ["mast cell tumor present", "no evidence of neoplasia", "carcinoma malignant", "benign biopsy specimen"] * (n // 4)
    gate = np.array([1.0, 0.0, 1.0, 0.0] * (n // 4), dtype=np.float32)
    terms = np.zeros((n, 4), dtype=np.float32)
    terms[gate == 1, 0] = 1.0
    return tokenizer, tokenizer(texts)["input_ids"], terms, gate


def test_fit_runs_epochs_with_micro_batches_logs_both_losses_and_calls_the_hook(tiny_modernbert_dir):
    tokenizer, token_ids, terms, gate = _encoded(tiny_modernbert_dir)
    torch.manual_seed(0)
    model = JointClassifier(new_encoder(str(tiny_modernbert_dir)), ["a", "b", "c", "d"])
    before = model.head.weight.detach().clone()
    r = replace(TINY, epochs=2, batch_size=4, token_budget=16, gate_weight=0.5)
    seen = []
    steps, epochs = loop.fit(model, tokenizer, token_ids, terms, gate, r, "cpu",
                             on_epoch_end=lambda e: seen.append(e) or {"marker": 1.0})
    assert seen == [0, 1] and len(steps) == 8 and [e["epoch"] for e in epochs] == [0, 1]
    assert all(e["marker"] == 1.0 and np.isfinite(e["train_term_loss"]) and np.isfinite(e["train_gate_loss"]) for e in epochs)
    assert all(s["loss"] == pytest.approx(s["term_loss"] + 0.5 * s["gate_loss"]) for s in steps)
    assert not torch.equal(before, model.head.weight)


def test_fit_stops_after_max_steps(tiny_modernbert_dir):
    tokenizer, token_ids, terms, gate = _encoded(tiny_modernbert_dir)
    model = JointClassifier(new_encoder(str(tiny_modernbert_dir)), ["a", "b", "c", "d"])
    steps, epochs = loop.fit(model, tokenizer, token_ids, terms, gate, replace(TINY, epochs=3), "cpu", max_steps=3)
    assert len(steps) == 3 and len(epochs) == 1


# --- train ------------------------------------------------------------------------------------------------------------

def test_train_writes_a_pending_generation_with_no_thresholds(world, tiny_modernbert_dir):
    out = world.root / "candidate"
    manifest = train_mod.train(world.labels, world.split, out, TINY, "cpu", model_id=str(tiny_modernbert_dir))
    data = load_training_data(labels_mod.load_labels_table(world.labels), world.split)

    assert generation.read_terms(out) == data.vocab
    assert {p.name for p in out.iterdir()} == {"encoder", "head.pt", "gate_head.pt", "terms.json", "term_info.json",
                                              "manifest.json"}
    assert manifest["calibration"] == {"status": "pending"} and manifest["status"] == "candidate"
    assert manifest["parents"] == {"split_id": world.split, "labels_source": world.labels, "silver_id": "silver-test",
                                   "gold_train_snapshot": None, "gold_train_codes": 0}
    assert manifest["model"]["n_terms"] == 5 and manifest["model"]["max_length"] == 64
    assert manifest["model"]["encoder_id"] == str(tiny_modernbert_dir)
    assert manifest["recipe"]["epochs"] == 1 and manifest["recipe"]["holdout_seed"] == 1234
    training = manifest["training"]
    assert training["seed"] == 42 and training["n_holdout"] == 4 and training["n_train"] == len(data.case_ids) - 4
    assert {"heldout_gate_auc", "heldout_gate_loss", "heldout_term_top1"} <= set(training["holdout"])
    assert {"torch", "transformers", "python"} <= set(training["library_versions"])

    info = generation.read_term_info(out)
    assert info["Mast cell sarcoma"] == TERM_ROW["Mast cell sarcoma"] and len(info) == 6    # the whole taxonomy, 6 terms
    verify_manifest(out)
    assert generation.verify_spec(out, require_calibrated=False)["calibration"]["status"] == "pending"
    with pytest.raises(generation.GenerationError, match="calibrate"):
        generation.verify_spec(out)


def test_train_holds_out_cases_from_both_heads_and_never_trains_on_them(world, tiny_modernbert_dir, monkeypatch):
    seen = {}
    real = train_mod.fit_model

    def spy(texts, term_targets, gate_targets, vocab, recipe, device, model_id, max_steps=0, holdout=None):
        seen.update(texts=texts, gate=gate_targets, held=holdout.texts)
        return real(texts, term_targets, gate_targets, vocab, recipe, device, model_id, max_steps, holdout)

    monkeypatch.setattr(train_mod, "fit_model", spy)
    train_mod.train(world.labels, world.split, world.root / "candidate", TINY, "cpu", model_id=str(tiny_modernbert_dir))
    data = load_training_data(labels_mod.load_labels_table(world.labels), world.split)
    _, held_idx = split_holdout(data.gate_targets, TINY.holdout_fraction, TINY.holdout_seed)
    assert seen["held"] == [data.texts[i] for i in held_idx]
    assert seen["texts"] == [data.texts[i] for i in range(len(data.texts)) if i not in held_idx]


def test_train_refuses_a_non_empty_out_dir_a_corrupt_split_and_unknown_terms(world, tiny_modernbert_dir):
    out = world.root / "candidate"
    out.mkdir()
    (out / "old.txt").write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError):
        train_mod.train(world.labels, world.split, out, TINY, "cpu", model_id=str(tiny_modernbert_dir))

    fx.make_three_way_split_generation("corrupt", TRAIN, TRAIN[:2], TEST)       # train and calibration overlap
    with pytest.raises(GuardViolation):
        train_mod.train(world.labels, "corrupt", world.root / "c2", TINY, "cpu", model_id=str(tiny_modernbert_dir))

    table = pd.read_csv(world.labels, dtype=str, keep_default_na=False)
    table.loc[table["case_id"] == IDS[0], "matched_term"] = "Not in the taxonomy"
    table.to_csv(world.root / "unknown.csv", index=False)
    with pytest.raises(ValueError, match="not in the taxonomy"):
        train_mod.train(str(world.root / "unknown.csv"), world.split, world.root / "c3", TINY, "cpu",
                        model_id=str(tiny_modernbert_dir))


def test_train_with_a_mixed_lineage_table_fails_before_training(world, tiny_modernbert_dir):
    table = pd.read_csv(world.labels, dtype=str, keep_default_na=False)
    table.loc[0, "silver_generation"] = "silver-other"
    table.to_csv(world.root / "mixed.csv", index=False)
    with pytest.raises(ValueError, match="mixes 2 silver_generation"):
        train_mod.train(str(world.root / "mixed.csv"), world.split, world.root / "c", TINY, "cpu",
                        model_id=str(tiny_modernbert_dir))
    assert not (world.root / "c").exists()


# --- calibrate: the grid, the FP cap, the review cut -----------------------------------------------------------------

CTERMS = ["t1", "t2", "t3", "t4"]
CTERM_INFO = {"t1": ("G1", "c1"), "t2": ("G1", "c2"), "t3": ("G2", "c3"), "t4": ("G3", "c4")}
CUNCOMMON = frozenset({"G3"})


def _calibration_fixture():
    ranked = {  # case -> (term indices best first, probabilities, gate probability)
        "a": ([0, 1, 2, 3], [0.90, 0.60, 0.20, 0.10], 0.95),   # cancer t1: top-1 good, second term same group
        "b": ([0, 2, 1, 3], [0.80, 0.30, 0.20, 0.10], 0.90),   # cancer t3: completely off (G1 vs G2)
        "c": ([0, 1, 2, 3], [0.70, 0.65, 0.20, 0.10], 0.60),   # non-cancer, gate .6
        "d": ([2, 0, 1, 3], [0.50, 0.40, 0.20, 0.10], 0.10),   # non-cancer, gate .1
        "e": ([1, 0, 2, 3], [0.90, 0.20, 0.10, 0.05], 0.30),   # cancer t2, low gate .3
        "h": ([2, 0, 1, 3], [0.80, 0.55, 0.20, 0.10], 0.85),   # cancer t4 (uncommon G3), predicts G2
        "k": ([0, 2, 1, 3], [0.85, 0.45, 0.20, 0.10], 0.99),   # two terms t1 and t3
    }
    ids = list(ranked)
    scores = RankedScores(CTERMS, ids, np.array([ranked[c][0] for c in ids]), np.array([ranked[c][1] for c in ids]),
                          np.array([ranked[c][2] for c in ids]))
    labels = pd.DataFrame({
        "case_id": ["a", "b", "c", "d", "e", "f", "g", "h", "k", "k"],
        "matched_term": ["t1", "t3", "", "", "t2", "t1", "t1", "t4", "t1", "t3"],
        "matched_group": ["G1", "G2", "", "", "G1", "G1", "G1", "G3", "G1", "G2"]})
    universe = frozenset(ranked) | {"f", "g"}   # f, g: labelled, never scored (no report text)
    return scores, labels, universe


def _direct_counts(scores, labels, universe, th):
    preds = decide.decide_all(scores, th, CTERM_INFO, "g")
    summary = verdicts.summarize(verdicts.score(labels[labels["case_id"].isin(universe)], preds, CUNCOMMON))
    return np.array([summary[v] for v in verdicts.VERDICTS])


def test_pattern_counts_equal_the_real_verdict_table_at_every_grid_point():
    scores, labels, universe = _calibration_fixture()
    pc = cal.build_pattern_counts(scores, labels, universe, CUNCOMMON, CTERM_INFO)
    for gate_t in cal.GATE_GRID:
        for tau in cal.TAU_GRID:
            for rho in cal.RHO_GRID:
                th = Thresholds(gate_t, tau, rho)
                assert pc.totals(th).tolist() == _direct_counts(scores, labels, universe, th).tolist(), th


def test_fit_thresholds_returns_the_max_objective_with_good_tie_break_and_the_gate_curve():
    scores, labels, universe = _calibration_fixture()
    fitted = cal.fit_thresholds(cal.build_pattern_counts(scores, labels, universe, CUNCOMMON, CTERM_INFO))

    best_key, best_th, per_gate = None, None, {}
    for gate_t in cal.GATE_GRID:        # brute force through the real verdict table, grid order, strict improvement
        for tau in cal.TAU_GRID:
            for rho in cal.RHO_GRID:
                th = Thresholds(gate_t, tau, rho)
                counts = _direct_counts(scores, labels, universe, th)
                key = ((counts[0] + counts[1]) / counts.sum(), counts[0] / counts.sum())
                if best_key is None or key > best_key:
                    best_key, best_th = key, th
                per_gate[gate_t] = max(per_gate.get(gate_t, (-1, -1)), key)
    assert fitted["thresholds"] == best_th
    assert (fitted["gs_share"], fitted["good_share"]) == pytest.approx(best_key)
    assert [row["gate_t"] for row in fitted["curve"]] == list(cal.GATE_GRID)
    for row in fitted["curve"]:
        assert (row["best_gs"], row["best_good"]) == pytest.approx(per_gate[row["gate_t"]])
    assert max(row["best_gs"] for row in fitted["curve"]) == pytest.approx(fitted["gs_share"])


def _fp_trade_off_counts():
    """3 cases whose verdicts depend on the gate: A cancer (gate .9), B non-cancer (gate .5), C cancer (gate .3).
    Rejected: A -> FN, B -> nothing, C -> FN. Passed (top-1): A -> Good, B -> FP, C -> Good. Uncapped best is
    gate_t .05 (G+S 2/3, FP 1/3); a low cap forces gate_t > .5 (A Good, C FN: G+S 1/2, FP 0)."""
    counts = np.zeros((3, 3, 5), dtype=np.int64)
    counts[cal.REJECT, 0, 4] = counts[cal.REJECT, 2, 4] = 1
    for pattern in (cal.TOP1, cal.TOP2):
        counts[pattern, 0, 0] = counts[pattern, 2, 0] = 1
        counts[pattern, 1, 3] = 1
    return cal.PatternCounts(np.array([0.9, 0.5, 0.3]), np.array([0.9, 0.9, 0.9]), np.array([0.05] * 3), counts,
                             np.zeros(5, dtype=np.int64))


def test_fp_cap_excludes_points_over_the_cap_and_keeps_the_best_eligible_one():
    pc = _fp_trade_off_counts()
    uncapped = cal.fit_thresholds(pc)
    assert uncapped["thresholds"].gate_t == 0.05 and uncapped["verdicts"]["false_positive_share"] == pytest.approx(1 / 3)

    capped = cal.fit_thresholds(pc, 0.1)
    best_key, best_th = None, None            # brute force over the eligible points, grid order, strict improvement
    for gate_t in cal.GATE_GRID:
        for tau in cal.TAU_GRID:
            for rho in cal.RHO_GRID:
                th = Thresholds(gate_t, tau, rho)
                if pc.totals(th)[cal.FP_INDEX] / pc.totals(th).sum() > 0.1:
                    continue
                if best_key is None or pc.objective(th) > best_key:
                    best_key, best_th = pc.objective(th), th
    assert capped["thresholds"] == best_th == Thresholds(0.55, 0.1, 0.0)
    assert (capped["gs_share"], capped["good_share"]) == pytest.approx((0.5, 0.5))
    assert capped["verdicts"]["false_positive_share"] == 0.0
    assert [row["eligible"] for row in capped["curve"]] == [g > 0.5 for g in cal.GATE_GRID]
    for row in capped["curve"]:
        assert (row["best_gs"] is None) == (not row["eligible"]) and (row["best_fp_share"] is None) == (not row["eligible"])
        assert row["best_fp_share"] in (None, 0.0)


def test_no_cap_and_a_cap_nothing_exceeds_give_the_uncapped_result():
    scores, labels, universe = _calibration_fixture()
    pc = cal.build_pattern_counts(scores, labels, universe, CUNCOMMON, CTERM_INFO)
    assert cal.fit_thresholds(pc, None) == cal.fit_thresholds(pc) == cal.fit_thresholds(pc, 1.0)
    assert all(row["eligible"] for row in cal.fit_thresholds(pc)["curve"])


def test_an_infeasible_fp_cap_raises_instead_of_relaxing():
    pc = _fp_trade_off_counts()
    pc = cal.PatternCounts(pc.gate_probs, pc.p1, pc.p2, pc.counts, np.array([0, 0, 0, 1, 0], dtype=np.int64))
    assert cal.fit_thresholds(pc)["verdicts"]["false_positive"] >= 1     # every point carries a fixed FP row
    with pytest.raises(ValueError, match="not relaxing the cap"):
        cal.fit_thresholds(pc, 0.0)


@pytest.mark.parametrize("cap", [None, 0.12])
def test_fp_share_in_the_curve_and_the_chosen_verdicts_equal_verdicts_summarize(cap):
    scores, labels, universe = _calibration_fixture()
    fitted = cal.fit_thresholds(cal.build_pattern_counts(scores, labels, universe, CUNCOMMON, CTERM_INFO), cap)

    def summary(th):
        preds = decide.decide_all(scores, th, CTERM_INFO, "g")
        return verdicts.summarize(verdicts.score(labels[labels["case_id"].isin(universe)], preds, CUNCOMMON))

    chosen = fitted["thresholds"]
    expected = summary(chosen)
    assert fitted["verdicts"] == pytest.approx(expected) and fitted["verdicts"]["total"] == expected["total"]
    assert fitted["verdicts"]["good_plus_slight_share"] == pytest.approx(fitted["gs_share"])
    for row in fitted["curve"]:
        if row["eligible"]:
            assert row["best_fp_share"] == pytest.approx(summary(Thresholds(row["gate_t"], row["best_tau"], row["best_rho"]))["false_positive_share"])
            assert cap is None or row["best_fp_share"] <= cap
        assert row["at_chosen_fp_share"] == pytest.approx(summary(Thresholds(row["gate_t"], chosen.tau, chosen.rho))["false_positive_share"])


def test_grids_contain_the_required_points():
    assert 0.8 in cal.GATE_GRID and cal.GATE_GRID[0] == 0.05 and cal.GATE_GRID[-1] == 0.95
    assert len(cal.GATE_GRID) == 19 and NEVER in cal.TAU_GRID and cal.TAU_GRID[0] == 0.1
    assert cal.RHO_GRID == (0.0, 0.25, 0.5, 0.75)


def test_review_cut_queues_the_target_share_of_distinct_confidences_exactly():
    probs = np.linspace(0.05, 0.95, 20)                     # 20 distinct confidences
    for share in (0.0, 0.1, 0.2336, 0.5, 0.9):
        cut = cal.fit_review_cut(probs, share)
        assert (probs < cut).sum() == round(share * 20)
    assert cal.fit_review_cut(probs[::-1].copy(), 0.25) == cal.fit_review_cut(probs, 0.25)   # order does not matter


def test_review_cut_with_tied_confidences_gets_the_nearest_achievable_share():
    probs = np.array([0.2] * 10 + [0.5] * 10 + [0.9] * 80)
    cut = cal.fit_review_cut(probs, 0.18)
    assert cut == 0.9 and (probs < cut).mean() == 0.20           # the only shares on offer are 0, 0.10 and 0.20
    assert (probs < cal.fit_review_cut(probs, 0.0)).sum() == 0


def test_review_cut_without_accepted_cases_raises():
    with pytest.raises(ValueError, match="no accepted"):
        cal.fit_review_cut(np.array([]), 0.2)


def test_calibration_defaults_and_the_pinned_uncommon_groups_list_are_in_config():
    assert config.FP_CAP == 0.0243 and config.REVIEW_TARGET_SHARE == 0.2336
    assert config.UNCOMMON_GROUPS_TXT == config.ML_ROOT / "evaluation" / "uncommon_groups.txt"
    groups = [line for line in config.UNCOMMON_GROUPS_TXT.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(groups) == len(set(groups)) == 24
    assert not hasattr(config, "EMBEDDING_CACHE_DIR")


# --- calibrate on a trained tiny generation --------------------------------------------------------------------------

@pytest.fixture
def candidate(world, tiny_modernbert_dir):
    out = world.root / "candidate"
    train_mod.train(world.labels, world.split, out, TINY, "cpu", model_id=str(tiny_modernbert_dir))
    return out


def test_calibrate_writes_thresholds_and_binds_them_to_the_weights(world, candidate):
    block = cal.calibrate(candidate, world.labels, world.split, 1.0, 0.25, "cpu")

    values = generation.read_thresholds(candidate)
    assert set(values) == {"gate_t", "tau", "rho", "review_cut", "max_fp_share", "review_target_share"}
    assert values["gate_t"] in cal.GATE_GRID and values["tau"] in cal.TAU_GRID and values["rho"] in cal.RHO_GRID
    assert values["max_fp_share"] == 1.0 and values["review_target_share"] == 0.25

    manifest = generation.verify_spec(candidate)           # now passes the full check
    calibration = manifest["calibration"]
    assert calibration["status"] == "calibrated" and calibration["weights_sha256"] == manifest["model"]["weights_sha256"]
    assert (calibration["partition"], calibration["split_id"], calibration["labels"]) == ("calibration", world.split, world.labels)
    assert calibration["n_cases"] == 6 and calibration["n_cases_with_text"] == 6
    assert calibration["values"] == {k: values[k] for k in ("gate_t", "tau", "rho", "review_cut")}
    assert calibration["chosen_verdicts_in_sample"]["total"] > 0 and len(calibration["curve_along_gate_t"]) == 19
    assert block["values"] == calibration["values"] and block["partition_gs_share"] == calibration["partition_gs_share"]
    loaded = load_generation(candidate, "cpu")
    assert loaded.thresholds == Thresholds(values["gate_t"], values["tau"], values["rho"])


def test_calibrate_review_cut_comes_from_the_two_decimal_confidences_of_accepted_cases(world, candidate):
    cal.calibrate(candidate, world.labels, world.split, 1.0, 0.5, "cpu")
    values = generation.read_thresholds(candidate)
    # recompute the accepted calibration cases' rank-1 confidences exactly as the predictions CSV carries them
    reports = text.read_reports()
    reports = reports[reports["case_id"].isin(CALIBRATION)]
    model, tokenizer = load_joint(candidate)
    term, gate, _ = scoring.logits_for(model, tokenizer, text.build_report_texts(reports), 64, "cpu")
    ranked = scoring.rank(reports["case_id"].tolist(), model.terms, term, gate)
    accepted = ranked.gate_probs >= values["gate_t"]
    confidence = np.array([float(f"{p:.2f}") for p in ranked.probs[accepted, 0]])
    assert values["review_cut"] == cal.fit_review_cut(confidence, 0.5)
    assert read_manifest(candidate)["calibration"]["n_accepted"] == int(accepted.sum())


def test_a_generation_is_calibrated_once(world, candidate):
    cal.calibrate(candidate, world.labels, world.split, 1.0, 0.25, "cpu")
    with pytest.raises(generation.GenerationError, match="recalibrate"):
        cal.calibrate(candidate, world.labels, world.split, 1.0, 0.25, "cpu")


def test_calibrate_refuses_cases_the_generation_was_trained_on(world, candidate):
    fx.make_three_way_split_generation("leaky", train=CALIBRATION + TRAIN, calibration=[], test=TEST)
    manifest = read_manifest(candidate)
    manifest["parents"]["split_id"] = "leaky"            # the split the weights were fitted on
    (candidate / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(GuardViolation, match="train partition"):
        cal.calibrate(candidate, world.labels, world.split, 1.0, 0.25, "cpu")
    assert not (candidate / "thresholds.json").exists()


def test_calibrate_needs_labels_on_the_calibration_partition(world, candidate):
    table = pd.read_csv(world.labels, dtype=str, keep_default_na=False)
    table[table["case_id"].isin(TRAIN)].to_csv(world.root / "train_only.csv", index=False)
    with pytest.raises(ValueError, match="no rows on the calibration partition"):
        cal.calibrate(candidate, str(world.root / "train_only.csv"), world.split, 1.0, 0.25, "cpu")


def test_calibrate_refuses_a_split_without_a_calibration_partition(world, candidate):
    fx.make_three_way_split_generation("no-cal", train=TRAIN, calibration=[], test=TEST)
    with pytest.raises(ValueError, match="no calibration partition"):
        cal.calibrate(candidate, world.labels, "no-cal", 1.0, 0.25, "cpu")


# --- crossfit ---------------------------------------------------------------------------------------------------------

def test_crossfit_scores_every_train_case_once_with_a_model_that_did_not_see_it(world, tiny_modernbert_dir):
    out = world.root / "oof" / "cf.csv"
    path = crossfit(world.labels, world.split, 2, TINY, "cpu", out, model_id=str(tiny_modernbert_dir))
    assert path == out
    rows = pd.read_csv(out, dtype={"case_id": str}, keep_default_na=False)
    assert list(rows.columns) == COLUMNS == ["case_id", "fold", "gate_target", "gate_prob", "term", "term_prob"]

    data = load_training_data(labels_mod.load_labels_table(world.labels), world.split)
    assert sorted(rows["case_id"].unique()) == sorted(data.case_ids)
    assert (rows.groupby("case_id")["fold"].nunique() == 1).all()                     # one fold, one gate prob per case
    assert (rows.groupby("case_id")["gate_prob"].nunique() == 1).all() and rows["gate_prob"].between(0, 1).all()
    folds = dict(zip(data.case_ids, assign_folds(data.gate_targets, 2, TINY.seed)))
    assert all(folds[c] == f for c, f in zip(rows["case_id"], rows["fold"]))

    for i, case_id in enumerate(IDS):                       # one row per labelled term; one empty-term row otherwise
        if case_id not in folds:
            continue
        case = rows[rows["case_id"] == case_id]
        assert case["gate_target"].unique().tolist() == [int(i < 15)]
        assert sorted(case["term"]) == (sorted(_terms_of(i)) if i < 15 else [""])

    # term_prob is empty exactly when the term has no positive among the other fold's training cases
    for _, row in rows[rows["term"] != ""].iterrows():
        others = [j for j, c in enumerate(data.case_ids) if folds[c] != row["fold"]]
        in_vocab = data.term_targets[others][:, data.vocab.index(row["term"])].sum() > 0
        assert (row["term_prob"] != "") == in_vocab
        assert row["term_prob"] == "" or 0 <= float(row["term_prob"]) <= 1
    assert rows.loc[rows["term"] == SINGLETON, "term_prob"].tolist() == [""]             # a one-case term is never in its own fold's vocabulary
    assert rows.loc[rows["term"] == "", "term_prob"].eq("").all()


def test_crossfit_refuses_a_corrupt_split(world, tiny_modernbert_dir):
    fx.make_three_way_split_generation("corrupt", TRAIN, TRAIN[:2], TEST)
    with pytest.raises(GuardViolation):
        crossfit(world.labels, "corrupt", 2, TINY, "cpu", world.root / "cf.csv", model_id=str(tiny_modernbert_dir))
