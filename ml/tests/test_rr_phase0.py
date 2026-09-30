"""experiments/retrieve_rerank: losses, recall metrics, report text, batching. Synthetic data only."""

from __future__ import annotations

import math

import pandas as pd
import pytest
import torch

from experiments.retrieve_rerank import batching, recall_metrics, report_text, retriever
from experiments.retrieve_rerank.openjev_pilot import chunk_sizes
from experiments.retrieve_rerank.pilot_score import ece, in_fit_half


def _ce(z_gold: float, z_negs: list[float]) -> float:
    """-log softmax of the gold logit over {gold} + negatives, by hand."""
    return -(z_gold - math.log(math.exp(z_gold) + sum(math.exp(z) for z in z_negs)))


def test_softmax_loss_masks_other_positives():
    logits = torch.tensor([[2.0, 1.0, 0.5, -1.0]])
    targets = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    expected = (_ce(2.0, [0.5, -1.0]) + _ce(1.0, [0.5, -1.0])) / 2
    assert retriever.softmax_loss(logits, targets).item() == pytest.approx(expected, rel=1e-5)


def test_softmax_loss_ignores_other_positive_logit_for_each_term():
    targets = torch.tensor([[1.0, 1.0, 0.0]])
    single = torch.tensor([[1.0, 0.0, 0.0]])
    # Raising the OTHER positive's logit must not add to the first term's loss: only the second term's
    # own CE (which it then wins more easily) changes.
    low = retriever.softmax_loss(torch.tensor([[1.0, -5.0, 0.0]]), targets).item()
    high = retriever.softmax_loss(torch.tensor([[1.0, 5.0, 0.0]]), targets).item()
    assert high < low
    assert low == pytest.approx((_ce(1.0, [0.0]) + _ce(-5.0, [0.0])) / 2, rel=1e-5)
    assert retriever.softmax_loss(single, torch.tensor([[1.0, 0.0, 0.0]])).item() == pytest.approx(_ce(1.0, [0.0, 0.0]), rel=1e-5)


def test_softmax_loss_averages_per_case_then_over_cases():
    logits = torch.tensor([[1.0, 0.0, 0.0], [0.0, 3.0, 1.0]])
    targets = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
    case_a = _ce(1.0, [0.0, 0.0])
    case_b = (_ce(3.0, [0.0]) + _ce(1.0, [0.0])) / 2  # each gold term competes only with the one non-gold term
    assert retriever.softmax_loss(logits, targets).item() == pytest.approx((case_a + case_b) / 2, rel=1e-5)


def test_bce_loss_is_sum_over_terms_mean_over_cases():
    logits = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    targets = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    assert retriever.bce_loss(logits, targets).item() == pytest.approx(2 * math.log(2), rel=1e-6)


def test_recall_metrics_toy():
    ranked = {"a": ["t1", "t2", "t3", "t4"], "b": ["t3", "t1", "t2", "t4"], "c": ["t4", "t3", "t2", "t1"]}
    gold = {"a": frozenset({"t1"}), "b": frozenset({"t1", "t9"}), "c": frozenset({"t1"})}
    term_group = {"t1": "G1", "t2": "G1", "t3": "G2", "t4": "G3", "t9": "G4"}
    train_freq = {"t1": 60, "t2": 3, "t3": 10, "t4": 25}
    report = recall_metrics.recall_report(ranked, gold, term_group, train_freq, ns=(1, 2, 4))
    assert report["n_pairs"] == 4 and report["n_oov_pairs"] == 1
    assert report["pair_recall"] == {1: 1 / 4, 2: 2 / 4, 4: 3 / 4}      # t9 never ranked: always a miss
    assert report["case_recall"] == {1: 1 / 3, 2: 1 / 3, 4: 2 / 3}     # b needs t9 (out of vocabulary); c has t1 only at rank 4
    assert report["group_pair_recall"][1] == pytest.approx(1 / 4)       # only a's top-1 group matches; b, b's t9 and c miss
    assert report["by_case_size"]["single_term"]["n_cases"] == 2
    assert report["by_case_size"]["multi_term"]["pair_recall"][4] == 1 / 2
    assert report["pair_recall_by_train_freq"][">=50"]["n_pairs"] == 3
    assert report["pair_recall_by_train_freq"]["<5"]["n_pairs"] == 1    # the out-of-vocabulary t9 (frequency 0)


def test_freq_buckets():
    assert [recall_metrics.freq_bucket(n) for n in (0, 4, 5, 19, 20, 49, 50, 900)] == \
        ["<5", "<5", "5-19", "5-19", "20-49", "20-49", ">=50", ">=50"]


def test_report_text_field_order_and_headings():
    row = pd.Series({c: "" for c in report_text.REPORT_COLUMNS})
    row["ADDENDUM"] = "add x"
    row["CLINICAL ABSTRACT"] = "  abs x "
    row["FINAL COMMENT"] = float("nan")
    row["HISTOPATHOLOGICAL SUMMARY"] = "hist x"
    assert report_text.build_report_text(row) == (
        "CLINICAL ABSTRACT: abs x\nHISTOPATHOLOGICAL SUMMARY: hist x\nADDENDUM: add x")
    assert report_text.REPORT_COLUMNS == (
        "CLINICAL ABSTRACT", "GROSS DESCRIPTION", "HISTOPATHOLOGICAL SUMMARY", "COMMENT",
        "FINAL COMMENT", "ANCILLARY TESTS", "ADDENDUM", "ADDITIONAL INFORMATION")


def test_build_report_texts_keys_by_case_id():
    df = pd.DataFrame([{**{c: "" for c in report_text.REPORT_COLUMNS}, "case_id": " CASE-1 ", "COMMENT": "c"}])
    assert report_text.build_report_texts(df) == {"CASE-1": "COMMENT: c"}


def test_micro_batches_respect_token_budget_and_cover_all_cases():
    lengths = [100, 4096, 4096, 50, 4096, 4096, 4096]
    batches = batching.micro_batches(list(range(len(lengths))), lengths, 8192)
    assert sorted(i for b in batches for i in b) == list(range(len(lengths)))
    assert all(max(lengths[i] for i in b) * len(b) <= 8192 for b in batches)


def test_bucketed_steps_partition_the_cases():
    steps = batching.bucketed_steps(list(range(100)), 8, seed=1, block_steps=4)
    assert sorted(i for s in steps for i in s) == list(range(100))
    assert all(len(s) <= 8 for s in steps)


def test_encode_reports_truncates_keeping_closing_token_and_counts_over():
    class Tok:
        def __call__(self, texts, add_special_tokens, truncation):
            return {"input_ids": [[0] + [5] * (len(t) - 2) + [9] for t in texts]}
    ids, over = batching.encode_reports(Tok(), ["x" * 4, "x" * 10], max_length=6)
    assert over == 1 and ids[0] == [0, 5, 5, 9] and ids[1] == [0, 5, 5, 5, 5, 9]


def test_chunk_sizes_balanced_and_bounded():
    for total in (3, 12, 13, 32, 33):
        sizes = chunk_sizes(total, 12)
        assert sum(sizes) == total and max(sizes) <= 12 and max(sizes) - min(sizes) <= 1


def test_ece_perfectly_calibrated_and_overconfident():
    import numpy as np
    assert ece(np.array([1.0, 1.0]), np.array([True, True])) == pytest.approx(0.0)
    assert ece(np.array([0.9, 0.9, 0.9, 0.9]), np.array([True, False, False, False])) == pytest.approx(0.65)


def test_fit_half_splits_calibration_style_ids_roughly_evenly():
    ids = [f"CASE-{i:05d}" for i in range(2000)]
    share = sum(in_fit_half(c) for c in ids) / len(ids)
    assert 0.4 < share < 0.6
