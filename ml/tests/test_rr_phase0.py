"""experiments/retrieve_rerank: losses, recall metrics, report text, batching. Synthetic data only."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch

from experiments.retrieve_rerank import batching, recall_metrics, report_text, retriever
from experiments.retrieve_rerank import retriever_only_score as ros
from experiments.retrieve_rerank.eval_recall import DEFAULT_MAX_LENGTH, trained_max_length
from experiments.retrieve_rerank.openjev_pilot import chunk_sizes, read_done
from experiments.retrieve_rerank.pilot_score import (case_view, ece, in_fit_half, make_predictions, probs,
                                                     verdict_summary)


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
    assert ece(np.array([1.0, 1.0]), np.array([True, True])) == pytest.approx(0.0)
    assert ece(np.array([0.9, 0.9, 0.9, 0.9]), np.array([True, False, False, False])) == pytest.approx(0.65)


def test_fit_half_splits_calibration_style_ids_roughly_evenly():
    ids = [f"CASE-{i:05d}" for i in range(2000)]
    share = sum(in_fit_half(c) for c in ids) / len(ids)
    assert 0.4 < share < 0.6


def _record() -> dict:
    def cand(term, rank, gold, ent):
        return {"term": term, "rank": rank, "gold": gold, "probs": [0.0, ent, 0.0]}
    return {"case_id": "c1", "is_cancer": 1, "q1_yes": [0, 0.9, 0], "q1_no": [0, 0.1, 0],
            "candidates": [cand("t1", 1, False, 0.2), cand("t2", 2, True, 0.6), cand("t3", 5, False, 0.4),
                           cand("t4", 9, True, 0.8), cand("t9", None, True, 0.5)]}


def test_case_view_retriever_excludes_gold_only_and_beyond_n():
    view = case_view(_record(), "retriever", 5)
    assert view["terms"] == ["t2", "t3", "t1"]                  # sorted by entailment; t4 (rank 9) and t9 (None) out
    assert view["gold"].tolist() == [True, False, False]
    assert view["p_yes"] == pytest.approx(0.9)
    assert case_view(_record(), "retriever", 0) is None


def test_case_view_oracle_is_top_n_plus_gold_whatever_the_run_n():
    record = _record()
    assert case_view(record, "oracle", 5)["terms"] == ["t4", "t2", "t9", "t3", "t1"]
    assert case_view(record, "oracle", 2)["terms"] == ["t4", "t2", "t9", "t1"]   # t3 (rank 5) is not gold, n=2: out


def test_probs_temperature_one_is_normalised_entailment_and_argmax_is_preserved():
    view = {"ent": np.array([0.6, 0.3, 0.1])}
    assert probs(view, 1.0) == pytest.approx(view["ent"] / view["ent"].sum())
    for temperature in (0.3, 2.0, 5.0):
        p = probs(view, temperature)
        assert p.sum() == pytest.approx(1.0) and p.argmax() == 0 and list(np.argsort(-p)) == [0, 1, 2]


def test_head_runs_fp32_under_bf16_autocast():
    class Encoder(torch.nn.Module):
        config = type("Config", (), {"hidden_size": 8})()

        def forward(self, input_ids, attention_mask):
            return type("Out", (), {"last_hidden_state": torch.randn(*input_ids.shape, 8)})()

    model = retriever.TermRetriever(Encoder(), ["a", "b", "c"])
    ids, mask = torch.ones(2, 4, dtype=torch.long), torch.ones(2, 4, dtype=torch.long)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert model(ids, mask).dtype == torch.float32
        assert torch.nn.functional.linear(torch.randn(2, 8), torch.randn(3, 8)).dtype == torch.bfloat16  # autocast is on


def test_make_predictions_feed_the_verdict_scoring(tmp_path):
    data = type("Data", (), {"term_group": {"t1": "G1", "t2": "G2"}, "term_code": {"t1": "c1", "t2": "c2"}})()
    def view(case_id, is_cancer, terms, ent, p_yes):
        return {"case_id": case_id, "is_cancer": is_cancer, "terms": terms, "ent": np.array(ent),
                "gold": np.array([True] + [False] * (len(terms) - 1)), "p_yes": p_yes}
    views = [view("a", 1, ["t1", "t2"], [0.9, 0.5], 0.95),      # good, plus a second term
             view("b", 1, ["t1", "t2"], [0.9, 0.001], 0.2),     # gate-rejected -> Non-Cancer -> false negative
             view("n", 0, ["t1", "t2"], [0.9, 0.001], 0.1)]     # gate-rejected non-cancer -> true negative
    predictions = make_predictions(views, temperature=1.0, gate=0.5, second=0.3, data=data)
    assert predictions.loc[predictions["case_id"] != "a", "predicted_term"].tolist() == ["Non-Cancer", "Non-Cancer"]
    assert predictions[predictions["case_id"] == "a"]["predicted_term"].tolist() == ["t1", "t2"]
    labels = pd.DataFrame({"case_id": ["a", "b", "n"], "matched_term": ["t1", "t1", ""], "matched_group": ["G1", "G1", ""]})
    summary = verdict_summary(predictions, labels, ["a", "b", "n"], frozenset(), {"a": 1.0, "b": 1.0, "n": 1.0})
    assert summary["good"] == 1 and summary["false_negative"] == 1 and summary["false_positive"] == 0


def test_wp14_gate_passes_only_cases_with_a_cancer_row():
    baseline = pd.DataFrame({"case_id": ["rej", "rej2", "ok", "ok", "mixed", "mixed"],
                             "predicted_group": ["Non-Cancer", "Non-Cancer", "G1", "G2", "Non-Cancer", "G1"]})
    passed = ros.wp14_passes(baseline)
    assert passed == {"ok", "mixed"}                      # only-Non-Cancer rows -> rejected
    passes, threshold = ros.fit_gate(ros.WP14, [], passed)
    assert threshold is None
    assert [passes({"case_id": c}) for c in ("rej", "ok", "absent")] == [False, True, False]


def test_retriever_views_sigmoid_rank_order_and_gold():
    candidates = pd.DataFrame({"case_id": ["a", "a", "b"], "rank": ["2", "1", "1"], "term": ["t2", "t1", "t3"],
                               "score": ["0.0", "2.0", "-2.0"]})
    views = ros.retriever_views(candidates, {"a": frozenset({"t1"})})
    assert views["a"]["terms"] == ["t1", "t2"] and views["a"]["gold"].tolist() == [True, False]
    assert views["a"]["prob"] == pytest.approx([1 / (1 + math.exp(-2.0)), 0.5])
    assert views["a"]["is_cancer"] and not views["b"]["is_cancer"]


def test_retriever_only_predictions_top1_second_and_gate_rejection():
    data = type("Data", (), {"term_group": {"t1": "G1", "t2": "G2"}, "term_code": {"t1": "c1", "t2": "c2"}})()
    def view(case_id, is_cancer, probs):
        return {"case_id": case_id, "is_cancer": is_cancer, "terms": ["t1", "t2"], "prob": np.array(probs),
                "gold": np.array([True, False])}
    views = [view("hi", 1, [0.9, 0.6]), view("lo", 1, [0.9, 0.1]), view("rej", 1, [0.9, 0.6]), view("neg", 0, [0.9, 0.6])]
    predictions = ros.make_predictions(views, lambda v: v["case_id"] in {"hi", "lo", "neg"}, second=0.5, data=data)
    terms = predictions.groupby("case_id")["predicted_term"].apply(list).to_dict()
    assert terms == {"hi": ["t1", "t2"], "lo": ["t1"], "rej": ["Non-Cancer"], "neg": ["t1", "t2"]}
    assert set(predictions.loc[predictions["case_id"] == "hi", "method"]) == {ros.METHOD}
    labels = pd.DataFrame({"case_id": ["hi", "lo", "rej", "neg"], "matched_term": ["t1", "t1", "t1", ""],
                           "matched_group": ["G1", "G1", "G1", ""]})
    summary = verdict_summary(predictions, labels, ["hi", "lo", "rej", "neg"], frozenset(), dict.fromkeys(labels["case_id"], 1.0))
    # hi: good + completely_off (t2 is in no gold group); lo: good; rej: FN; neg: 2 FP rows
    assert summary["good"] == 2 and summary["completely_off"] == 1 and summary["false_negative"] == 1
    assert summary["false_positive"] == 2


def test_trained_max_length_from_train_log_else_default(tmp_path):
    assert trained_max_length(tmp_path) == DEFAULT_MAX_LENGTH
    (tmp_path / "train_log.json").write_text('{"summary": {"args": {"max_length": "8192"}}, "steps": []}', encoding="utf-8")
    assert trained_max_length(tmp_path) == 8192


def test_read_done_drops_only_a_torn_last_line(tmp_path):
    raw = tmp_path / "raw.jsonl"
    raw.write_text('{"case_id": "a"}\n{"case_id": "b"}\n{"case_id": "c", "cand', encoding="utf-8")
    assert read_done(raw) == {"a", "b"}
    assert raw.read_text(encoding="utf-8") == '{"case_id": "a"}\n{"case_id": "b"}\n'   # appends stay valid JSONL
    raw.write_text('{"case_id": "a"}\nnot json\n{"case_id": "b"}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        read_done(raw)
    assert read_done(tmp_path / "missing.jsonl") == set()
