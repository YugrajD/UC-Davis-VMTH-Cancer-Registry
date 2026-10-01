"""bronze_rr: decide rows, the calibration grid, gate helpers, the old-gate loader, the generation manifest, the
test-set guard and (on CUDA) the shared training loop. Synthetic data only."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest
import torch

from bronze_rr import calibrate, decide, encoder as enc, encoder_train, gate, generation, report_text, scoring
from bronze_rr.decide import NEVER, Thresholds
from bronze_rr.scoring import RetrieverScores
from evaluation import verdicts
from generations.manifest import ManifestError

TERMS = ["t1", "t2", "t3", "t4"]
TERM_INFO = {"t1": ("G1", "c1"), "t2": ("G1", "c2"), "t3": ("G2", "c3"), "t4": ("G3", "c4")}
UNCOMMON = frozenset({"G3"})
PRODUCTION_COLUMNS = ["case_id", "diagnosis_index", "predicted_term", "predicted_group", "predicted_code",
                      "case_presence_prob", "confidence", "group_prob", "method", "generation_id"]


def _scores(rows: dict[str, tuple[list[int], list[float]]]) -> RetrieverScores:
    ids = list(rows)
    return RetrieverScores(TERMS, ids, np.array([rows[c][0] for c in ids]), np.array([rows[c][1] for c in ids]))


# --- decide ---------------------------------------------------------------------------------------------------

def _case(gate_prob, th, probs=(0.9, 0.6)):
    return decide.case_rows("c", gate_prob, ["t1", "t3"], list(probs), th, TERM_INFO, "rr-test")


def test_rejected_case_is_one_non_cancer_row_with_the_production_method_string():
    (row,) = _case(0.30, Thresholds(0.5, 0.1, 0.0))
    assert row == {"case_id": "c", "case_presence_prob": "0.3000", "group_prob": "", "generation_id": "rr-test",
                   "diagnosis_index": 1, "predicted_term": "Non-Cancer", "predicted_group": "Non-Cancer",
                   "predicted_code": "", "confidence": "0.00", "method": "rejected_by_case_presence"}


def test_top1_only_row_carries_group_code_and_two_decimal_confidence():
    (row,) = _case(0.95, Thresholds(0.5, NEVER, 0.0))
    assert (row["predicted_term"], row["predicted_group"], row["predicted_code"]) == ("t1", "G1", "c1")
    assert row["confidence"] == "0.90" and row["case_presence_prob"] == "0.9500" and row["method"] == "retriever_top1"
    assert row["diagnosis_index"] == 1 and row["group_prob"] == ""


def test_second_term_needs_both_tau_and_rho():
    assert len(_case(0.9, Thresholds(0.5, 0.5, 0.5))) == 2          # p2 .6 >= .5, ratio .667 >= .5
    assert len(_case(0.9, Thresholds(0.5, 0.7, 0.0))) == 1          # p2 below tau
    assert len(_case(0.9, Thresholds(0.5, 0.5, 0.75))) == 1         # ratio .667 below rho
    rows = _case(0.9, Thresholds(0.5, 0.5, 0.5))
    assert [r["diagnosis_index"] for r in rows] == [1, 2] and rows[1]["predicted_term"] == "t3"
    assert rows[1]["confidence"] == "0.60" and rows[1]["predicted_group"] == "G2"


def test_gate_threshold_is_inclusive_like_production():
    assert _case(0.5, Thresholds(0.5, NEVER, 0.0))[0]["method"] == "retriever_top1"


def test_decide_all_columns_order_methods_and_empty_text():
    scores = _scores({"a": ([0, 2, 1, 3], [0.9, 0.6, 0.2, 0.1]), "n": ([1, 0, 2, 3], [0.8, 0.5, 0.2, 0.1]),
                      "empty": ([0, 1, 2, 3], [0.9, 0.8, 0.2, 0.1]), "nogate": ([0, 1, 2, 3], [0.9, 0.8, 0.2, 0.1])})
    out = decide.decide_all(scores, {"a": 0.9, "n": 0.1, "empty": 0.9}, {"a", "n", "nogate"},
                            Thresholds(0.5, 0.5, 0.5), TERM_INFO, "rr-x")
    assert list(out.columns) == PRODUCTION_COLUMNS == decide.COLUMNS
    assert "empty" not in set(out["case_id"])                                     # no report text -> no row
    assert out.groupby("case_id").size().to_dict() == {"a": 2, "n": 1, "nogate": 1}
    assert out.loc[out["case_id"] == "nogate", "method"].tolist() == ["rejected_by_case_presence"]  # no gate prob = 0
    assert set(out["method"]) <= {"retriever_top1", "rejected_by_case_presence"}
    assert "unidentified_cancer" not in set(out["method"]) and "Unidentified Group" not in set(out["predicted_term"])
    assert (out["generation_id"] == "rr-x").all()


def test_rejected_cancer_case_scores_as_false_negative():
    scores = _scores({"a": ([0, 1, 2, 3], [0.9, 0.6, 0.2, 0.1])})
    out = decide.decide_all(scores, {"a": 0.1}, {"a"}, Thresholds(0.5, NEVER, 0.0), TERM_INFO, "g")
    labels = pd.DataFrame({"case_id": ["a"], "matched_term": ["t1"], "matched_group": ["G1"]})
    assert verdicts.score(labels, out, UNCOMMON)["verdict"].tolist() == ["false_negative"]


def test_cases_with_text_follows_the_production_definition_of_empty():
    columns = list(report_text.REPORT_COLUMNS) + ["case_id"]
    rows = [{**{c: "" for c in columns}, "case_id": case, **cells} for case, cells in
            [("abstract_only", {"CLINICAL ABSTRACT": "x"}), ("comment", {"COMMENT": "x"}), ("hist", {"HISTOPATHOLOGICAL SUMMARY": "x"}),
             ("blank", {"COMMENT": "  "}), ("ancillary", {"ANCILLARY TESTS": "x"})]]
    assert report_text.cases_with_text(pd.DataFrame(rows)) == {"comment", "hist", "ancillary"}


# --- calibrate ------------------------------------------------------------------------------------------------

def _calibration_fixture():
    ranked = {  # case -> (term indices best first, probabilities)
        "a": ([0, 1, 2, 3], [0.90, 0.60, 0.20, 0.10]),   # cancer t1: top-1 good, second term same group
        "b": ([0, 2, 1, 3], [0.80, 0.30, 0.20, 0.10]),   # cancer t3: top-1 slight? no, G1 vs G2 -> completely off
        "c": ([0, 1, 2, 3], [0.70, 0.65, 0.20, 0.10]),   # non-cancer, gate .6
        "d": ([2, 0, 1, 3], [0.50, 0.40, 0.20, 0.10]),   # non-cancer, gate .1
        "e": ([1, 0, 2, 3], [0.90, 0.20, 0.10, 0.05]),   # cancer t2, low gate .3
        "f": ([0, 1, 2, 3], [0.90, 0.20, 0.10, 0.05]),   # cancer t1 but no report text
        "h": ([2, 0, 1, 3], [0.80, 0.55, 0.20, 0.10]),   # cancer t4 (uncommon G3), predicts G2
        "k": ([0, 2, 1, 3], [0.85, 0.45, 0.20, 0.10]),   # two terms t1 and t3
    }
    gate_probs = {"a": 0.95, "b": 0.90, "c": 0.60, "d": 0.10, "e": 0.30, "f": 0.95, "h": 0.85, "k": 0.99}
    has_text = frozenset(set(ranked) - {"f"})
    labels = pd.DataFrame({
        "case_id": ["a", "b", "c", "d", "e", "f", "g", "h", "k", "k"],
        "matched_term": ["t1", "t3", "", "", "t2", "t1", "t1", "t4", "t1", "t3"],
        "matched_group": ["G1", "G2", "", "", "G1", "G1", "G1", "G3", "G1", "G2"]})
    universe = frozenset(ranked) | {"g"}   # g: labelled, never scored (no report)
    return _scores(ranked), gate_probs, has_text, labels, universe


def _direct_counts(scores, gate_probs, has_text, labels, universe, th):
    preds = decide.decide_all(scores, gate_probs, has_text, th, TERM_INFO, "g")
    table = verdicts.score(labels[labels["case_id"].isin(universe)], preds, UNCOMMON)
    summary = verdicts.summarize(table)
    return np.array([summary[v] for v in verdicts.VERDICTS])


def test_pattern_counts_equal_the_real_verdict_table_at_every_grid_point():
    scores, gate_probs, has_text, labels, universe = _calibration_fixture()
    pc = calibrate.build_pattern_counts(scores, np.array([gate_probs[c] for c in scores.case_ids]), has_text, labels,
                                        universe, UNCOMMON, TERM_INFO)
    for gate_t in calibrate.GATE_GRID:
        for tau in calibrate.TAU_GRID:
            for rho in calibrate.RHO_GRID:
                th = Thresholds(gate_t, tau, rho)
                assert pc.totals(th).tolist() == _direct_counts(scores, gate_probs, has_text, labels, universe, th).tolist(), th


def test_fit_thresholds_returns_the_max_objective_with_good_tie_break_and_the_gate_curve():
    scores, gate_probs, has_text, labels, universe = _calibration_fixture()
    pc = calibrate.build_pattern_counts(scores, np.array([gate_probs[c] for c in scores.case_ids]), has_text, labels,
                                        universe, UNCOMMON, TERM_INFO)
    fitted = calibrate.fit_thresholds(pc)

    best_key, best_th, per_gate = None, None, {}
    for gate_t in calibrate.GATE_GRID:        # brute force through the real verdict table, grid order, strict improvement
        for tau in calibrate.TAU_GRID:
            for rho in calibrate.RHO_GRID:
                th = Thresholds(gate_t, tau, rho)
                counts = _direct_counts(scores, gate_probs, has_text, labels, universe, th)
                key = ((counts[0] + counts[1]) / counts.sum(), counts[0] / counts.sum())
                if best_key is None or key > best_key:
                    best_key, best_th = key, th
                per_gate[gate_t] = max(per_gate.get(gate_t, (-1, -1)), key)
    assert fitted["thresholds"] == best_th
    assert (fitted["gs_share"], fitted["good_share"]) == pytest.approx(best_key)
    assert [row["gate_t"] for row in fitted["curve"]] == list(calibrate.GATE_GRID)
    for row in fitted["curve"]:
        assert (row["best_gs"], row["best_good"]) == pytest.approx(per_gate[row["gate_t"]])
    assert max(row["best_gs"] for row in fitted["curve"]) == pytest.approx(fitted["gs_share"])


def test_grids_contain_the_required_points():
    assert 0.8 in calibrate.GATE_GRID and calibrate.GATE_GRID[0] == 0.05 and calibrate.GATE_GRID[-1] == 0.95
    assert len(calibrate.GATE_GRID) == 19 and NEVER in calibrate.TAU_GRID and calibrate.TAU_GRID[0] == 0.1
    assert calibrate.RHO_GRID == (0.0, 0.25, 0.5, 0.75)


# --- gate helpers ---------------------------------------------------------------------------------------------

def test_holdout_is_stratified_deterministic_and_a_partition():
    targets = np.array([0.0] * 100 + [1.0] * 40)
    train_idx, held_idx = gate.split_holdout(targets)
    assert (train_idx, held_idx) == gate.split_holdout(targets)
    assert gate.split_holdout(targets, seed=1)[1] != held_idx
    assert sorted(train_idx + held_idx) == list(range(140)) and not set(train_idx) & set(held_idx)
    assert targets[held_idx].sum() == 2 and (targets[held_idx] == 0).sum() == 5       # 5% of each class
    assert held_idx == sorted(held_idx) and train_idx == sorted(train_idx)


def test_roc_auc():
    assert gate.roc_auc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == 1.0
    assert gate.roc_auc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([0, 0, 1, 1])) == 0.0
    assert gate.roc_auc(np.array([0.5, 0.5, 0.5, 0.5]), np.array([0, 1, 0, 1])) == 0.5
    scores, pos = np.array([0.1, 0.4, 0.35, 0.8]), np.array([0, 0, 1, 1])
    assert gate.roc_auc(scores, pos) == pytest.approx(0.75)
    assert np.isnan(gate.roc_auc(scores, np.zeros(4)))


def test_old_gate_probs_are_one_per_case_and_absent_cases_read_zero(tmp_path):
    csv = tmp_path / "p.csv"
    csv.write_text("case_id,diagnosis_index,case_presence_prob,method\n"
                   "a,1,0.9100,retriever\na,2,0.9100,retriever\nb,1,0.0200,rejected\nz,1,,empty\n", encoding="utf-8")
    probs = gate.load_old_gate_probs(csv)
    assert probs == {"a": 0.91, "b": 0.02}
    assert gate.probs_for(["a", "b", "missing"], probs).tolist() == [0.91, 0.02, 0.0]
    csv.write_text("case_id,case_presence_prob\na,0.9100\na,0.8000\n", encoding="utf-8")
    with pytest.raises(ValueError, match="more than one"):
        gate.load_old_gate_probs(csv)


# --- generation manifest --------------------------------------------------------------------------------------

def _fake_model(directory):
    (directory / "encoder").mkdir(parents=True)
    (directory / "encoder" / "model.safetensors").write_bytes(b"weights" + directory.name.encode())
    (directory / "head.pt").write_bytes(b"head")
    (directory / "terms.json").write_text('["t1"]', encoding="utf-8")
    (directory / "train_log.json").write_text('{"summary": {"args": {"seed": "7"}}}', encoding="utf-8")


@pytest.mark.parametrize("new_gate", [True, False])
def test_generation_manifest_hashes_every_file_and_detects_tampering(tmp_path, new_gate):
    retriever_dir, gate_dir = tmp_path / "ret", tmp_path / "gt"
    _fake_model(retriever_dir)
    _fake_model(gate_dir)
    old_csv = tmp_path / "old.csv"
    old_csv.write_text("case_id,case_presence_prob\na,0.9\n", encoding="utf-8")
    uncommon = tmp_path / "u.txt"
    uncommon.write_text("G3\n", encoding="utf-8")
    th = Thresholds(0.8, 0.4, 0.25)

    directory = generation.write_generation(
        tmp_path / "gens", generation_id=generation.new_generation_id(), retriever_dir=retriever_dir,
        gate_dir=gate_dir if new_gate else None, old_gate_csv=None if new_gate else old_csv, thresholds=th,
        calibration={"status": "calibrated"}, split_id="s", silver_id="v", uncommon_groups_txt=uncommon, max_length=4096)
    assert re.fullmatch(r"rr-\d{8}T\d{6}Z", directory.name)
    manifest, loaded = generation.load_generation_info(directory)
    assert loaded == th and manifest["generation_id"] == directory.name and manifest["max_length"] == 4096
    assert manifest["retriever"]["seed"] == "7" and manifest["calibration"]["values"] == th.as_dict()
    assert all(re.fullmatch(r"[0-9a-f]{64}", h) for h in manifest["files"].values())
    expected = {"retriever/encoder/model.safetensors", "retriever/head.pt", "retriever/terms.json",
                "retriever/train_log.json", "thresholds.json", "uncommon_groups.txt"}
    expected |= {"gate/encoder/model.safetensors", "gate/head.pt", "gate/terms.json", "gate/train_log.json"} if new_gate \
        else {"gate/gate_source.json"}
    assert set(manifest["files"]) == expected

    (directory / "thresholds.json").write_text('{"gate_t": 0.1, "tau": 0.1, "rho": 0.0}', encoding="utf-8")
    with pytest.raises(ManifestError):
        generation.load_generation_info(directory)


def test_write_generation_needs_exactly_one_gate_source(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        generation.write_generation(tmp_path, generation_id="rr-x", retriever_dir=tmp_path, gate_dir=None,
                                    old_gate_csv=None, thresholds=Thresholds(0.5, 0.5, 0.5), calibration={},
                                    split_id="s", silver_id="v", uncommon_groups_txt=tmp_path, max_length=1)


# --- test-set guard -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("partition", ["test", "all"])
def test_scoring_test_partition_needs_the_explicit_flag(partition):
    with pytest.raises(SystemExit, match="i-understand-this-spends-the-test-set"):
        scoring.check_partition_allowed(partition, allow_test=False)
    scoring.check_partition_allowed(partition, allow_test=True)


@pytest.mark.parametrize("partition", ["train", "calibration"])
def test_scoring_non_test_partitions_need_no_flag(partition):
    scoring.check_partition_allowed(partition, allow_test=False)


def test_retriever_frame_is_numbers_only_and_ranked():
    logits = np.linspace(-5.0, -4.0, 60, dtype=np.float32).reshape(1, 60)
    logits[0, 5], logits[0, 9] = 3.0, 2.0
    frame = scoring.retriever_frame(["a"], logits)
    assert list(frame.columns)[:3] == ["case_id", "t00", "t01"] and frame.shape[1] == 1 + 2 * scoring.TOP_K
    assert frame.loc[0, ["t00", "t01"]].tolist() == [5, 9]
    assert frame.loc[0, ["p00", "p01"]].to_numpy(dtype=float) == pytest.approx(1 / (1 + np.exp(-np.array([3.0, 2.0]))))
    back = scoring.retriever_scores_from_frame(frame, [f"x{i}" for i in range(60)])
    assert back.ranked_terms(0, 2) == ["x5", "x9"] and back.probs.shape == (1, scoring.TOP_K)
    assert (np.diff(back.probs[0]) <= 0).all()


# --- shared training loop (needs CUDA: the loop autocasts on "cuda") --------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="encoder_train.fit autocasts on CUDA")
def test_fit_runs_an_epoch_logs_it_and_calls_the_epoch_hook(tiny_bert_dir):
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(tiny_bert_dir))
    model = enc.PooledClassifier(AutoModel.from_pretrained(str(tiny_bert_dir)), 1).cuda()
    texts = ["mast cell tumor present", "no evidence of neoplasia", "carcinoma malignant", "benign biopsy specimen"] * 4
    token_ids = tokenizer(texts)["input_ids"]
    targets = np.array([[1.0], [0.0], [1.0], [0.0]] * 4, dtype=np.float32)
    seen = []
    steps, epochs = encoder_train.fit(
        model, tokenizer, token_ids, targets, enc.bce_loss, epochs=2, lr=1e-4, head_lr=1e-3, weight_decay=0.01,
        warmup_frac=0.1, batch_size=4, token_budget=64, seed=0, on_epoch_end=lambda e: seen.append(e) or {"marker": 1.0})
    assert seen == [0, 1] and len(steps) == 8 and [e["epoch"] for e in epochs] == [0, 1]
    assert all(e["marker"] == 1.0 and np.isfinite(e["train_loss"]) for e in epochs)
    logits, over = scoring.logits_for(model, tokenizer, texts, max_length=128, token_budget=64, device="cuda")
    assert logits.shape == (16, 1) and over == 0
