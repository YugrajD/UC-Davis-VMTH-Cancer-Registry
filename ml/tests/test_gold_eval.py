"""evaluation/gold_eval.py: the four results on a hand-built gold store, ledger, silver and predictions.

Scenario (terms from the synthetic taxonomy; weights hand-computed):

  case  origin        stratum           gold                         silver (stage)                       bronze (top conf)
  E1    eval_batch    Mast Cell Tumors  MCT malignant                MCT malignant (tier1_exact) + no_signal MCT sarcoma 0.95
  E2    eval_batch    Mast Cell Tumors  MCT malignant + Fibrosarcoma MCT sarcoma (tier2_fuzzy)             MCT malignant 0.55, SCC 0.30
  E3    eval_batch    Rare Sarcomas     Fibrosarcoma                 Uncertain (vague)                     Fibrosarcoma 0.15
  E4    eval_batch    no_cancer         NO_CANCER                    no_signal                             Non-Cancer
  E5    eval_batch    no_cancer         NO_CANCER                    SCC (tier3_llm)                       Non-Cancer
  U1    random_slice  (rate 0.25)       MCT malignant                —                                     Non-Cancer
  E6    review_queue on test (never used)          T1  review_queue on train (gold-train, never used)
  E7    drawn in batch 2, not yet reviewed

Two-batch series: batch 1 draws E1 (N_h 10), E3 (N_h 4), E4 (N_h 20); batch 2 draws E2, E5, E7.
Pooled weights over reviewed cases: Mast Cell Tumors 10/2 = 5, Rare Sarcomas 4/1 = 4 (E7 does not
dilute it), no_cancer 20/2 = 10; U1 = 1/0.25 = 4. Summing per-batch sample_weight would give E1 10.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import config
import io_utils
from coding.rule import UnknownDecisionError
from evaluation import gold_eval, intervals
from manual_audit.cause_pass import CAUSE_STORE_FIELDS
from manual_audit.eval_batch import EVAL_BATCH_LEDGER_FIELDS

from . import fixtures as fx

GEN = "gold-eval-gen"
SILVER = "gold-eval-silver"
SPLIT = "gold-eval-split"
UNCOMMON = ["Rare Sarcomas", "Rare Carcinomas"]
N_BOOT = 50

MCT_M = ("Mast cell tumor, malignant", "Mast Cell Tumors", "1001/3")
MCT_S = ("Mast cell sarcoma", "Mast Cell Tumors", "1003/3")
FIBRO = ("Fibrosarcoma, NOS", "Rare Sarcomas", "2001/3")
SCC = ("Squamous cell carcinoma, NOS", "Rare Carcinomas", "3001/3")
NONE = ("", "", "")


def _gold(case_id, origin, term=None, **extra):
    t, g, c = term if term else ("", "", "NO_CANCER")
    return {"case_id": case_id, "term": t, "group": g, "code": c, "origin": origin, **extra}


def _silver(case_id, number, term, method, stage):
    t, g, c = term
    return (case_id, number, "invented diagnosis text", t, g, c, "", method, 1.0, stage)


def _bronze(case_id, rank, term, confidence):
    if term is None:
        return (case_id, rank, "Non-Cancer", "Non-Cancer", "", "0.1000", "0.00", "0.0000",
                "rejected_by_case_presence", GEN)
    t, g, c = term
    return (case_id, rank, t, g, c, "0.9000", f"{confidence:.2f}", "0.9000", "label_presence", GEN)


def _ledger_row(case_id, batch, stratum, N_h, n_h, mode):
    row = {f: "" for f in EVAL_BATCH_LEDGER_FIELDS}
    row.update(case_id=case_id, batch_id=batch, split_id=SPLIT, silver_id=SILVER, stratum=stratum,
               N_h=N_h, n_h=n_h, sample_weight=f"{N_h / n_h:.6f}", review_mode=mode)
    return row


def build(root, monkeypatch, review_mode="app_non_blind"):
    fx.point_evaluation_config_at(monkeypatch, root)
    fx.make_three_way_split_generation(SPLIT, train=["T1"], calibration=["C1"],
                                       test=["E1", "E2", "E3", "E4", "E5", "E6", "E7"])
    fx.make_gold_store_csv([
        _gold("E1", "eval_batch", MCT_M),
        _gold("E2", "eval_batch", MCT_M), _gold("E2", "eval_batch", FIBRO),
        _gold("E3", "eval_batch", FIBRO),
        _gold("E4", "eval_batch"),
        _gold("E5", "eval_batch"),
        _gold("U1", "random_slice", MCT_M, slice_rate="0.25", upload_period="2026-Q1"),
        _gold("E6", "review_queue", SCC),
        _gold("T1", "review_queue", SCC),
    ])
    io_utils.write_csv(pd.DataFrame([
        _ledger_row("E1", "b1", "Mast Cell Tumors", 10, 1, review_mode),
        _ledger_row("E3", "b1", "Rare Sarcomas", 4, 1, review_mode),
        _ledger_row("E4", "b1", "no_cancer", 20, 1, review_mode),
        _ledger_row("E2", "b2", "Mast Cell Tumors", 10, 1, review_mode),
        _ledger_row("E5", "b2", "no_cancer", 20, 1, review_mode),
        _ledger_row("E7", "b2", "Rare Sarcomas", 4, 1, review_mode),
    ], columns=EVAL_BATCH_LEDGER_FIELDS), config.EVAL_BATCH_LEDGER_CSV)
    fx.make_silver_generation(SILVER, [
        _silver("E1", 1, MCT_M, "Exact", "tier1_exact"),
        _silver("E1", 2, NONE, "No Match", "no_signal"),
        _silver("E2", 1, MCT_S, "Fuzzy", "tier2_fuzzy"),
        _silver("E3", 1, NONE, "Uncertain", "tier3_llm"),
        _silver("E4", 1, NONE, "No Match", "no_signal"),
        _silver("E5", 1, SCC, "LLM", "tier3_llm"),
        _silver("E6", 1, SCC, "Exact", "tier1_exact"),
    ])
    predictions = fx.make_bronze_predictions_csv(root / "predictions.csv", [
        _bronze("E1", 1, MCT_S, 0.95),
        _bronze("E2", 1, MCT_M, 0.55), _bronze("E2", 2, SCC, 0.30),
        _bronze("E3", 1, FIBRO, 0.15),
        _bronze("E4", 1, None, 0),
        _bronze("E5", 1, None, 0),
        _bronze("U1", 1, None, 0),
        _bronze("E6", 1, MCT_M, 0.99),
    ])
    fx.make_minimal_generation(config.REPORT_MAPPING_CURRENT_DIR, GEN, UNCOMMON)
    config.CAUSE_STORE_CSV.parent.mkdir(parents=True, exist_ok=True)
    io_utils.write_csv(pd.DataFrame([
        {"case_id": "E2", "gold_code": "2001/3", "method": "silver", "input_supports": "yes"},
        {"case_id": "E3", "gold_code": "2001/3", "method": "silver", "input_supports": "no"},
        {"case_id": "E1", "gold_code": "1001/3", "method": "bronze", "input_supports": "yes"},
    ], columns=CAUSE_STORE_FIELDS).fillna(""), config.CAUSE_STORE_CSV)
    return gold_eval.run(predictions, SILVER, SPLIT, n_boot=N_BOOT)


@pytest.fixture
def result(tmp_path, monkeypatch):
    return build(tmp_path, monkeypatch)


def _row(table, **keys):
    mask = np.ones(len(table), dtype=bool)
    for column, value in keys.items():
        mask &= (table[column] == value).to_numpy()
    assert mask.sum() == 1, keys
    return table[mask].iloc[0]


def test_weights_pool_the_two_batch_series(result):
    cases = result["cases"]
    assert cases["weight"].to_dict() == {"E1": 5.0, "E2": 5.0, "E3": 4.0, "E4": 10.0, "E5": 10.0, "U1": 4.0}
    assert cases.loc["U1", "stratum"] == "random_slice:2026-Q1"
    assert result["counts"]["unreviewed_ledger_cases"] == 1


def test_queue_and_gold_train_rows_are_never_used(result):
    assert set(result["cases"].index) == {"E1", "E2", "E3", "E4", "E5", "U1"}
    assert not {"E6", "T1"} & set(result["misses"]["case_id"])
    assert result["counts"]["cases"] == 6 and result["counts"]["gold_codes"] == 5


def test_result_1_silver_vs_gold_by_stage(result):
    table = result["silver_vs_gold"]
    assert _row(table, stage="tier1_exact")[["codes", "good"]].tolist() == [1, 1.0]
    fuzzy = _row(table, stage="tier2_fuzzy")  # slightly_off + the uncovered Fibrosarcoma FN, weight 5 each
    assert (fuzzy["codes"], fuzzy["good"], fuzzy["gs"], fuzzy["false_negative"]) == (2, 0.0, 0.5, 0.5)
    assert _row(table, stage="tier3_llm")["false_positive"] == 1.0
    no_signal = _row(table, stage="no_signal")  # E4: a correct abstention only
    assert (no_signal["codes"], no_signal["tn_cases"]) == (0, 1)
    overall = _row(table, stage="all")  # weights 5 + 5 + 5 + 4 + 10 = 29
    assert overall["cases"] == 4 and overall["codes"] == 5
    assert overall["good"] == pytest.approx(5 / 29) and overall["gs"] == pytest.approx(10 / 29)
    assert result["counts"]["cases_without_silver"] == 1  # U1


def test_vague_rows_are_reported_as_their_own_stage(result):
    vague = _row(result["silver_vs_gold"], stage="vague:tier3_llm_uncertain")
    # Silver would have said "no code" for E3; gold has Fibrosarcoma -> one FN, weight 4.
    assert (vague["cases"], vague["codes"], vague["false_negative"]) == (1, 1, 1.0)
    assert gold_eval.stage_label("tier3_llm", "Uncertain") == "vague:tier3_llm_uncertain"
    assert gold_eval.stage_label("tier3_no_candidates", "No Match") == "vague:tier3_no_candidates"
    assert gold_eval.stage_label("tier3_llm", "No Match") == "tier3_llm_declined"
    assert gold_eval.stage_label("tier3_llm", "LLM") == "tier3_llm"
    with pytest.raises(UnknownDecisionError):
        gold_eval.stage_label("tier3_llm", "Exact")


def test_result_2_bronze_vs_gold_by_band(result):
    table = result["bronze_vs_gold"]
    # E1 slight (5); E2 good + slight (5, 5); E3 good (4); U1 FN (4): total weight 23.
    overall = _row(table, band="all")
    assert overall["good"] == pytest.approx(9 / 23) and overall["gs"] == pytest.approx(19 / 23)
    assert overall["tn_cases"] == 2  # E4, E5
    assert _row(table, band="0.8-1.0")[["cases", "good", "gs"]].tolist() == [1, 0.0, 1.0]
    assert _row(table, band="0.4-0.6")[["codes", "good"]].tolist() == [2, 0.5]
    assert _row(table, band="0.0-0.2")["good"] == 1.0
    non_cancer = _row(table, band="non_cancer")
    assert (non_cancer["codes"], non_cancer["false_negative"], non_cancer["tn_cases"]) == (1, 1.0, 2)

    # The Kish/Wilson interval is taken on the case weights and strata.
    hits = np.array([0, 1, 0, 1, 0])  # E1 slight, E2 good, E2 slight, E3 good, U1 FN
    weights = [5, 5, 5, 4, 4]
    strata = ["Mast Cell Tumors"] * 3 + ["Rare Sarcomas", "random_slice:2026-Q1"]
    _, low, high = intervals.weighted_proportion(hits, weights, strata)
    assert (overall["good_lo"], overall["good_hi"]) == pytest.approx((low, high))
    assert overall["good_boot_lo"] <= overall["good"] <= overall["good_boot_hi"]


def test_confidence_bands():
    assert gold_eval.confidence_band(0.0) == "0.0-0.2"
    assert gold_eval.confidence_band(0.2) == "0.2-0.4"
    assert gold_eval.confidence_band(0.7999) == "0.6-0.8"
    assert gold_eval.confidence_band(1.0) == "0.8-1.0"
    predictions = pd.DataFrame({
        "case_id": ["A", "A", "B"], "diagnosis_index": ["2", "1", "1"],
        "predicted_term": ["x", "y", "Non-Cancer"], "confidence": ["0.95", "0.45", "0.00"],
    })
    bands = gold_eval.case_bands(predictions, ["A", "B", "C"])
    assert bands.to_dict() == {"A": "0.4-0.6", "B": "non_cancer", "C": "no_prediction"}


def test_result_3_bronze_vs_silver_on_the_same_cases(result):
    table = result["bronze_vs_silver"]
    vs_silver = _row(table, ruler="silver")
    # E1 slight 5, E2 slight 5 + completely_off 5, E3 FP 4, E5 FN 10 (U1, E4 true negatives): 29.
    assert vs_silver["codes"] == 5
    assert (vs_silver["good"], vs_silver["gs"]) == pytest.approx((0.0, 10 / 29))
    assert _row(table, ruler="gold")["good"] == pytest.approx(9 / 23)
    assert _row(table, ruler="silver - gold (good)")["good"] == pytest.approx(0.0 - 9 / 23)


def test_result_4_disagreement_cells(result):
    table = result["disagreements"]
    assert _row(table, stage="tier1_exact", band="0.8-1.0")[["disagreements", "silver_right"]].tolist() == [1, 1.0]
    assert _row(table, stage="tier3_llm", band="non_cancer")["bronze_right"] == 1.0
    assert _row(table, stage="tier2_fuzzy", band="0.4-0.6")["neither_right"] == 1.0
    assert _row(table, stage="no_signal", band="non_cancer")["disagreements"] == 0
    overall = _row(table, stage="all")  # disagreeing: E1 (5), E2 (5), E3 (4), E5 (10)
    assert (overall["cases"], overall["disagreements"]) == (5, 4)
    assert (overall["silver_right"], overall["bronze_right"], overall["neither_right"]) == pytest.approx(
        (5 / 24, 14 / 24, 5 / 24))


def test_misses_and_cause_split(result):
    misses = result["misses"]
    assert sorted(map(tuple, misses[misses["method"] == "silver"][["case_id", "gold_code"]].values)) == [
        ("E2", "1001/3"), ("E2", "2001/3"), ("E3", "2001/3")]
    assert sorted(map(tuple, misses[misses["method"] == "bronze"][["case_id", "gold_code"]].values)) == [
        ("E1", "1001/3"), ("E2", "2001/3"), ("U1", "1001/3")]
    assert set(misses["source_version"]) == {SILVER, GEN}
    cause = result["cause_split"].set_index("method")
    assert cause.loc["silver", ["misses", "reviewed", "method_error", "input_gap"]].tolist() == [3, 2, 1, 1]
    assert cause.loc["silver", "method_error_share"] == pytest.approx(5 / 9)  # E2 (5) yes, E3 (4) no
    assert cause.loc["bronze", ["misses", "reviewed", "method_error"]].tolist() == [3, 1, 1]


def test_non_blind_note(tmp_path, monkeypatch, result):
    assert result["notes"] == [gold_eval.NON_BLIND_NOTE]
    assert build(tmp_path / "blind", monkeypatch, review_mode="blind")["notes"] == []


def test_representativeness_in_the_run(result):
    rep = result["representativeness"]
    assert result["group_basis"] == f"silver {SILVER}"
    groups = rep["major_groups"]["table"]
    # Major groups come from silver codes (MCT 2, Rare Carcinomas 2); Rare Sarcomas has gold but no silver code.
    assert sorted(groups.index) == ["Mast Cell Tumors", "Rare Carcinomas"]
    assert groups.loc["Mast Cell Tumors", "gold_codes"] == 3 and groups.loc["Rare Carcinomas", "gold_codes"] == 0
    assert rep["random_slice"] == {"cases": 1, "pass": True}
    assert not rep["pass"]


def test_representativeness_criteria_flip():
    def gold_rows(n_codes, origin="eval_batch"):
        return pd.DataFrame({"case_id": [f"C{i}" for i in range(n_codes)], "code": "1001/3",
                             "group": "Big", "origin": origin})

    group_codes = pd.Series({"Big": 990, "Tiny": 9, "Rare": 1})  # Tiny is 0.9%, Big 99%, Rare 0.1%
    test_codes = pd.Series({"Big": 29})
    failing = gold_eval.representativeness(gold_rows(29), 0.051, group_codes, test_codes)
    assert not failing["ci_half_width"]["pass"] and not failing["major_groups"]["pass"]
    assert not failing["random_slice"]["pass"] and not failing["pass"]
    assert list(failing["major_groups"]["table"].index) == ["Big"]
    assert failing["major_groups"]["unreachable_from_test"] == ["Big"]

    rows = pd.concat([gold_rows(29), gold_rows(1, origin="random_slice").assign(case_id="U")])
    passing = gold_eval.representativeness(rows, 0.05, group_codes, pd.Series({"Big": 30}))
    assert passing["ci_half_width"]["pass"] and passing["major_groups"]["pass"] and passing["random_slice"]["pass"]
    assert passing["pass"] and passing["major_groups"]["unreachable_from_test"] == []


def test_no_gold_eval_fails_cleanly(tmp_path, monkeypatch):
    fx.point_evaluation_config_at(monkeypatch, tmp_path)
    fx.make_three_way_split_generation(SPLIT, train=["T1"], calibration=["C1"], test=["E1"])
    fx.make_gold_store_csv([_gold("T1", "review_queue", SCC)])  # gold-train only
    with pytest.raises(gold_eval.GoldEvalError, match="no gold-eval rows"):
        gold_eval.run(tmp_path / "unused.csv", SILVER, SPLIT)


def test_ledger_from_another_split_is_refused(result):
    ledger = pd.DataFrame([_ledger_row("E1", "b1", "Mast Cell Tumors", 10, 1, "app_non_blind")])
    gold_rows = pd.DataFrame([_gold("E1", "eval_batch", MCT_M, slice_rate="", upload_period="")])
    with pytest.raises(gold_eval.GoldEvalError, match="drawn on split"):
        gold_eval.case_weights(gold_rows, ledger, "another-split")
    assert math.isclose(gold_eval.case_weights(gold_rows, ledger, SPLIT).loc["E1", "weight"], 10.0)


def _evaluate_script():
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("ml_next_scripts_evaluate", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_cli_gold_prints_counts_and_writes_misses(tmp_path, monkeypatch, capsys):
    build(tmp_path, monkeypatch)
    misses_out = tmp_path / "misses.csv"
    monkeypatch.setattr("sys.argv", ["evaluate.py", "gold", "--predictions", str(tmp_path / "predictions.csv"),
                                     "--silver", SILVER, "--split", SPLIT, "--n-boot", "20",
                                     "--misses-out", str(misses_out)])
    assert _evaluate_script().main() == 0
    out = capsys.readouterr().out
    assert out.startswith(f"NOTE: {gold_eval.NON_BLIND_NOTE}")
    assert "invented" not in out
    assert "method-error share  55.6%" in out  # silver: E2 (weight 5) yes, E3 (weight 4) no
    assert len(pd.read_csv(misses_out)) == 6


def test_cli_fails_cleanly_without_gold_or_audit_rows(tmp_path, monkeypatch, capsys):
    fx.point_evaluation_config_at(monkeypatch, tmp_path)
    fx.make_three_way_split_generation(SPLIT, train=["T1"], calibration=["C1"], test=["E1"])
    script = _evaluate_script()
    monkeypatch.setattr("sys.argv", ["evaluate.py", "gold", "--predictions", "x.csv", "--silver", SILVER,
                                     "--split", SPLIT])
    assert script.main() == 1
    monkeypatch.setattr("sys.argv", ["evaluate.py", "audit-rates"])
    assert script.main() == 1
    err = capsys.readouterr().err
    assert "no gold-eval rows" in err and "no audit rows" in err
