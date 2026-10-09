"""report_mapping/inference: ranked scores, the decision rule, predict_frame and run_predict on a tiny generation."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

import config
from evaluation import verdicts
from report_mapping import generation
from report_mapping.inference import decide, scoring
from report_mapping.inference.decide import NEVER, Thresholds
from report_mapping.inference.predict import load_generation, predict_frame, run_predict
from report_mapping.inference.scoring import RankedScores

from . import fixtures as fx

TERMS = ["t1", "t2", "t3", "t4"]
TERM_INFO = {"t1": ("G1", "c1"), "t2": ("G1", "c2"), "t3": ("G2", "c3"), "t4": ("G3", "c4")}
UNCOMMON = frozenset({"G3"})
PRODUCTION_COLUMNS = ["case_id", "diagnosis_index", "predicted_term", "predicted_group", "predicted_code",
                      "case_presence_prob", "confidence", "group_prob", "method", "generation_id"]


def _scores(rows: dict[str, tuple[list[int], list[float], float]]) -> RankedScores:
    ids = list(rows)
    return RankedScores(TERMS, ids, np.array([rows[c][0] for c in ids]), np.array([rows[c][1] for c in ids]),
                        np.array([rows[c][2] for c in ids]))


# --- decide ---------------------------------------------------------------------------------------------------

def _case(gate_prob, th, probs=(0.9, 0.6)):
    return decide.case_rows("c", gate_prob, ["t1", "t3"], list(probs), th, TERM_INFO, "gen-test")


def test_rejected_case_is_one_non_cancer_row_with_the_production_method_string():
    (row,) = _case(0.30, Thresholds(0.5, 0.1, 0.0))
    assert row == {"case_id": "c", "case_presence_prob": "0.3000", "group_prob": "", "generation_id": "gen-test",
                   "diagnosis_index": 1, "predicted_term": "Non-Cancer", "predicted_group": "Non-Cancer",
                   "predicted_code": "", "confidence": "0.00", "method": "rejected_by_case_presence"}


def test_top1_only_row_carries_group_code_and_two_decimal_confidence():
    (row,) = _case(0.95, Thresholds(0.5, NEVER, 0.0))
    assert (row["predicted_term"], row["predicted_group"], row["predicted_code"]) == ("t1", "G1", "c1")
    assert row["confidence"] == "0.90" and row["case_presence_prob"] == "0.9500" and row["method"] == "term_head"
    assert row["diagnosis_index"] == 1 and row["group_prob"] == ""


def test_second_term_needs_both_tau_and_rho():
    assert len(_case(0.9, Thresholds(0.5, 0.5, 0.5))) == 2          # p2 .6 >= .5, ratio .667 >= .5
    assert len(_case(0.9, Thresholds(0.5, 0.7, 0.0))) == 1          # p2 below tau
    assert len(_case(0.9, Thresholds(0.5, 0.5, 0.75))) == 1         # ratio .667 below rho
    rows = _case(0.9, Thresholds(0.5, 0.5, 0.5))
    assert [r["diagnosis_index"] for r in rows] == [1, 2] and rows[1]["predicted_term"] == "t3"
    assert rows[1]["confidence"] == "0.60" and rows[1]["predicted_group"] == "G2"


def test_gate_threshold_is_inclusive_like_production():
    assert _case(0.5, Thresholds(0.5, NEVER, 0.0))[0]["method"] == "term_head"


def test_decide_all_columns_order_and_methods():
    scores = _scores({"a": ([0, 2, 1, 3], [0.9, 0.6, 0.2, 0.1], 0.9), "n": ([1, 0, 2, 3], [0.8, 0.5, 0.2, 0.1], 0.1)})
    out = decide.decide_all(scores, Thresholds(0.5, 0.5, 0.5), TERM_INFO, "gen-x")
    assert list(out.columns) == PRODUCTION_COLUMNS == decide.COLUMNS
    assert out.groupby("case_id").size().to_dict() == {"a": 2, "n": 1}
    assert out.loc[out["case_id"] == "n", "method"].tolist() == ["rejected_by_case_presence"]
    assert set(out["method"]) <= {"term_head", "rejected_by_case_presence"}
    assert "retriever_top1" not in set(out["method"])
    assert "unidentified_cancer" not in set(out["method"]) and "Unidentified Group" not in set(out["predicted_term"])
    assert (out["generation_id"] == "gen-x").all() and (out["group_prob"] == "").all()


def test_rejected_cancer_case_scores_as_false_negative():
    scores = _scores({"a": ([0, 1, 2, 3], [0.9, 0.6, 0.2, 0.1], 0.1)})
    out = decide.decide_all(scores, Thresholds(0.5, NEVER, 0.0), TERM_INFO, "g")
    labels = pd.DataFrame({"case_id": ["a"], "matched_term": ["t1"], "matched_group": ["G1"]})
    assert verdicts.score(labels, out, UNCOMMON)["verdict"].tolist() == ["false_negative"]


def test_decide_all_of_nothing_is_an_empty_frame_with_the_columns():
    nothing = RankedScores(TERMS, [], np.zeros((0, 4), int), np.zeros((0, 4)), np.zeros(0))
    out = decide.decide_all(nothing, Thresholds(0.5, 0.5, 0.5), TERM_INFO, "g")
    assert list(out.columns) == decide.COLUMNS and out.empty


# --- scoring ----------------------------------------------------------------------------------------------------

def test_rank_keeps_the_top_49_by_stable_argsort_and_sigmoid_probabilities():
    terms = [f"x{i}" for i in range(60)]
    logits = np.linspace(-5.0, -4.0, 60, dtype=np.float32).reshape(1, 60)
    logits[0, 5], logits[0, 9], logits[0, 20], logits[0, 21] = 3.0, 2.0, 1.0, 1.0   # the last two tie
    ranked = scoring.rank(["a"], terms, logits, np.array([0.0], dtype=np.float32))   # gate logit 0 -> 0.5
    assert scoring.TOP_K == 49 and ranked.term_idx.shape == ranked.probs.shape == (1, 49)
    assert ranked.term_idx[0, :4].tolist() == [5, 9, 20, 21]                          # ties keep input order
    assert ranked.probs[0, :2] == pytest.approx(1 / (1 + np.exp(-np.array([3.0, 2.0]))))
    assert (np.diff(ranked.probs[0]) <= 0).all()
    assert ranked.ranked_terms(0, 2) == ["x5", "x9"] and ranked.gate_probs.tolist() == pytest.approx([0.5])


def test_scoring_defaults():
    assert scoring.DEFAULT_TOKEN_BUDGET == 32768


# --- predict_frame / run_predict on a tiny generation -------------------------------------------------------------

def _reports() -> pd.DataFrame:
    return pd.DataFrame({
        "case_id": ["A", "B", "C", "D"],
        "HISTOPATHOLOGICAL SUMMARY": ["mast cell tumor present", "", "no evidence of neoplasia", None],
        "ADDITIONAL INFORMATION": ["", "carcinoma malignant", "", ""],     # B has text only in a column the old rule ignored
        "COMMENT": ["", "", "", "  "],                                       # D has no text at all
    })


@pytest.fixture
def loaded(tmp_path, tiny_modernbert_dir):
    return load_generation(fx.build_tiny_generation(tmp_path / "gen", tiny_modernbert_dir), "cpu")


def test_load_generation_reads_everything_from_the_generation(loaded):
    assert loaded.terms == fx.TINY_TERMS and loaded.max_length == 64 and loaded.generation_id.startswith("gen-")
    assert loaded.thresholds == Thresholds(0.5, 0.1, 0.25)
    assert loaded.term_info["Fibrosarcoma, NOS"] == ("Rare Sarcomas", "2001/3")
    assert not loaded.model.training


def test_load_generation_refuses_an_uncalibrated_generation(tmp_path, tiny_modernbert_dir):
    pending = fx.build_tiny_generation(tmp_path / "p", tiny_modernbert_dir, calibrated=False)
    with pytest.raises(generation.GenerationError):
        load_generation(pending, "cpu")


def test_predict_frame_writes_rows_only_for_cases_with_text(loaded):
    loose = replace(loaded, thresholds=Thresholds(0.0, 0.0, 0.0))       # every case accepted, both terms
    out = predict_frame(_reports(), loose, "cpu")
    assert list(out.columns) == decide.COLUMNS
    assert out.groupby("case_id").size().to_dict() == {"A": 2, "B": 2, "C": 2}      # D: no text, no rows
    assert out["case_id"].tolist() == ["A", "A", "B", "B", "C", "C"] and set(out["method"]) == {"term_head"}
    assert (out["generation_id"] == loose.generation_id).all()


def test_predict_frame_rejects_below_the_gate_and_matches_the_models_own_numbers(loaded):
    shut = replace(loaded, thresholds=Thresholds(2.0, NEVER, 0.0))
    out = predict_frame(_reports(), shut, "cpu")
    assert out["case_id"].tolist() == ["A", "B", "C"] and set(out["method"]) == {"rejected_by_case_presence"}
    batch = loaded.tokenizer(["HISTOPATHOLOGICAL SUMMARY: mast cell tumor present"], return_tensors="pt")
    with torch.no_grad():
        _, gate = loaded.model(batch["input_ids"], batch["attention_mask"])
    assert float(out.loc[out["case_id"] == "A", "case_presence_prob"].iloc[0]) == pytest.approx(
        torch.sigmoid(gate).item(), abs=5e-5)


def test_predict_frame_handles_reports_with_no_text_and_missing_columns(loaded):
    assert predict_frame(pd.DataFrame({"case_id": ["A"], "COMMENT": [""]}), loaded, "cpu").empty
    only_comment = predict_frame(pd.DataFrame({"case_id": ["A"], "COMMENT": ["mast cell tumor"]}), loaded, "cpu")
    assert only_comment["case_id"].unique().tolist() == ["A"]


def test_a_repeated_case_id_gets_its_own_rows(loaded):
    reports = pd.DataFrame({"case_id": ["A", "A"], "COMMENT": ["mast cell tumor", "no evidence of neoplasia"]})
    out = predict_frame(reports, replace(loaded, thresholds=Thresholds(0.0, NEVER, 0.0)), "cpu")
    assert out["case_id"].tolist() == ["A", "A"] and out["confidence"].nunique() >= 1


def test_run_predict_writes_the_csv_named_for_the_generation_and_equals_predict_frame(tmp_path, monkeypatch,
                                                                                    tiny_modernbert_dir):
    generation_dir = fx.build_tiny_generation(tmp_path / "gen", tiny_modernbert_dir)
    reports = _reports()
    reports["case_id"] = ["  A ", "B", "C", "D"]            # report.csv ids go through clean_text
    (tmp_path / "data").mkdir()
    reports.to_csv(tmp_path / "data" / "report.csv", index=False, encoding="latin-1")
    monkeypatch.setattr(config, "REPORT_CSV", tmp_path / "data" / "report.csv")
    monkeypatch.setattr(config, "PREDICTIONS_DIR", tmp_path / "predictions")

    path = run_predict(generation_dir, "cpu")
    loaded = load_generation(generation_dir, "cpu")
    assert path == tmp_path / "predictions" / f"{loaded.generation_id}_predictions.csv"
    written = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert list(written.columns) == PRODUCTION_COLUMNS
    expected = predict_frame(_reports(), loaded, "cpu")          # in memory, the worker's path
    assert written.to_dict("records") == expected.astype(str).to_dict("records")
    assert set(written["case_id"]) == {"A", "B", "C"}

    out = run_predict(generation_dir, "cpu", out=tmp_path / "elsewhere" / "p.csv")
    assert out.read_bytes() == path.read_bytes()


def test_inference_imports_without_diagnosis_mapping_evaluation_guards_scipy_or_sklearn():
    # transformers imports sklearn/scipy when installed; block them (as in the serving image, which has neither)
    # so any import of them by report_mapping/inference fails the subprocess.
    code = (
        "import sys\n"
        "for name in ('scipy', 'sklearn', 'diagnosis_mapping', 'evaluation', 'generations.guards'):\n"
        "    sys.modules[name] = None\n"
        "import report_mapping.inference.predict\n"
        "loaded = [m for m in sys.modules if sys.modules[m] is not None and m.split('.')[0] in\n"
        "          ('scipy', 'sklearn', 'diagnosis_mapping', 'evaluation')]\n"
        "sys.exit(1 if loaded else 0)\n")
    result = subprocess.run([sys.executable, "-c", code], cwd=config.ML_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
