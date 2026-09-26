"""evaluation/silver_eval.py: scoring a partition, breakdowns, md5 half, history line."""

from __future__ import annotations

import csv

import pandas as pd
import pytest

import config
from evaluation import silver_eval, verdicts
from generations.splits import in_sweep_half

from . import fixtures as fx

GEN = "silver-eval-gen"
UNCOMMON = ["Rare Sarcomas", "Rare Carcinomas"]
LABEL_ROWS = [
    ("S1", "Mast cell tumor, malignant", "Mast Cell Tumors", "1001/3"),
    ("S2", "Fibrosarcoma, NOS", "Rare Sarcomas", "2001/3"),
    ("S3", "", "", ""),
    ("S4", "Squamous cell carcinoma, NOS", "Rare Carcinomas", "3001/3"),
    ("S5", "Mast cell tumor, malignant", "Mast Cell Tumors", "1001/3"),  # calibration: never scored on test
]
PREDICTION_ROWS = [
    ("S1", 1, "Mast cell tumor, malignant", "Mast Cell Tumors", "1001/3", "0.9", "0.90", "0.9", "label_presence", GEN),
    ("S2", 1, "Mast cell sarcoma", "Mast Cell Tumors", "1003/3", "0.9", "0.80", "0.9", "label_presence", GEN),
    ("S3", 1, "Fibrosarcoma, NOS", "Rare Sarcomas", "2001/3", "0.9", "0.70", "0.9", "label_presence", GEN),
    ("S4", 1, "Non-Cancer", "Non-Cancer", "", "0.1", "0.00", "0.0", "rejected_by_case_presence", GEN),
    ("S5", 1, "Fibrosarcoma, NOS", "Rare Sarcomas", "2001/3", "0.9", "0.70", "0.9", "label_presence", GEN),
]


@pytest.fixture
def env(tmp_path, monkeypatch):
    fx.point_evaluation_config_at(monkeypatch, tmp_path)
    fx.make_three_way_split_generation("se-split", train=["T1"], calibration=["S5"], test=["S1", "S2", "S3", "S4"])
    labels = tmp_path / "labels.csv"
    pd.DataFrame(LABEL_ROWS, columns=["case_id", "matched_term", "matched_group", "matched_code"]).to_csv(
        labels, index=False, encoding="utf-8")
    predictions = fx.make_bronze_predictions_csv(tmp_path / "predictions.csv", PREDICTION_ROWS)
    fx.make_minimal_generation(config.REPORT_MAPPING_CURRENT_DIR, GEN, UNCOMMON)
    return {"labels": str(labels), "predictions": predictions}


def test_hand_case_matches_verdicts_summarize_and_hand_counts(env):
    result = silver_eval.run(env["predictions"], env["labels"], "se-split", "test")
    summary = result["summary"]
    # S1 good; S2 completely_off + FN (Fibrosarcoma uncovered); S3 FP; S4 FN (Non-Cancer on a cancer case).
    assert (summary["total"], summary["good"], summary["slightly_off"], summary["completely_off"],
            summary["false_positive"], summary["false_negative"]) == (5, 1, 0, 1, 1, 2)

    labels = pd.DataFrame(LABEL_ROWS, columns=["case_id", "matched_term", "matched_group", "matched_code"])
    predictions = silver_eval.read_predictions(env["predictions"])
    test = ["S1", "S2", "S3", "S4"]
    expected = verdicts.summarize(verdicts.score(labels[labels["case_id"].isin(test)],
                                                 predictions[predictions["case_id"].isin(test)], UNCOMMON))
    assert summary == expected

    by_group = result["by_group"]
    assert by_group.loc["Rare Sarcomas", "n"] == 2 and by_group.loc["Rare Sarcomas", "false_negative"] == 1
    assert by_group.loc["Mast Cell Tumors", "good_share"] == 1.0
    assert "Fibrosarcoma, NOS" in result["by_term"].index


def test_history_line_appended(env):
    silver_eval.run(env["predictions"], env["labels"], "se-split", "test")
    silver_eval.run(env["predictions"], env["labels"], "se-split", "test", half="eval")
    with open(config.SILVER_EVAL_HISTORY_CSV, encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2
    assert rows[0]["generation_id"] == GEN and rows[0]["labels"] == env["labels"]
    assert rows[0]["split_id"] == "se-split" and rows[0]["partition"] == "test"
    assert rows[0]["total"] == "5" and rows[0]["good_plus_slight_pct"] == "20.0"
    assert rows[1]["partition"] == "test:eval-half"


def test_history_path_override(env, tmp_path):
    path = tmp_path / "elsewhere" / "history.csv"
    silver_eval.run(env["predictions"], env["labels"], "se-split", "test", history_csv=path)
    assert path.is_file() and not config.SILVER_EVAL_HISTORY_CSV.exists()


def test_md5_half_restricts_the_partition(env):
    test = {"S1", "S2", "S3", "S4"}
    assert silver_eval.partition_case_ids("se-split", "test", "eval") == {c for c in test if not in_sweep_half(c)}
    assert silver_eval.partition_case_ids("se-split", "test", "sweep") == {c for c in test if in_sweep_half(c)}


def test_refuses_foreign_generation_and_missing_partition(env, tmp_path):
    other = fx.make_minimal_generation(tmp_path / "other-gen", "other-gen", [])
    with pytest.raises(silver_eval.SilverEvalError, match="generation_id"):
        silver_eval.run(env["predictions"], env["labels"], "se-split", "test", generation=str(other))
    fx.make_two_way_split_generation("se-two-way", ["T1"], ["S1"])
    with pytest.raises(silver_eval.SilverEvalError, match="no 'calibration' partition"):
        silver_eval.run(env["predictions"], env["labels"], "se-two-way", "calibration")
