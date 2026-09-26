"""Tests for coding.queue: the review queue."""

from __future__ import annotations

from pathlib import Path

import pytest

from coding.queue import LOW_CONF_BRONZE, NO_EVIDENCE, VAGUE_SILVER, build_review_queue
from coding.rule import UnknownDecisionError

from . import fixtures as fx


@pytest.fixture
def scenario(monkeypatch, tmp_path: Path) -> dict:
    return fx.build_coding_scenario(monkeypatch, tmp_path)


def test_gold_cases_are_skipped(scenario):
    C = fx.CodingCaseIDs
    df = build_review_queue(scenario["silver_id"], scenario["split_id"], scenario["predictions_csv"])
    queued = set(df["case_id"])
    # G_VAGUE is vague and G_BRONZE is low-confidence bronze, but both have gold.
    assert C.G_VAGUE not in queued
    assert C.G_BRONZE not in queued


def test_vague_silver_cases_are_queued(scenario):
    C = fx.CodingCaseIDs
    df = build_review_queue(scenario["silver_id"], scenario["split_id"], scenario["predictions_csv"])
    row = df[df["case_id"] == C.VAGUE_NOGOLD].iloc[0]
    assert row["reason"] == VAGUE_SILVER
    assert row["partition"] == "train"
    assert float(row["priority"]) == pytest.approx(0.9)


def test_vague_test_partition_case_is_queued_with_its_own_partition(scenario):
    """The strategy allows reviewing vague test cases; their gold stays on the
    eval side and is never gold-eval (this module just records the partition)."""
    C = fx.CodingCaseIDs
    df = build_review_queue(scenario["silver_id"], scenario["split_id"], scenario["predictions_csv"])
    row = df[df["case_id"] == C.TEST_VAGUE].iloc[0]
    assert row["reason"] == VAGUE_SILVER
    assert row["partition"] == "test"


def test_decisive_silver_cases_are_not_queued(scenario):
    C = fx.CodingCaseIDs
    df = build_review_queue(scenario["silver_id"], scenario["split_id"], scenario["predictions_csv"])
    queued = set(df["case_id"])
    assert C.DECISIVE_CANCER not in queued
    assert C.DECISIVE_NONCANCER not in queued


def test_low_confidence_bronze_cases_are_queued(scenario):
    C = fx.CodingCaseIDs
    df = build_review_queue(scenario["silver_id"], scenario["split_id"], scenario["predictions_csv"])
    queued = {row["case_id"]: row["reason"] for _, row in df.iterrows()}
    assert queued[C.BRONZE_LOW_CONF] == LOW_CONF_BRONZE
    assert queued[C.BRONZE_LOW_MARGIN] == LOW_CONF_BRONZE
    assert queued[C.BRONZE_METHOD_FLAG] == LOW_CONF_BRONZE


def test_high_confidence_bronze_case_is_not_queued(scenario):
    C = fx.CodingCaseIDs
    df = build_review_queue(scenario["silver_id"], scenario["split_id"], scenario["predictions_csv"])
    assert C.BRONZE_HIGH not in set(df["case_id"])


def test_queue_orders_test_partition_after_train_and_calibration(scenario):
    """WP9 fix 3: every test-partition item sorts after every train/calibration
    item (so a queue series works through the eval-side cases last), and each
    of those two blocks is separately priority-descending."""
    df = build_review_queue(scenario["silver_id"], scenario["split_id"], scenario["predictions_csv"])
    partitions = list(df["partition"])
    priorities = df["priority"].astype(float).tolist()

    non_test_priorities = [p for p, part in zip(priorities, partitions) if part != "test"]
    test_priorities = [p for p, part in zip(priorities, partitions) if part == "test"]
    assert non_test_priorities == sorted(non_test_priorities, reverse=True)
    assert test_priorities == sorted(test_priorities, reverse=True)
    # Every non-test row precedes every test row.
    if non_test_priorities and test_priorities:
        last_non_test = max(i for i, part in enumerate(partitions) if part != "test")
        first_test = min(i for i, part in enumerate(partitions) if part == "test")
        assert last_non_test < first_test

    # VAGUE_NOGOLD (0.90, train) must still sort ahead of TEST_VAGUE (0.40, test).
    C = fx.CodingCaseIDs
    order = list(df["case_id"])
    assert order.index(C.VAGUE_NOGOLD) < order.index(C.TEST_VAGUE)


def test_no_evidence_case_is_queued_with_lowest_priority(monkeypatch, tmp_path):
    """WP9 fix 2: a case with no gold, no diagnosis row and no bronze
    prediction at all (e.g. a split member whose report never made it into
    report.csv) must still be queued, not silently dropped."""
    fx.point_coding_config_at(monkeypatch, tmp_path)
    fx.make_three_way_split_generation(
        "no-evidence-split", train=["CASE-HAS-BRONZE", "CASE-NO-EVIDENCE"], calibration=[], test=[],
    )
    fx.make_silver_generation("no-evidence-silver", [])
    predictions_csv = fx.make_bronze_predictions_csv(tmp_path / "predictions.csv", [
        ("CASE-HAS-BRONZE", 1, "T", "G", "9001/3", "0.9000", "0.90", "0.90", "label_presence", "gen-ne"),
    ])
    df = build_review_queue("no-evidence-silver", "no-evidence-split", predictions_csv)
    row = df[df["case_id"] == "CASE-NO-EVIDENCE"]
    assert len(row) == 1
    assert row.iloc[0]["reason"] == NO_EVIDENCE
    assert row.iloc[0]["partition"] == "train"
    # Lowest priority: sorts after the high-confidence bronze case's queue-worthy peers.
    assert float(row.iloc[0]["priority"]) < 0


def test_lineage_columns(scenario):
    df = build_review_queue(scenario["silver_id"], scenario["split_id"], scenario["predictions_csv"])
    assert (df["silver_generation"] == scenario["silver_id"]).all()
    assert (df["bronze_generation"] == scenario["generation_id"]).all()


def test_unknown_decision_pair_raises(monkeypatch, tmp_path):
    fx.point_coding_config_at(monkeypatch, tmp_path)
    fx.make_three_way_split_generation("bad-pair-split", train=["CASE-BAD"], calibration=[], test=[])
    fx.make_silver_generation("bad-pair-silver", [
        ("CASE-BAD", 1, "invented diagnosis text", "", "", "", "", "Fuzzy", 0.4, "tier1_exact"),  # tier1_exact/Fuzzy is not a real pair
    ])
    predictions_csv = fx.make_bronze_predictions_csv(tmp_path / "predictions.csv", [
        ("CASE-BAD", 1, "T", "G", "9999/9", "0.9000", "0.90", "0.90", "label_presence", "gen-x"),
    ])
    with pytest.raises(UnknownDecisionError):
        build_review_queue("bad-pair-silver", "bad-pair-split", predictions_csv)
