"""Tests for coding.corrected: the report mapping's training labels."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import config
from coding.corrected import build_corrected_annotations, write_corrected_annotations
from coding.rule import UnknownDecisionError
from generations import guards

from . import fixtures as fx


@pytest.fixture
def scenario(monkeypatch, tmp_path: Path) -> dict:
    return fx.build_coding_scenario(monkeypatch, tmp_path)


def _rows_for(df: pd.DataFrame, case_id: str) -> pd.DataFrame:
    return df[df["case_id"] == case_id]


def test_gold_replaces_decisive_silver_rows(scenario):
    C = fx.CodingCaseIDs
    df = build_corrected_annotations(scenario["silver_id"], scenario["split_id"])
    rows = _rows_for(df, C.G_DIAG)
    assert len(rows) == 1
    assert rows.iloc[0]["matched_code"] == "9990/3"  # gold's, not silver's 1001/3
    assert rows.iloc[0]["label_source"] == "gold"


def test_gold_replaces_vague_silver_rows(scenario):
    C = fx.CodingCaseIDs
    df = build_corrected_annotations(scenario["silver_id"], scenario["split_id"])
    rows = _rows_for(df, C.G_VAGUE)
    assert len(rows) == 1
    assert rows.iloc[0]["matched_code"] == "9991/3"
    assert rows.iloc[0]["label_source"] == "gold"


def test_gold_no_cancer_replaces_vague_silver(scenario):
    C = fx.CodingCaseIDs
    df = build_corrected_annotations(scenario["silver_id"], scenario["split_id"])
    rows = _rows_for(df, C.G_NOCANCER)
    assert len(rows) == 1
    assert rows.iloc[0]["matched_term"] == ""
    # WP9 fix 7: the NO_CANCER sentinel must not leak into matched_code — a
    # gold non-cancer row reads the same as a silver one here (both "").
    assert rows.iloc[0]["matched_code"] == ""
    assert rows.iloc[0]["label_source"] == "gold"


def test_decisive_silver_without_gold_is_kept(scenario):
    C = fx.CodingCaseIDs
    df = build_corrected_annotations(scenario["silver_id"], scenario["split_id"])
    rows = _rows_for(df, C.DECISIVE_CANCER)
    assert len(rows) == 1
    assert rows.iloc[0]["matched_code"] == "1003/3"
    assert rows.iloc[0]["label_source"] == "silver"


def test_decisive_non_cancer_silver_has_empty_term(scenario):
    C = fx.CodingCaseIDs
    df = build_corrected_annotations(scenario["silver_id"], scenario["split_id"])
    rows = _rows_for(df, C.DECISIVE_NONCANCER)
    assert len(rows) == 1
    assert rows.iloc[0]["matched_term"] == ""
    assert rows.iloc[0]["label_source"] == "silver"


def test_vague_without_gold_is_excluded(scenario):
    C = fx.CodingCaseIDs
    df = build_corrected_annotations(scenario["silver_id"], scenario["split_id"])
    assert _rows_for(df, C.VAGUE_NOGOLD).empty


def test_no_calibration_or_test_cases(scenario):
    """TEST_VAGUE sits in the test partition and must never appear, vague or not."""
    C = fx.CodingCaseIDs
    df = build_corrected_annotations(scenario["silver_id"], scenario["split_id"])
    assert _rows_for(df, C.TEST_VAGUE).empty
    from generations.splits import load_split
    split = load_split(scenario["split_id"])
    assert not set(df["case_id"]) & (split.calibration | split.test)


def test_lineage_columns_are_stamped_on_every_row(scenario):
    df = build_corrected_annotations(scenario["silver_id"], scenario["split_id"])
    assert (df["silver_generation"] == scenario["silver_id"]).all()
    assert (df["gold_snapshot"] != "").all()


def test_write_corrected_annotations_passes_guards(scenario):
    out_path = write_corrected_annotations(scenario["silver_id"], scenario["split_id"])
    assert out_path.is_file()
    passed = guards.check_all(scenario["split_id"], labels_csv=out_path)
    assert "labels train-only" in passed


def test_write_leaves_prior_version_untouched_on_guard_failure(scenario, monkeypatch):
    """WP9 fix 5: guards run on the in-memory frame BEFORE any write, so a
    guard failure never touches out_path at all — the prior version (if any)
    is simply never opened, let alone rewritten."""
    # Write once successfully, capture its content.
    out_path = write_corrected_annotations(scenario["silver_id"], scenario["split_id"])
    original = out_path.read_bytes()
    original_mtime = out_path.stat().st_mtime_ns

    def _fail(*args, **kwargs):
        raise guards.GuardViolation("synthetic failure for the guard-first test")

    monkeypatch.setattr("coding.corrected.guards.check_labels_train_only", _fail)
    with pytest.raises(guards.GuardViolation):
        write_corrected_annotations(scenario["silver_id"], scenario["split_id"])
    assert out_path.read_bytes() == original
    assert out_path.stat().st_mtime_ns == original_mtime  # never even reopened for writing
    assert not out_path.with_name(out_path.name + ".tmp").exists()  # no leftover temp file either


def test_write_creates_no_file_on_guard_failure_with_no_prior_version(monkeypatch, tmp_path):
    fx.point_coding_config_at(monkeypatch, tmp_path)
    fx.make_three_way_split_generation("rollback-split", train=["CASE-OK"], calibration=[], test=[])
    fx.make_silver_generation("rollback-silver", [
        ("CASE-OK", 1, "invented diagnosis text", "Mast cell tumor, malignant", "Mast Cell Tumors", "1001/3",
         "mast cell tumor", "Exact", 1.0, "tier1_exact"),
    ])

    def _fail(*args, **kwargs):
        raise guards.GuardViolation("synthetic failure")

    monkeypatch.setattr("coding.corrected.guards.check_corrected_sources", _fail)
    assert not config.CORRECTED_ANNOTATIONS_CSV.exists()
    with pytest.raises(guards.GuardViolation):
        write_corrected_annotations("rollback-silver", "rollback-split")
    assert not config.CORRECTED_ANNOTATIONS_CSV.exists()


def test_unknown_decision_pair_raises(monkeypatch, tmp_path):
    fx.point_coding_config_at(monkeypatch, tmp_path)
    fx.make_three_way_split_generation("bad-pair-split", train=["CASE-BAD"], calibration=[], test=[])
    fx.make_silver_generation("bad-pair-silver", [
        ("CASE-BAD", 1, "invented diagnosis text", "", "", "", "", "Fuzzy", 0.4, "tier1_exact"),  # tier1_exact/Fuzzy is not a real pair
    ])
    with pytest.raises(UnknownDecisionError):
        build_corrected_annotations("bad-pair-silver", "bad-pair-split")
