"""Tests for coding.adopt: gold > silver > bronze, one adopted code set per case."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from coding.adopt import adopt_codes
from coding.queue import LOW_CONF_BRONZE, build_review_queue
from coding.rule import UnknownDecisionError
from manual_audit.gold import NO_CANCER

from . import fixtures as fx


@pytest.fixture
def scenario(monkeypatch, tmp_path: Path) -> dict:
    return fx.build_coding_scenario(monkeypatch, tmp_path)


def _rows_for(df: pd.DataFrame, case_id: str) -> pd.DataFrame:
    return df[df["case_id"] == case_id]


def test_gold_overrides_decisive_silver(scenario):
    C = fx.CodingCaseIDs
    df = adopt_codes(scenario["silver_id"], scenario["split_id"], scenario["predictions_csv"])
    rows = _rows_for(df, C.G_DIAG)
    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["code"] == "9990/3"  # gold's code, not silver's 1001/3
    assert row["code_source"] == "manual"
    assert row["review_status"] == "confirmed"


def test_gold_overrides_vague_silver(scenario):
    C = fx.CodingCaseIDs
    df = adopt_codes(scenario["silver_id"], scenario["split_id"], scenario["predictions_csv"])
    rows = _rows_for(df, C.G_VAGUE)
    assert len(rows) == 1
    assert rows.iloc[0]["code"] == "9991/3"
    assert rows.iloc[0]["code_source"] == "manual"


def test_gold_no_cancer_overrides_vague_silver(scenario):
    C = fx.CodingCaseIDs
    df = adopt_codes(scenario["silver_id"], scenario["split_id"], scenario["predictions_csv"])
    rows = _rows_for(df, C.G_NOCANCER)
    assert len(rows) == 1
    assert rows.iloc[0]["code"] == NO_CANCER
    assert rows.iloc[0]["term"] == ""
    assert rows.iloc[0]["code_source"] == "manual"


def test_gold_overrides_low_confidence_bronze(scenario):
    C = fx.CodingCaseIDs
    df = adopt_codes(scenario["silver_id"], scenario["split_id"], scenario["predictions_csv"])
    rows = _rows_for(df, C.G_BRONZE)
    assert len(rows) == 1
    assert rows.iloc[0]["code"] == "9992/3"
    assert rows.iloc[0]["code_source"] == "manual"
    assert rows.iloc[0]["review_status"] == "confirmed"


def test_bronze_never_overrides_decisive_silver(scenario):
    """DECISIVE_CANCER's bronze row disagrees (and is high-confidence) — silver wins."""
    C = fx.CodingCaseIDs
    df = adopt_codes(scenario["silver_id"], scenario["split_id"], scenario["predictions_csv"])
    rows = _rows_for(df, C.DECISIVE_CANCER)
    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["code"] == "1003/3"  # silver's code
    assert row["code"] != "9999/0"  # bronze's disagreeing code
    assert row["code_source"] == "diagnosis"
    assert row["review_status"] == "auto_accepted"
    assert row["source_confidence"] == "tier1_exact"


def test_decisive_non_cancer_silver_adopts_no_cancer_sentinel(scenario):
    C = fx.CodingCaseIDs
    df = adopt_codes(scenario["silver_id"], scenario["split_id"], scenario["predictions_csv"])
    rows = _rows_for(df, C.DECISIVE_NONCANCER)
    assert len(rows) == 1
    assert rows.iloc[0]["code"] == NO_CANCER
    assert rows.iloc[0]["code_source"] == "diagnosis"
    assert rows.iloc[0]["review_status"] == "auto_accepted"


def test_vague_silver_without_gold_gets_no_adopted_code(scenario):
    C = fx.CodingCaseIDs
    df = adopt_codes(scenario["silver_id"], scenario["split_id"], scenario["predictions_csv"])
    assert _rows_for(df, C.VAGUE_NOGOLD).empty
    assert _rows_for(df, C.TEST_VAGUE).empty


def test_bronze_high_confidence_adopts_all_its_codes_auto_accepted(scenario):
    C = fx.CodingCaseIDs
    df = adopt_codes(scenario["silver_id"], scenario["split_id"], scenario["predictions_csv"])
    rows = _rows_for(df, C.BRONZE_HIGH)
    assert set(rows["code"]) == {"B001", "B002"}
    assert (rows["code_source"] == "report").all()
    assert (rows["review_status"] == "auto_accepted").all()


def test_bronze_low_confidence_by_threshold_is_queued(scenario):
    C = fx.CodingCaseIDs
    df = adopt_codes(scenario["silver_id"], scenario["split_id"], scenario["predictions_csv"])
    rows = _rows_for(df, C.BRONZE_LOW_CONF)
    assert list(rows["code"]) == ["B003"]
    assert rows.iloc[0]["review_status"] == "queued"


def test_bronze_low_confidence_by_margin_is_queued(scenario):
    C = fx.CodingCaseIDs
    df = adopt_codes(scenario["silver_id"], scenario["split_id"], scenario["predictions_csv"])
    rows = _rows_for(df, C.BRONZE_LOW_MARGIN)
    assert set(rows["code"]) == {"B004", "B005"}
    assert (rows["review_status"] == "queued").all()


def test_bronze_method_flag_is_queued_no_cancer(scenario):
    C = fx.CodingCaseIDs
    df = adopt_codes(scenario["silver_id"], scenario["split_id"], scenario["predictions_csv"])
    rows = _rows_for(df, C.BRONZE_METHOD_FLAG)
    assert len(rows) == 1
    assert rows.iloc[0]["code"] == NO_CANCER
    assert rows.iloc[0]["review_status"] == "queued"
    assert rows.iloc[0]["code_source"] == "report"


def test_every_case_has_exactly_one_code_source(scenario):
    df = adopt_codes(scenario["silver_id"], scenario["split_id"], scenario["predictions_csv"])
    per_case_sources = df.groupby("case_id")["code_source"].nunique()
    assert (per_case_sources == 1).all()


def test_every_case_is_adopted_or_queued(scenario):
    """WP9 fix 2 invariant: every case in split.train ∪ split.calibration ∪
    split.test ∪ silver ∪ bronze must be either adopted (coding.adopt) or
    queued (coding.queue) — never neither."""
    from coding.queue import build_review_queue
    from diagnosis_mapping.silver import load_silver
    from generations.splits import load_split

    split = load_split(scenario["split_id"])
    silver = load_silver(scenario["silver_id"])
    bronze = pd.read_csv(scenario["predictions_csv"], dtype=str, keep_default_na=False)
    universe = (
        split.train | split.calibration | split.test
        | set(silver["case_id"]) | set(bronze["case_id"])
    )

    adopted = adopt_codes(scenario["silver_id"], scenario["split_id"], scenario["predictions_csv"])
    queued = build_review_queue(scenario["silver_id"], scenario["split_id"], scenario["predictions_csv"])
    covered = set(adopted["case_id"]) | set(queued["case_id"])

    assert universe - covered == set()


def test_gold_source_version_is_origin_and_batch(monkeypatch, tmp_path):
    """WP9 fix 6: gold rows are stamped from their own origin (+ batch_or_export_id
    when there is one), not gold_snapshot_hash — which only covers gold-train and
    would silently misattribute an eval_batch/random_slice row's provenance."""
    fx.point_coding_config_at(monkeypatch, tmp_path)
    fx.make_three_way_split_generation("prov-split", train=["CASE-EB", "CASE-RQ"], calibration=[], test=[])
    fx.make_silver_generation("prov-silver", [])
    fx.make_gold_store_csv([
        {"case_id": "CASE-EB", "code": "9001/3", "term": "T1", "group": "G1",
         "origin": "eval_batch", "batch_or_export_id": "eval-batch-1"},
        {"case_id": "CASE-RQ", "code": "9002/3", "term": "T2", "group": "G2", "origin": "review_queue"},
    ])
    predictions_csv = fx.make_bronze_predictions_csv(tmp_path / "predictions.csv", [])
    df = adopt_codes("prov-silver", "prov-split", predictions_csv)
    by_case = {row["case_id"]: row for _, row in df.iterrows()}
    assert by_case["CASE-EB"]["source_version"] == "eval_batch:eval-batch-1"
    assert by_case["CASE-RQ"]["source_version"] == "review_queue"  # no batch id -> origin alone


def test_bronze_adoption_sorts_diagnosis_index_numerically(monkeypatch, tmp_path):
    """WP9 fix 7: diagnosis_index is read as str; sorting it lexically would put
    rank 10 before rank 2. A duplicate code at both ranks must dedup to the
    numerically-first (rank 2) row, not the lexically-first (rank 10) one."""
    fx.point_coding_config_at(monkeypatch, tmp_path)
    fx.make_three_way_split_generation("sort-split", train=["CASE-SORT"], calibration=[], test=[])
    fx.make_silver_generation("sort-silver", [])
    predictions_csv = fx.make_bronze_predictions_csv(tmp_path / "predictions.csv", [
        ("CASE-SORT", 10, "Wrong Term", "Wrong Group", "DUPE", "0.9000", "0.90", "0.90",
         "label_presence", "gen-sort"),
        ("CASE-SORT", 2, "Correct Term", "Correct Group", "DUPE", "0.9000", "0.90", "0.90",
         "label_presence", "gen-sort"),
    ])
    df = adopt_codes("sort-silver", "sort-split", predictions_csv)
    rows = df[df["case_id"] == "CASE-SORT"]
    assert len(rows) == 1
    assert rows.iloc[0]["term"] == "Correct Term"


def test_unidentified_cancer_bronze_case_is_not_adopted_but_is_queued(monkeypatch, tmp_path):
    """WP9 fix 7: the case-presence gate passed (probably cancer) but no label
    resolved — must not adopt the NO_CANCER sentinel. It always scores 0.0, so
    the review queue picks it up as low-confidence bronze."""
    fx.point_coding_config_at(monkeypatch, tmp_path)
    fx.make_three_way_split_generation("uic-split", train=["CASE-UIC"], calibration=[], test=[])
    fx.make_silver_generation("uic-silver", [])
    predictions_csv = fx.make_bronze_predictions_csv(tmp_path / "predictions.csv", [
        ("CASE-UIC", 1, "Unidentified Group", "Unidentified Group", "", "0.9000", "0.00", "0.90",
         "unidentified_cancer", "gen-uic"),
    ])
    df = adopt_codes("uic-silver", "uic-split", predictions_csv)
    assert df[df["case_id"] == "CASE-UIC"].empty

    queue_df = build_review_queue("uic-silver", "uic-split", predictions_csv)
    row = queue_df[queue_df["case_id"] == "CASE-UIC"].iloc[0]
    assert row["reason"] == LOW_CONF_BRONZE


def test_bronze_low_confidence_on_a_non_rank1_row_is_queued(monkeypatch, tmp_path):
    """WP9 fix 4: the backend flags a row when ITS OWN confidence is below the
    floor, not only rank 1's. A high-confidence rank-1 row beside a
    below-floor rank-2 row must still queue the case."""
    fx.point_coding_config_at(monkeypatch, tmp_path)
    fx.make_three_way_split_generation("rank2-split", train=["CASE-RANK2"], calibration=[], test=[])
    fx.make_silver_generation("rank2-silver", [])
    predictions_csv = fx.make_bronze_predictions_csv(tmp_path / "predictions.csv", [
        ("CASE-RANK2", 1, "T1", "G1", "R001", "0.9000", "0.90", "0.90", "label_presence", "gen-rank2"),
        ("CASE-RANK2", 2, "T2", "G1", "R002", "0.9000", "0.10", "0.10", "label_presence", "gen-rank2"),
    ])
    df = adopt_codes("rank2-silver", "rank2-split", predictions_csv)
    rows = df[df["case_id"] == "CASE-RANK2"]
    assert set(rows["code"]) == {"R001", "R002"}
    assert (rows["review_status"] == "queued").all()


def test_unknown_decision_pair_raises(monkeypatch, tmp_path):
    """A silver row outside the vagueness table must raise, not silently adopt or queue."""
    fx.point_coding_config_at(monkeypatch, tmp_path)
    fx.make_three_way_split_generation("bad-pair-split", train=["CASE-BAD"], calibration=[], test=[])
    fx.make_silver_generation("bad-pair-silver", [
        ("CASE-BAD", 1, "invented diagnosis text", "", "", "", "", "Fuzzy", 0.4, "tier1_exact"),  # tier1_exact/Fuzzy is not a real pair
    ])
    predictions_csv = fx.make_bronze_predictions_csv(tmp_path / "predictions.csv", [])
    with pytest.raises(UnknownDecisionError):
        adopt_codes("bad-pair-silver", "bad-pair-split", predictions_csv)
