"""Tests for coding.combine: gold > silver > bronze, one combined prediction per case."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import config
from coding.combine import combine_predictions
from coding.queue import LOW_CONF_BRONZE, build_review_queue
from coding.rule import UnknownDecisionError
from manual_audit.gold import NO_CANCER

from . import fixtures as fx


@pytest.fixture
def scenario(monkeypatch, tmp_path: Path) -> dict:
    return fx.build_coding_scenario(monkeypatch, tmp_path)


def _rows_for(df: pd.DataFrame, case_id: str) -> pd.DataFrame:
    return df[df["case_id"] == case_id]


def _single_case_world(monkeypatch, tmp_path, name: str, prediction_rows: list[tuple], review_cut: float = 0.30,
                       generation_id: str | None = None) -> tuple[str, str, Path]:
    """A world with a current generation (id ``gen-<name>``, ``review_cut``) and the given bronze rows; returns
    (silver id, split id, predictions csv). The split holds ``CASE-<name>``."""
    fx.point_coding_config_at(monkeypatch, tmp_path)
    fx.make_review_generation(config.REPORT_MAPPING_CURRENT_DIR, generation_id or f"gen-{name}", review_cut)
    fx.make_three_way_split_generation(f"{name}-split", train=[f"CASE-{name}"], calibration=[], test=[])
    fx.make_silver_generation(f"{name}-silver", [])
    return f"{name}-silver", f"{name}-split", fx.make_bronze_predictions_csv(tmp_path / "predictions.csv", prediction_rows)


def test_gold_overrides_decisive_silver(scenario):
    C = fx.CodingCaseIDs
    df = combine_predictions(scenario["silver_id"], scenario["split_id"], scenario["predictions_csv"])
    rows = _rows_for(df, C.G_DIAG)
    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["code"] == "9990/3"  # gold's code, not silver's 1001/3
    assert row["code_source"] == "manual"
    assert row["review_status"] == "confirmed"


def test_gold_overrides_vague_silver(scenario):
    C = fx.CodingCaseIDs
    df = combine_predictions(scenario["silver_id"], scenario["split_id"], scenario["predictions_csv"])
    rows = _rows_for(df, C.G_VAGUE)
    assert len(rows) == 1
    assert rows.iloc[0]["code"] == "9991/3"
    assert rows.iloc[0]["code_source"] == "manual"


def test_gold_no_cancer_overrides_vague_silver(scenario):
    C = fx.CodingCaseIDs
    df = combine_predictions(scenario["silver_id"], scenario["split_id"], scenario["predictions_csv"])
    rows = _rows_for(df, C.G_NOCANCER)
    assert len(rows) == 1
    assert rows.iloc[0]["code"] == NO_CANCER
    assert rows.iloc[0]["term"] == ""
    assert rows.iloc[0]["code_source"] == "manual"


def test_gold_overrides_low_confidence_bronze(scenario):
    C = fx.CodingCaseIDs
    df = combine_predictions(scenario["silver_id"], scenario["split_id"], scenario["predictions_csv"])
    rows = _rows_for(df, C.G_BRONZE)
    assert len(rows) == 1
    assert rows.iloc[0]["code"] == "9992/3"
    assert rows.iloc[0]["code_source"] == "manual"
    assert rows.iloc[0]["review_status"] == "confirmed"


def test_bronze_never_overrides_decisive_silver(scenario):
    """DECISIVE_CANCER's bronze row disagrees (and is high-confidence) — silver wins."""
    C = fx.CodingCaseIDs
    df = combine_predictions(scenario["silver_id"], scenario["split_id"], scenario["predictions_csv"])
    rows = _rows_for(df, C.DECISIVE_CANCER)
    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["code"] == "1003/3"  # silver's code
    assert row["code"] != "9999/0"  # bronze's disagreeing code
    assert row["code_source"] == "diagnosis"
    assert row["review_status"] == "auto_accepted"
    assert row["source_confidence"] == "tier1_exact"


def test_decisive_non_cancer_silver_takes_no_cancer_sentinel(scenario):
    C = fx.CodingCaseIDs
    df = combine_predictions(scenario["silver_id"], scenario["split_id"], scenario["predictions_csv"])
    rows = _rows_for(df, C.DECISIVE_NONCANCER)
    assert len(rows) == 1
    assert rows.iloc[0]["code"] == NO_CANCER
    assert rows.iloc[0]["code_source"] == "diagnosis"
    assert rows.iloc[0]["review_status"] == "auto_accepted"


def test_vague_silver_without_gold_gets_no_combined_prediction(scenario):
    C = fx.CodingCaseIDs
    df = combine_predictions(scenario["silver_id"], scenario["split_id"], scenario["predictions_csv"])
    assert _rows_for(df, C.VAGUE_NOGOLD).empty
    assert _rows_for(df, C.TEST_VAGUE).empty


def test_bronze_high_confidence_takes_all_its_codes_auto_accepted(scenario):
    C = fx.CodingCaseIDs
    df = combine_predictions(scenario["silver_id"], scenario["split_id"], scenario["predictions_csv"])
    rows = _rows_for(df, C.BRONZE_HIGH)
    assert set(rows["code"]) == {"B001", "B002"}
    assert (rows["code_source"] == "report").all()
    assert (rows["source_version"] == scenario["generation_id"]).all()
    assert (rows["review_status"] == "auto_accepted").all()
    assert (rows["n_codes"] == 2).all()
    assert (_rows_for(df, C.DECISIVE_NONCANCER)["n_codes"] == 1).all()


def test_bronze_rank1_below_review_cut_is_queued(scenario):
    C = fx.CodingCaseIDs
    df = combine_predictions(scenario["silver_id"], scenario["split_id"], scenario["predictions_csv"])
    rows = _rows_for(df, C.BRONZE_LOW_CONF)
    assert list(rows["code"]) == ["B003"]
    assert rows.iloc[0]["review_status"] == "queued"


def test_bronze_tight_margin_is_not_queued(scenario):
    """There is no margin rule: rank 1 at 0.50 and rank 2 at 0.49 are both above review_cut."""
    C = fx.CodingCaseIDs
    df = combine_predictions(scenario["silver_id"], scenario["split_id"], scenario["predictions_csv"])
    rows = _rows_for(df, C.BRONZE_TIGHT_MARGIN)
    assert set(rows["code"]) == {"B004", "B005"}
    assert (rows["review_status"] == "auto_accepted").all()


def test_bronze_low_confidence_on_a_non_rank1_row_is_not_queued(scenario):
    """Only the rank-1 confidence is compared with review_cut."""
    C = fx.CodingCaseIDs
    df = combine_predictions(scenario["silver_id"], scenario["split_id"], scenario["predictions_csv"])
    rows = _rows_for(df, C.BRONZE_RANK2_LOW)
    assert set(rows["code"]) == {"B006", "B007"}
    assert (rows["review_status"] == "auto_accepted").all()


def test_bronze_gate_rejected_is_queued_no_cancer(scenario):
    C = fx.CodingCaseIDs
    df = combine_predictions(scenario["silver_id"], scenario["split_id"], scenario["predictions_csv"])
    rows = _rows_for(df, C.BRONZE_METHOD_FLAG)
    assert len(rows) == 1
    assert rows.iloc[0]["code"] == NO_CANCER
    assert rows.iloc[0]["review_status"] == "queued"
    assert rows.iloc[0]["code_source"] == "report"


def test_every_case_has_exactly_one_code_source(scenario):
    df = combine_predictions(scenario["silver_id"], scenario["split_id"], scenario["predictions_csv"])
    per_case_sources = df.groupby("case_id")["code_source"].nunique()
    assert (per_case_sources == 1).all()


def test_every_case_is_coded_or_queued(scenario):
    """WP9 fix 2 invariant: every case in split.train ∪ split.calibration ∪
    split.test ∪ silver ∪ bronze must be either coded (coding.combine) or
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

    combined = combine_predictions(scenario["silver_id"], scenario["split_id"], scenario["predictions_csv"])
    queued = build_review_queue(scenario["silver_id"], scenario["split_id"], scenario["predictions_csv"])
    covered = set(combined["case_id"]) | set(queued["case_id"])

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
    df = combine_predictions("prov-silver", "prov-split", predictions_csv)   # no bronze-only case: no generation needed
    by_case = {row["case_id"]: row for _, row in df.iterrows()}
    assert by_case["CASE-EB"]["source_version"] == "eval_batch:eval-batch-1"
    assert by_case["CASE-RQ"]["source_version"] == "review_queue"  # no batch id -> origin alone


def test_bronze_codes_sort_diagnosis_index_numerically(monkeypatch, tmp_path):
    """WP9 fix 7: diagnosis_index is read as str; sorting it lexically would put
    rank 10 before rank 2. A duplicate code at both ranks must dedup to the
    numerically-first (rank 2) row, not the lexically-first (rank 10) one."""
    silver_id, split_id, predictions_csv = _single_case_world(monkeypatch, tmp_path, "sort", [
        ("CASE-sort", 10, "Wrong Term", "Wrong Group", "DUPE", "0.9000", "0.90", "", "term_head", "gen-sort"),
        ("CASE-sort", 2, "Correct Term", "Correct Group", "DUPE", "0.9000", "0.90", "", "term_head", "gen-sort"),
    ])
    df = combine_predictions(silver_id, split_id, predictions_csv)
    rows = df[df["case_id"] == "CASE-sort"]
    assert len(rows) == 1
    assert rows.iloc[0]["term"] == "Correct Term"


def test_review_cut_boundary_and_a_rejected_row_anywhere(monkeypatch, tmp_path):
    """Queued iff a row is gate-rejected, there is no rank-1 row, or rank-1 confidence < review_cut (== is accepted)."""
    gen = "gen-cut"
    rows = [
        ("CASE-AT-CUT", 1, "T", "G", "A001", "0.9000", "0.30", "", "term_head", gen),
        ("CASE-BELOW", 1, "T", "G", "A002", "0.9000", "0.29", "", "term_head", gen),
        ("CASE-NO-RANK1", 2, "T", "G", "A003", "0.9000", "0.99", "", "term_head", gen),
        ("CASE-REJECT-RANK2", 1, "T", "G", "A004", "0.9000", "0.99", "", "term_head", gen),
        ("CASE-REJECT-RANK2", 2, "Non-Cancer", "Non-Cancer", "", "0.9000", "0.00", "", "rejected_by_case_presence", gen),
    ]
    silver_id, _, predictions_csv = _single_case_world(monkeypatch, tmp_path, "cut", rows)
    fx.make_three_way_split_generation("cut-all", train=sorted({r[0] for r in rows}), calibration=[], test=[])
    df = combine_predictions(silver_id, "cut-all", predictions_csv)
    status = df.groupby("case_id")["review_status"].agg(set).to_dict()
    assert status == {"CASE-AT-CUT": {"auto_accepted"}, "CASE-BELOW": {"queued"}, "CASE-NO-RANK1": {"queued"},
                      "CASE-REJECT-RANK2": {"queued"}}
    queue = build_review_queue(silver_id, "cut-all", predictions_csv)
    assert set(queue["case_id"]) == {"CASE-BELOW", "CASE-NO-RANK1", "CASE-REJECT-RANK2"}
    assert set(queue["reason"]) == {LOW_CONF_BRONZE}


def test_the_review_cut_comes_from_the_generation(monkeypatch, tmp_path):
    rows = [("CASE-rc", 1, "T", "G", "A001", "0.9000", "0.60", "", "term_head", "gen-rc")]
    silver_id, split_id, predictions_csv = _single_case_world(monkeypatch, tmp_path, "rc", rows, review_cut=0.50)
    assert combine_predictions(silver_id, split_id, predictions_csv)["review_status"].tolist() == ["auto_accepted"]
    assert build_review_queue(silver_id, split_id, predictions_csv).empty
    fx.make_review_generation(config.REPORT_MAPPING_CURRENT_DIR, "gen-rc", 0.70)
    assert combine_predictions(silver_id, split_id, predictions_csv)["review_status"].tolist() == ["queued"]
    assert build_review_queue(silver_id, split_id, predictions_csv)["reason"].tolist() == [LOW_CONF_BRONZE]


def test_a_named_generation_is_used_instead_of_current(monkeypatch, tmp_path):
    rows = [("CASE-ng", 1, "T", "G", "A001", "0.9000", "0.60", "", "term_head", "gen-other")]
    silver_id, split_id, predictions_csv = _single_case_world(monkeypatch, tmp_path, "ng", rows)
    other = fx.make_review_generation(tmp_path / "other-generation", "gen-other", 0.90)
    with pytest.raises(ValueError, match="gen-other"):
        combine_predictions(silver_id, split_id, predictions_csv)   # current is gen-ng
    assert combine_predictions(silver_id, split_id, predictions_csv, generation=str(other))["review_status"].tolist() == ["queued"]


def test_predictions_of_another_generation_are_refused(monkeypatch, tmp_path):
    rows = [("CASE-mm", 1, "T", "G", "A001", "0.9000", "0.90", "", "term_head", "gen-old")]
    silver_id, split_id, predictions_csv = _single_case_world(monkeypatch, tmp_path, "mm", rows)
    with pytest.raises(ValueError, match=r"gen-old.*gen-mm"):
        combine_predictions(silver_id, split_id, predictions_csv)
    with pytest.raises(ValueError, match=r"gen-old.*gen-mm"):
        build_review_queue(silver_id, split_id, predictions_csv)
    mixed = fx.make_bronze_predictions_csv(tmp_path / "mixed.csv", rows + [
        ("CASE-mm2", 1, "T", "G", "A002", "0.9000", "0.90", "", "term_head", "gen-mm")])
    with pytest.raises(ValueError, match="gen-mm"):
        build_review_queue(silver_id, split_id, mixed)


def test_coding_imports_no_torch():
    import subprocess
    import sys

    code = "import sys, coding.combine, coding.queue; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=config.ML_ROOT)
    assert out.stdout.strip() == "False"


def test_unidentified_cancer_and_the_old_constants_are_gone():
    import coding.combine
    import coding.queue

    for module in (coding.combine, coding.queue):
        assert not [name for name in vars(module) if "unidentified" in name.lower()]
        assert "unidentified" not in (module.__doc__ or "").lower()
    assert not hasattr(coding.combine, "BRONZE_LOW_CONFIDENCE_THRESHOLD")
    assert not hasattr(coding.combine, "BRONZE_LOW_MARGIN_THRESHOLD")


@pytest.mark.xfail(strict=True, reason="a failed Tier-3 request is recorded as tier3_llm/'No Match' (decisive non-cancer)")
def test_timed_out_tier3_call_is_not_coded_as_no_cancer(monkeypatch, tmp_path):
    """A Tier-3 request that times out never judged the diagnosis, so the case
    must be queued, not auto-accepted as NO_CANCER."""
    import requests

    from diagnosis_mapping import llm_client, silver
    from taxonomy.taxonomy import TaxonomyLabel

    def _timeout(*a, **k):
        raise requests.Timeout("read timed out")
    monkeypatch.setattr(llm_client, "chat", _timeout)
    labels = [TaxonomyLabel(code="8000/3", group="Round Cell Tumors", term="Mast cell tumor, malignant")]
    diagnoses = pd.DataFrame([{"case_id": "CASE-TIMEOUT", "diagnosis_number": 1,
                               "diagnosis": "round cell tumor, cannot specify subtype"}])
    out_df, _counters = silver.run_cascade(diagnoses, labels)

    fx.point_coding_config_at(monkeypatch, tmp_path)
    fx.make_three_way_split_generation("timeout-split", train=["CASE-TIMEOUT"], calibration=[], test=[])
    fx.make_silver_generation("timeout-silver", list(out_df.itertuples(index=False, name=None)))
    predictions_csv = fx.make_bronze_predictions_csv(tmp_path / "predictions.csv", [])
    df = combine_predictions("timeout-silver", "timeout-split", predictions_csv)
    assert df[df["case_id"] == "CASE-TIMEOUT"].empty


def test_unknown_decision_pair_raises(monkeypatch, tmp_path):
    """A silver row outside the vagueness table must raise, not silently code or queue."""
    fx.point_coding_config_at(monkeypatch, tmp_path)
    fx.make_three_way_split_generation("bad-pair-split", train=["CASE-BAD"], calibration=[], test=[])
    fx.make_silver_generation("bad-pair-silver", [
        ("CASE-BAD", 1, "invented diagnosis text", "", "", "", "", "Fuzzy", 0.4, "tier1_exact"),  # tier1_exact/Fuzzy is not a real pair
    ])
    predictions_csv = fx.make_bronze_predictions_csv(tmp_path / "predictions.csv", [])
    with pytest.raises(UnknownDecisionError):
        combine_predictions("bad-pair-silver", "bad-pair-split", predictions_csv)
