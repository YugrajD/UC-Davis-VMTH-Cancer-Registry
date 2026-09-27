"""manual_audit/gold.py: case-level gold ingest, loaders, snapshot hash, guard integration."""

from __future__ import annotations

import pandas as pd
import pytest

import config
import io_utils
from generations.guards import check_all
from generations.splits import create_three_way, load_split
from manual_audit import gold

from . import fixtures as fx

TWO_WAY_SPLIT_ID = "two-way"


def _rows(records, columns=("case_id", "term", "origin")):
    return pd.DataFrame(records, columns=list(columns))


def _eval_ledger(path, case_ids):
    """A minimal eval-batch ledger listing ``case_ids`` — enough for ingest_gold's
    "eval_batch rows must be ledgered" check, which only reads the case_id column."""
    io_utils.write_csv(pd.DataFrame({"case_id": list(case_ids)}), path)
    return path


@pytest.fixture
def three_way(tmp_path, monkeypatch):
    fx.point_manual_audit_config_at(monkeypatch, tmp_path)
    fx.make_two_way_split_generation(TWO_WAY_SPLIT_ID, fx.SPLIT_CASE_IDS[:32], fx.SPLIT_CASE_IDS[32:])
    create_three_way(TWO_WAY_SPLIT_ID, config.DEFAULT_SPLIT_ID)
    return load_split(config.DEFAULT_SPLIT_ID)


def test_ingest_valid_terms_fill_code_and_group(tmp_path, labels_csv):
    out = tmp_path / "gold.csv"
    ledger = _eval_ledger(tmp_path / "ledger.csv", ["CASE-A"])
    result = gold.ingest_gold(
        _rows([("CASE-A", "Mast cell tumor, malignant", "eval_batch"), ("CASE-B", "NO_CANCER", "random_slice")]),
        reviewer="Dr. Test", labels_csv=labels_csv, out_csv=out, eval_batch_ledger_csv=ledger, slice_rate="0.05",
    )
    assert result == {"added_cases": 2, "replaced_cases": 0, "total_rows": 2, "total_cases": 2}
    store = gold.load_gold(out)
    row_a = store[store["case_id"] == "CASE-A"].iloc[0]
    assert row_a["code"] == "1001/3"
    assert row_a["group"] == "Mast Cell Tumors"
    row_b = store[store["case_id"] == "CASE-B"].iloc[0]
    assert row_b["code"] == "NO_CANCER" and row_b["term"] == "" and row_b["group"] == ""
    assert (store["reviewer"] == "Dr. Test").all()


def test_refuses_missing_required_columns(labels_csv, tmp_path):
    rows = pd.DataFrame({"case_id": ["CASE-A"], "term": ["Mast cell tumor, malignant"]})  # no origin
    with pytest.raises(gold.GoldIngestError, match="origin"):
        gold.ingest_gold(rows, reviewer="Dr. Test", labels_csv=labels_csv, out_csv=tmp_path / "gold.csv")


def test_refuses_invalid_origin_value(labels_csv, tmp_path):
    with pytest.raises(gold.GoldIngestError, match="origin"):
        gold.ingest_gold(
            _rows([("CASE-A", "Mast cell tumor, malignant", "manual")]),
            reviewer="Dr. Test", labels_csv=labels_csv, out_csv=tmp_path / "gold.csv",
        )


def test_refuses_unresolvable_term(labels_csv, tmp_path):
    ledger = _eval_ledger(tmp_path / "ledger.csv", ["CASE-A"])
    with pytest.raises(gold.GoldIngestError, match="not a term in the taxonomy"):
        gold.ingest_gold(
            _rows([("CASE-A", "Not A Real Term", "eval_batch")]),
            reviewer="Dr. Test", labels_csv=labels_csv, out_csv=tmp_path / "gold.csv", eval_batch_ledger_csv=ledger,
        )


def test_refuses_mixed_origin_within_a_case(labels_csv, tmp_path):
    with pytest.raises(gold.GoldIngestError, match="origin"):
        gold.ingest_gold(
            _rows([("CASE-A", "Mast cell tumor, malignant", "eval_batch"),
                   ("CASE-A", "Mast cell tumor, benign", "review_queue")]),
            reviewer="Dr. Test", labels_csv=labels_csv, out_csv=tmp_path / "gold.csv",
        )


def test_refuses_no_cancer_next_to_a_term_two_rows(labels_csv, tmp_path):
    """The cross-row case: one row NO_CANCER (via code), another a real term for the same case."""
    with pytest.raises(gold.GoldIngestError, match="NO_CANCER"):
        gold.ingest_gold(
            pd.DataFrame([
                {"case_id": "CASE-A", "code": "NO_CANCER", "term": "", "origin": "review_queue"},
                {"case_id": "CASE-A", "code": "", "term": "Mast cell tumor, malignant", "origin": "review_queue"},
            ]),
            reviewer="Dr. Test", labels_csv=labels_csv, out_csv=tmp_path / "gold.csv",
        )


def test_refuses_no_cancer_next_to_a_code_same_row(labels_csv, tmp_path):
    """The same-row case: NO_CANCER in one column, a real code/term in the other."""
    with pytest.raises(gold.GoldIngestError, match="NO_CANCER"):
        gold.ingest_gold(
            pd.DataFrame([{"case_id": "CASE-A", "code": "NO_CANCER", "term": "Mast cell tumor, malignant",
                          "origin": "review_queue"}]),
            reviewer="Dr. Test", labels_csv=labels_csv, out_csv=tmp_path / "gold.csv",
        )


def test_refuses_term_and_code_that_disagree(labels_csv, tmp_path):
    with pytest.raises(gold.GoldIngestError, match="disagrees"):
        gold.ingest_gold(
            pd.DataFrame([{"case_id": "CASE-A", "term": "Mast cell tumor, malignant", "code": "2001/3",
                          "origin": "review_queue"}]),
            reviewer="Dr. Test", labels_csv=labels_csv, out_csv=tmp_path / "gold.csv",
        )


def test_allows_term_and_code_that_agree(labels_csv, tmp_path):
    out = tmp_path / "gold.csv"
    gold.ingest_gold(
        pd.DataFrame([{"case_id": "CASE-A", "term": "Mast cell tumor, malignant", "code": "1001/3",
                      "origin": "review_queue"}]),
        reviewer="Dr. Test", labels_csv=labels_csv, out_csv=out,
    )
    assert gold.load_gold(out).iloc[0]["code"] == "1001/3"


def test_refuses_blank_case_id(labels_csv, tmp_path):
    with pytest.raises(gold.GoldIngestError, match="blank case_id"):
        gold.ingest_gold(
            _rows([("", "Mast cell tumor, malignant", "eval_batch")]),
            reviewer="Dr. Test", labels_csv=labels_csv, out_csv=tmp_path / "gold.csv",
        )


def test_refuses_duplicate_case_term_rows(labels_csv, tmp_path):
    with pytest.raises(gold.GoldIngestError, match="duplicate"):
        gold.ingest_gold(
            _rows([("CASE-A", "Mast cell tumor, malignant", "review_queue"),
                   ("CASE-A", "Mast cell tumor, malignant", "review_queue")]),
            reviewer="Dr. Test", labels_csv=labels_csv, out_csv=tmp_path / "gold.csv",
        )


def test_code_only_row_allowed_for_review_queue(labels_csv, tmp_path):
    out = tmp_path / "gold.csv"
    gold.ingest_gold(
        _rows([("CASE-A", "1001/3", "review_queue")], columns=("case_id", "code", "origin")),
        reviewer="Dr. Test", labels_csv=labels_csv, out_csv=out,
    )
    row = gold.load_gold(out).iloc[0]
    assert row["code"] == "1001/3" and row["group"] == "Mast Cell Tumors" and row["term"] == "Mast cell tumor, malignant"


def test_code_only_row_refused_for_eval_batch_origin(labels_csv, tmp_path):
    ledger = _eval_ledger(tmp_path / "ledger.csv", ["CASE-A"])
    with pytest.raises(gold.GoldIngestError, match="must resolve a term"):
        gold.ingest_gold(
            _rows([("CASE-A", "1001/3", "eval_batch")], columns=("case_id", "code", "origin")),
            reviewer="Dr. Test", labels_csv=labels_csv, out_csv=tmp_path / "gold.csv", eval_batch_ledger_csv=ledger,
        )


def test_ambiguous_code_only_refused_for_any_origin(tmp_path):
    # A code-only row whose code maps to more than one taxonomy term is
    # refused outright, even for review_queue: a blank term reads as
    # no-cancer in a labels table, so there is no safe origin for it.
    labels_path = fx.make_labels_csv(tmp_path / "labels.csv")
    # Plain utf-8 append (not utf-8-sig): a fresh utf-8-sig encoder would insert
    # a second BOM mid-file, corrupting the first appended field.
    with open(labels_path, "a", encoding="utf-8", newline="") as file:
        file.write("1001/3,Mast Cell Tumors,Mast cell tumor variant,Preferred,,\n")
    with pytest.raises(gold.GoldIngestError, match="more than one taxonomy term"):
        gold.ingest_gold(
            _rows([("CASE-A", "1001/3", "review_queue")], columns=("case_id", "code", "origin")),
            reviewer="Dr. Test", labels_csv=labels_path, out_csv=tmp_path / "gold.csv",
        )


def test_reingest_replaces_a_cases_rows(labels_csv, tmp_path):
    out = tmp_path / "gold.csv"
    gold.ingest_gold(_rows([("CASE-A", "Mast cell tumor, malignant", "review_queue"),
                           ("CASE-A", "Mast cell sarcoma", "review_queue")]),
                     reviewer="Dr. Test", labels_csv=labels_csv, out_csv=out)
    result = gold.ingest_gold(_rows([("CASE-A", "Fibrosarcoma, NOS", "review_queue")]),
                              reviewer="Dr. Test", labels_csv=labels_csv, out_csv=out)
    assert result == {"added_cases": 0, "replaced_cases": 1, "total_rows": 1, "total_cases": 1}
    store = gold.load_gold(out)
    assert list(store["code"]) == ["2001/3"]


def test_reingest_refuses_origin_change(labels_csv, tmp_path):
    out = tmp_path / "gold.csv"
    gold.ingest_gold(_rows([("CASE-A", "Mast cell tumor, malignant", "review_queue")]),
                     reviewer="Dr. Test", labels_csv=labels_csv, out_csv=out)
    with pytest.raises(gold.GoldIngestError, match="origin would change"):
        gold.ingest_gold(_rows([("CASE-A", "Fibrosarcoma, NOS", "random_slice")]),
                         reviewer="Dr. Test", labels_csv=labels_csv, out_csv=out, slice_rate="0.05")
    # Refused: nothing changed.
    assert list(gold.load_gold(out)["origin"]) == ["review_queue"]


def test_ingest_refuses_unknown_columns_in_existing_store(labels_csv, tmp_path):
    out = tmp_path / "gold.csv"
    io_utils.write_csv(pd.DataFrame([{"case_id": "X", "unexpected_column": "y"}]), out)
    with pytest.raises(gold.GoldIngestError, match="unexpected"):
        gold.ingest_gold(_rows([("CASE-A", "Mast cell tumor, malignant", "review_queue")]),
                         reviewer="Dr. Test", labels_csv=labels_csv, out_csv=out)


def test_ingest_refuses_unledgered_eval_batch_case(labels_csv, tmp_path):
    ledger = _eval_ledger(tmp_path / "ledger.csv", ["CASE-OTHER"])  # CASE-A never drawn
    with pytest.raises(gold.GoldIngestError, match="not in the eval-batch ledger"):
        gold.ingest_gold(
            _rows([("CASE-A", "Mast cell tumor, malignant", "eval_batch")]),
            reviewer="Dr. Test", labels_csv=labels_csv, out_csv=tmp_path / "gold.csv", eval_batch_ledger_csv=ledger,
        )


def test_ingest_refuses_eval_batch_case_with_no_ledger_at_all(labels_csv, tmp_path):
    missing_ledger = tmp_path / "no_such_ledger.csv"
    with pytest.raises(gold.GoldIngestError, match="not in the eval-batch ledger"):
        gold.ingest_gold(
            _rows([("CASE-A", "Mast cell tumor, malignant", "eval_batch")]),
            reviewer="Dr. Test", labels_csv=labels_csv, out_csv=tmp_path / "gold.csv",
            eval_batch_ledger_csv=missing_ledger,
        )


def test_gold_eval_and_gold_train_split_by_origin_and_partition(three_way, labels_csv, tmp_path):
    out = config.GOLD_STORE_CSV
    train_case = sorted(three_way.train)[0]
    test_cases = sorted(three_way.test)
    eval_batch_case, random_slice_test_case = test_cases[0], test_cases[1]
    ledger = _eval_ledger(tmp_path / "ledger.csv", [eval_batch_case])
    gold.ingest_gold(_rows([(eval_batch_case, "Mast cell tumor, malignant", "eval_batch")]),
                     reviewer="Dr. Test", labels_csv=labels_csv, out_csv=out, eval_batch_ledger_csv=ledger)
    # random_slice, on another test-partition case, also lands in gold_eval.
    gold.ingest_gold(_rows([(random_slice_test_case, "Mast cell tumor, benign", "random_slice")]),
                     reviewer="Dr. Test", labels_csv=labels_csv, out_csv=out, slice_rate="0.05")
    # random_slice for an upload case with no historical split membership at all.
    upload_case = "UPLOAD-0001"
    gold.ingest_gold(_rows([(upload_case, "Mast cell sarcoma", "random_slice")]),
                     reviewer="Dr. Test", labels_csv=labels_csv, out_csv=out, slice_rate="0.05")
    gold.ingest_gold(_rows([(train_case, "Fibrosarcoma, NOS", "review_queue")]),
                     reviewer="Dr. Test", labels_csv=labels_csv, out_csv=out)
    # Eval-side (calibration-partition) review-queue gold: exists in the store,
    # belongs to neither pool.
    eval_side_queue_case = sorted(three_way.calibration)[0]
    gold.ingest_gold(_rows([(eval_side_queue_case, "Fibrosarcoma, NOS", "review_queue")]),
                     reviewer="Dr. Test", labels_csv=labels_csv, out_csv=out)

    # No split_id passed: gold_eval() now defaults to config.DEFAULT_SPLIT_ID,
    # which is exactly the split "three_way" created above.
    ge = gold.gold_eval(out)
    assert set(ge["origin"]) == {"eval_batch", "random_slice"}
    assert eval_side_queue_case not in set(ge["case_id"])
    assert eval_batch_case in set(ge["case_id"])
    assert random_slice_test_case in set(ge["case_id"])
    assert upload_case in set(ge["case_id"])  # random_slice upload: outside every partition, still allowed

    gt = gold.gold_train(config.DEFAULT_SPLIT_ID, out)
    assert set(gt["case_id"]) == {train_case}
    assert eval_side_queue_case not in set(gt["case_id"])  # eval-side queue gold excluded from gold_train


def test_gold_snapshot_hash_changes_with_train_gold(three_way, labels_csv):
    out = config.GOLD_STORE_CSV
    train_case = sorted(three_way.train)[0]
    before = gold.gold_snapshot_hash(config.DEFAULT_SPLIT_ID, out)
    gold.ingest_gold(_rows([(train_case, "Mast cell tumor, malignant", "review_queue")]),
                     reviewer="Dr. Test", labels_csv=labels_csv, out_csv=out)
    after = gold.gold_snapshot_hash(config.DEFAULT_SPLIT_ID, out)
    assert before != after
    assert after == gold.gold_snapshot_hash(config.DEFAULT_SPLIT_ID, out)  # stable / deterministic


def test_refuses_random_slice_missing_slice_rate(labels_csv, tmp_path):
    """WP7 fix 10: evaluation.gold_eval weights a random_slice case by
    1/slice_rate, so ingest must refuse one with no slice_rate at all."""
    with pytest.raises(gold.GoldIngestError, match="slice_rate"):
        gold.ingest_gold(
            _rows([("CASE-A", "Mast cell tumor, malignant", "random_slice")]),
            reviewer="Dr. Test", labels_csv=labels_csv, out_csv=tmp_path / "gold.csv",
        )


@pytest.mark.parametrize("bad_rate", ["0", "-0.1", "1.5", "not-a-number"])
def test_refuses_random_slice_out_of_range_slice_rate(labels_csv, tmp_path, bad_rate):
    with pytest.raises(gold.GoldIngestError, match="slice_rate"):
        gold.ingest_gold(
            _rows([("CASE-A", "Mast cell tumor, malignant", "random_slice")]),
            reviewer="Dr. Test", labels_csv=labels_csv, out_csv=tmp_path / "gold.csv", slice_rate=bad_rate,
        )


def test_allows_random_slice_slice_rate_at_the_boundary(labels_csv, tmp_path):
    """slice_rate is a fraction in (0, 1] — 1 itself is valid (a full census)."""
    out = tmp_path / "gold.csv"
    gold.ingest_gold(
        _rows([("CASE-A", "Mast cell tumor, malignant", "random_slice")]),
        reviewer="Dr. Test", labels_csv=labels_csv, out_csv=out, slice_rate="1",
    )
    assert gold.load_gold(out).iloc[0]["slice_rate"] == "1"


def test_nan_term_and_code_read_as_empty_not_the_string_nan(labels_csv, tmp_path):
    """WP7 fix 9: rows built from a mixed-dtype source (e.g. pandas leaving a
    blank cell as float NaN) must not turn into the literal string "nan"."""
    rows = pd.DataFrame({
        "case_id": ["CASE-A"], "term": ["Mast cell tumor, malignant"],
        "code": [float("nan")], "origin": ["review_queue"],
    })
    out = tmp_path / "gold.csv"
    gold.ingest_gold(rows, reviewer="Dr. Test", labels_csv=labels_csv, out_csv=out)
    assert gold.load_gold(out).iloc[0]["code"] == "1001/3"  # resolved from the term, not "nan"


def test_nan_case_id_is_treated_as_blank(labels_csv, tmp_path):
    rows = pd.DataFrame({
        "case_id": [float("nan")], "term": ["Mast cell tumor, malignant"], "origin": ["review_queue"],
    })
    with pytest.raises(gold.GoldIngestError, match="blank case_id"):
        gold.ingest_gold(rows, reviewer="Dr. Test", labels_csv=labels_csv, out_csv=tmp_path / "gold.csv")


def test_duplicate_code_after_resolution_error_names_two_terms(labels_csv, tmp_path):
    """WP7 fix 9: the post-resolution duplicate-code message should read as
    two different terms sharing a code, not a bare '(case, code)' pair."""
    with pytest.raises(gold.GoldIngestError, match="two terms that resolve to the same code"):
        gold.ingest_gold(
            pd.DataFrame([
                {"case_id": "CASE-A", "term": "Mast cell tumor, malignant", "code": "", "origin": "review_queue"},
                {"case_id": "CASE-A", "term": "", "code": "1001/3", "origin": "review_queue"},
            ]),
            reviewer="Dr. Test", labels_csv=labels_csv, out_csv=tmp_path / "gold.csv",
        )


def test_guards_check_all_passes_on_produced_gold_store(three_way, labels_csv, tmp_path):
    out = config.GOLD_STORE_CSV
    train_case, test_case = sorted(three_way.train)[0], sorted(three_way.test)[0]
    ledger = _eval_ledger(tmp_path / "ledger.csv", [test_case])
    gold.ingest_gold(_rows([(test_case, "Mast cell tumor, malignant", "eval_batch")]),
                     reviewer="Dr. Test", labels_csv=labels_csv, out_csv=out, eval_batch_ledger_csv=ledger)
    gold.ingest_gold(_rows([(train_case, "Mast cell tumor, benign", "review_queue")]),
                     reviewer="Dr. Test", labels_csv=labels_csv, out_csv=out)
    passed = check_all(config.DEFAULT_SPLIT_ID)
    assert "gold origins" in passed
    assert "gold-eval in test" in passed
