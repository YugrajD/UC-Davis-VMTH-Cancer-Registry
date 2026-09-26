"""generations/guards.py: every guard catches an injected leak and passes a clean case."""

from __future__ import annotations

import pandas as pd
import pytest

import config
import io_utils
from generations.guards import (
    GuardViolation, check_all, check_calibration_inputs, check_corrected_sources, check_covers_parent,
    check_disjoint, check_eval_queue_gold_not_trained, check_gold_eval_in_test, check_gold_origins,
    check_labels_train_only,
)
from generations.manifest import ManifestError, write_manifest
from generations.splits import LEGACY_SPLIT_ID, Split, create_three_way, import_legacy, load_split, split_dir

from . import fixtures as fx

SPLIT = Split(
    train=frozenset({"T1", "T2", "T3"}),
    calibration=frozenset({"C1", "C2"}),
    test=frozenset({"E1", "E2"}),
)


def _gold(rows):
    return pd.DataFrame(rows, columns=["case_id", "code", "origin"])


def _labels(case_ids):
    return pd.DataFrame({"case_id": case_ids, "matched_code": ["1001/3"] * len(case_ids)})


def _corrected(rows):
    return pd.DataFrame(rows, columns=["case_id", "matched_code", "label_source"])


CLEAN_GOLD = _gold([
    ("E1", "1001/3", "eval_batch"),
    ("E2", "NO_CANCER", "random_slice"),
    ("T1", "2001/3", "review_queue"),
    ("C1", "1001/3", "review_queue"),  # eval-side queue gold: allowed to exist, never to train
])


# --- disjoint / coverage ------------------------------------------------------

def test_disjoint_clean():
    check_disjoint(SPLIT)


@pytest.mark.parametrize("field, leaked", [("train", "C1"), ("train", "E1"), ("calibration", "E2")])
def test_disjoint_catches_overlap(field, leaked):
    leaky = Split(**{**SPLIT.__dict__, field: SPLIT.__dict__[field] | {leaked}})
    with pytest.raises(GuardViolation, match=f"1 case\\(s\\).*{leaked}"):
        check_disjoint(leaky)


def test_covers_parent_clean():
    parent = Split(train=SPLIT.train, calibration=frozenset(), test=SPLIT.calibration | SPLIT.test)
    check_covers_parent(SPLIT, parent)


def test_covers_parent_catches_changed_train_and_lost_case():
    parent = Split(train=SPLIT.train | {"T4"}, calibration=frozenset(), test=SPLIT.calibration | SPLIT.test)
    with pytest.raises(GuardViolation, match="T4"):
        check_covers_parent(SPLIT, parent)
    parent = Split(train=SPLIT.train, calibration=frozenset(), test=SPLIT.calibration | SPLIT.test | {"E3"})
    with pytest.raises(GuardViolation, match="E3"):
        check_covers_parent(SPLIT, parent)


# --- gold store ---------------------------------------------------------------

def test_gold_clean():
    check_gold_origins(CLEAN_GOLD)
    check_gold_eval_in_test(CLEAN_GOLD, SPLIT)


def test_gold_origins_catches_unknown_origin():
    gold = pd.concat([CLEAN_GOLD, _gold([("T2", "1001/3", "eval-batch"), ("T3", "1001/3", "")])])
    with pytest.raises(GuardViolation, match="2 case\\(s\\).*T2, T3"):
        check_gold_origins(gold)


def test_gold_eval_allows_upload_only_random_slice():
    gold = pd.concat([CLEAN_GOLD, _gold([("UPLOAD-1", "1001/3", "random_slice")])])  # in no partition
    check_gold_eval_in_test(gold, SPLIT)


@pytest.mark.parametrize("leaked, origin", [
    ("T2", "eval_batch"), ("C2", "eval_batch"), ("UPLOAD-1", "eval_batch"),
    ("T2", "random_slice"), ("C2", "random_slice"),
])
def test_gold_eval_catches_case_outside_test(leaked, origin):
    gold = pd.concat([CLEAN_GOLD, _gold([(leaked, "1001/3", origin)])])
    with pytest.raises(GuardViolation, match=f"1 case\\(s\\).*{leaked}"):
        check_gold_eval_in_test(gold, SPLIT)


# --- training labels ----------------------------------------------------------

def test_labels_clean():
    labels = _labels(["T1", "T2", "T3"])
    check_labels_train_only(labels, SPLIT)
    check_eval_queue_gold_not_trained(CLEAN_GOLD, labels, SPLIT)


@pytest.mark.parametrize("leaked", ["C2", "E1"])
def test_labels_train_only_catches_eval_side_case(leaked):
    with pytest.raises(GuardViolation, match=leaked):
        check_labels_train_only(_labels(["T1", leaked]), SPLIT)


def test_eval_queue_gold_catches_training_leak():
    # C1 has review-queue gold and sits in calibration; it must not reach training labels.
    with pytest.raises(GuardViolation, match="1 case\\(s\\).*C1"):
        check_eval_queue_gold_not_trained(CLEAN_GOLD, _labels(["T1", "C1"]), SPLIT)


def test_eval_queue_gold_allows_train_side_queue_gold():
    check_eval_queue_gold_not_trained(CLEAN_GOLD, _labels(["T1"]), SPLIT)  # T1 is train-side queue gold


# --- calibration inputs -------------------------------------------------------

def test_calibration_inputs_clean():
    check_calibration_inputs(["C1", "C2 "], SPLIT)


@pytest.mark.parametrize("leaked", ["T1", "E1"])
def test_calibration_inputs_catch_other_partition(leaked):
    with pytest.raises(GuardViolation, match=leaked):
        check_calibration_inputs(["C1", leaked], SPLIT)


# --- corrected annotations vs audit store ------------------------------------

def test_corrected_sources_clean():
    corrected = _corrected([("T1", "2001/3", "gold"), ("T2", "1001/3", "silver")])
    check_corrected_sources(corrected, CLEAN_GOLD)


def test_corrected_sources_catch_audit_label_source():
    corrected = _corrected([("T2", "1001/3", "silver"), ("T3", "1001/3", "audit")])
    with pytest.raises(GuardViolation, match="1 case\\(s\\).*T3"):
        check_corrected_sources(corrected, CLEAN_GOLD)


def test_corrected_sources_catch_gold_without_gold_row():
    # T3 is labelled gold but only has (audit) evidence — no gold-store row.
    corrected = _corrected([("T1", "2001/3", "gold"), ("T3", "1001/3", "gold")])
    with pytest.raises(GuardViolation, match="1 case\\(s\\).*T3"):
        check_corrected_sources(corrected, CLEAN_GOLD)


def test_violation_message_caps_examples():
    labels = _labels([f"E{i}" for i in range(10)])
    split = Split(train=frozenset(), calibration=frozenset(), test=frozenset(labels["case_id"]))
    with pytest.raises(GuardViolation, match=r"10 case\(s\).*\(\+5 more\)"):
        check_labels_train_only(labels, split)


# --- check_all over real on-disk stores (synthetic, redirected config) -------

@pytest.fixture
def three_way(tmp_path, monkeypatch):
    fx.point_generations_config_at(monkeypatch, tmp_path)
    fx.make_legacy_split_files()
    import_legacy()
    create_three_way(LEGACY_SPLIT_ID, "three-way")
    return load_split("three-way")


def _write(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    io_utils.write_csv(df, path)


def test_check_all_clean_without_stores(three_way):
    assert check_all("three-way") == ["disjoint", "covers parent"]


def test_check_all_clean_with_every_store(three_way, tmp_path):
    train, cal, test = sorted(three_way.train), sorted(three_way.calibration), sorted(three_way.test)
    _write(_gold([(test[0], "1001/3", "eval_batch"), (train[0], "1001/3", "review_queue"),
                  (cal[0], "1001/3", "review_queue")]), config.GOLD_STORE_CSV)
    _write(_corrected([(train[0], "1001/3", "gold"), (train[1], "", "silver")]), config.CORRECTED_ANNOTATIONS_CSV)
    _write(_labels(train[:3]), tmp_path / "labels.csv")
    passed = check_all("three-way", labels_csv=tmp_path / "labels.csv", calibration_case_ids=cal)
    assert len(passed) == 10


def test_check_all_reports_every_leak(three_way):
    train, cal, test = sorted(three_way.train), sorted(three_way.calibration), sorted(three_way.test)
    _write(_gold([(train[1], "1001/3", "eval_batch"), (cal[0], "1001/3", "review_queue")]), config.GOLD_STORE_CSV)
    _write(_corrected([(cal[0], "1001/3", "gold")]), config.CORRECTED_ANNOTATIONS_CSV)
    with pytest.raises(GuardViolation) as caught:
        check_all("three-way", calibration_case_ids=[test[0]])
    message = str(caught.value)
    for name in ["gold-eval in test", "corrected annotations train-only",
                 "corrected annotations: no eval-side queue gold", "calibration inputs"]:
        assert f"[{name}]" in message
    assert train[1] in message and cal[0] in message and test[0] in message


def test_check_all_legacy_has_no_calibration(three_way):
    with pytest.raises(GuardViolation, match="calibration inputs"):
        check_all(LEGACY_SPLIT_ID, calibration_case_ids=[sorted(three_way.calibration)[0]])


def test_check_all_rejects_tampered_split(three_way):
    with open(split_dir("three-way") / "test_cases.txt", "a", encoding="utf-8") as file:
        file.write(sorted(three_way.train)[0] + "\n")
    with pytest.raises(ManifestError, match="sha256"):
        check_all("three-way")


def test_check_all_rejects_store_missing_column(three_way):
    _write(pd.DataFrame({"case_id": [sorted(three_way.test)[0]], "code": ["1001/3"]}), config.GOLD_STORE_CSV)
    with pytest.raises(GuardViolation, match=r"lacks column\(s\) \['origin'\]"):
        check_all("three-way")


def test_check_all_rejects_zero_byte_store(three_way):
    config.CORRECTED_ANNOTATIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    config.CORRECTED_ANNOTATIONS_CSV.write_bytes(b"")
    with pytest.raises(GuardViolation, match="empty"):
        check_all("three-way")


def test_check_all_catches_overlap_in_self_consistent_split(three_way):
    # A split whose manifest matches its files, but whose calibration repeats a train case.
    leaked = sorted(three_way.train)[0]
    directory = split_dir("overlapping")
    directory.mkdir()
    (directory / "train_cases.txt").write_text("\n".join(sorted(three_way.train)) + "\n", encoding="utf-8")
    (directory / "calibration_cases.txt").write_text(
        "\n".join(sorted(three_way.calibration | {leaked})) + "\n", encoding="utf-8")
    (directory / "test_cases.txt").write_text("\n".join(sorted(three_way.test)) + "\n", encoding="utf-8")
    write_manifest(directory, {"split_id": "overlapping", "parent": None})
    with pytest.raises(GuardViolation, match=rf"\[disjoint\] 1 case\(s\) in both train and calibration: {leaked}"):
        check_all("overlapping")
