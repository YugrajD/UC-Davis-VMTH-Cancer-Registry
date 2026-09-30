"""report_mapping/training/labels.py: targets built from any labels table
(case_id, matched_term, matched_group, matched_code), guarded to the split's
train partition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from generations.guards import GuardViolation
from report_mapping.training import labels as labels_mod
from taxonomy.taxonomy import TaxonomyLabel

from . import fixtures as fx


def _labels_frame(rows: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["case_id", "matched_term", "matched_group", "matched_code"])


# ---------------------------------------------------------------------------
# lineage
# ---------------------------------------------------------------------------


def test_lineage_of_a_corrected_table_counts_gold_codes_only():
    df = _labels_frame([("C1", "T1", "G1", "8000/3"), ("C2", "", "", ""), ("C3", "T2", "G1", "8010/3")])
    df["label_source"] = ["gold", "gold", "silver"]  # C2 is a gold NO_CANCER row: no code
    df["silver_generation"] = "silver-7"
    df["gold_snapshot"] = "abc123"
    assert labels_mod.lineage(df, "corrected.csv") == {
        "silver_id": "silver-7", "gold_train_snapshot": "abc123", "gold_train_codes": 1}


def test_lineage_of_a_plain_table_is_unknown_silver_and_no_gold():
    df = _labels_frame([("C1", "T1", "G1", "8000/3")])
    assert labels_mod.lineage(df, "plain.csv") == {
        "silver_id": None, "gold_train_snapshot": None, "gold_train_codes": 0}


def test_lineage_refuses_a_table_mixing_silver_generations():
    df = _labels_frame([("C1", "T1", "G1", "8000/3"), ("C2", "T1", "G1", "8000/3")])
    df["silver_generation"] = ["silver-1", "silver-2"]
    with pytest.raises(ValueError, match="mixes 2 silver_generation"):
        labels_mod.lineage(df, "x")


# ---------------------------------------------------------------------------
# load_labels_table
# ---------------------------------------------------------------------------


def test_load_labels_table_from_csv_path(tmp_path):
    path = fx.make_training_annotation_csv(tmp_path / "annotation.csv")
    df = labels_mod.load_labels_table(str(path))
    assert len(df) == 30
    assert set(labels_mod.REQUIRED_COLUMNS).issubset(df.columns)


def test_load_labels_table_missing_columns_raises(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"case_id": ["A"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing column"):
        labels_mod.load_labels_table(str(path))


# ---------------------------------------------------------------------------
# select_train
# ---------------------------------------------------------------------------


def test_select_train_filters_to_train_partition(tmp_path, monkeypatch):
    fx.point_generations_config_at(monkeypatch, tmp_path)
    fx.make_two_way_split_generation("s1", train_ids=["A", "B"], test_ids=["C"])
    labels = _labels_frame([
        ("A", "Term A", "Group A", "1"), ("B", "", "", ""), ("C", "Term C", "Group C", "3"),
    ])
    filtered = labels_mod.select_train(labels, "s1")
    assert set(filtered["case_id"]) == {"A", "B"}


def test_select_train_refuses_when_split_partitions_overlap(tmp_path, monkeypatch):
    fx.point_generations_config_at(monkeypatch, tmp_path)
    # A corrupted split: "B" sits in both train and test. select_train's own
    # filtering (case_id in split.train) can't exclude it since it IS in
    # train -- the guard is the only thing that catches this.
    fx.make_two_way_split_generation("corrupt", train_ids=["A", "B"], test_ids=["B", "C"])
    labels = _labels_frame([("A", "", "", ""), ("B", "Term B", "Group B", "2")])
    with pytest.raises(GuardViolation):
        labels_mod.select_train(labels, "corrupt")


# ---------------------------------------------------------------------------
# training_case_ids
# ---------------------------------------------------------------------------


def test_training_case_ids_excludes_a_train_case_absent_from_labels(tmp_path, monkeypatch):
    from generations.splits import load_split

    fx.point_generations_config_at(monkeypatch, tmp_path)
    fx.make_two_way_split_generation("s2", train_ids=["A", "B", "C"], test_ids=["D"])
    split = load_split("s2")
    # "C" has an embedding and is in train, but no row at all in labels_train
    # (e.g. a vague-without-gold case dropped by coding.corrected) -- it must
    # not silently become a confirmed-non-cancer negative.
    labels_train = _labels_frame([("A", "Term A", "Group A", "1"), ("B", "", "", "")])
    cache_case_ids = ["A", "B", "C", "D"]  # "D" is test-side, also excluded

    result = labels_mod.training_case_ids(labels_train, cache_case_ids, split)
    assert result == ["A", "B"]


def test_training_case_ids_preserves_cache_order(tmp_path, monkeypatch):
    from generations.splits import load_split

    fx.point_generations_config_at(monkeypatch, tmp_path)
    fx.make_two_way_split_generation("s3", train_ids=["A", "B", "C"], test_ids=[])
    split = load_split("s3")
    labels_train = _labels_frame([("A", "", "", ""), ("B", "", "", ""), ("C", "", "", "")])

    result = labels_mod.training_case_ids(labels_train, ["C", "A", "B"], split)
    assert result == ["C", "A", "B"]


# ---------------------------------------------------------------------------
# gate_targets
# ---------------------------------------------------------------------------


def test_gate_targets():
    labels = _labels_frame([
        ("A", "Term A", "Group A", "1"), ("A", "", "", ""),
        ("B", "", "", ""), ("C", "Term C", "Group C", "3"),
    ])
    targets = labels_mod.gate_targets(labels, ["A", "B", "C", "D"])
    np.testing.assert_array_equal(targets, [1.0, 0.0, 1.0, 0.0])  # D: no row at all -> non-cancer


# ---------------------------------------------------------------------------
# group_targets
# ---------------------------------------------------------------------------


def test_group_targets_merges_below_threshold_and_forces_named_group():
    labels = _labels_frame([
        ("A", "T1", "Group X", "1"), ("B", "T2", "Group X", "1"), ("C", "T3", "Group Y", "2"),
        ("D", "T4", "Neoplasms, NOS", "3"), ("E", "", "", ""),
    ])
    case_ids = ["A", "B", "C", "D", "E"]
    gt = labels_mod.group_targets(labels, case_ids, uncommon_threshold=2, forced_uncommon=("Neoplasms, NOS",))

    assert gt.group_names == ["Group X", "Uncommon"]
    assert gt.uncommon_groups == ["Group Y", "Neoplasms, NOS"]
    expected = np.array([
        [1, 0], [1, 0], [0, 1], [0, 1], [0, 0],
    ], dtype=np.float32)
    np.testing.assert_array_equal(gt.targets, expected)
    assert gt.class_weights[0] == pytest.approx((5 - 2) / 2)   # Group X: 2 positives
    assert gt.class_weights[1] == pytest.approx((5 - 2) / 2)   # Uncommon: C + D = 2 positives


def test_group_targets_uncommon_threshold_zero_keeps_all_groups_separate():
    labels = _labels_frame([("A", "T1", "Group X", "1"), ("B", "T2", "Group Y", "2")])
    gt = labels_mod.group_targets(labels, ["A", "B"], uncommon_threshold=0, forced_uncommon=())
    assert gt.group_names == ["Group X", "Group Y"]
    assert gt.uncommon_groups == []


def test_group_targets_ignores_cases_outside_the_embedding_universe():
    labels = _labels_frame([("A", "T1", "Group X", "1"), ("Z", "T2", "Group X", "1")])
    gt = labels_mod.group_targets(labels, ["A"], uncommon_threshold=0, forced_uncommon=())
    assert gt.case_ids == ["A"]
    assert gt.targets.shape == (1, 1)
    assert gt.targets[0, 0] == 1.0


# ---------------------------------------------------------------------------
# label_presence_pairs
# ---------------------------------------------------------------------------


def test_label_presence_pairs_positive_and_negative_counts():
    taxonomy = [TaxonomyLabel(code=str(i), group="G", term=f"T{i}") for i in range(3)]
    labels = _labels_frame([("A", "T0", "G", "0"), ("B", "T1", "G", "1")])
    pairs = labels_mod.label_presence_pairs(labels, taxonomy, "G", negs_per_pos=2, seed=42)

    assert set(pairs.columns) == {"case_id", "label_term", "label_group", "target"}
    positives = pairs[pairs["target"] == 1]
    negatives = pairs[pairs["target"] == 0]
    assert len(positives) == 2
    assert len(negatives) == 4  # 2 positives * 2 negs_per_pos
    for _, row in negatives.iterrows():
        pos_term = positives.loc[positives["case_id"] == row["case_id"], "label_term"].iloc[0]
        assert row["label_term"] != pos_term


def test_label_presence_pairs_deterministic_given_seed():
    taxonomy = [TaxonomyLabel(code=str(i), group="G", term=f"T{i}") for i in range(6)]
    labels = _labels_frame([("A", "T0", "G", "0")])
    p1 = labels_mod.label_presence_pairs(labels, taxonomy, "G", negs_per_pos=3, seed=7)
    p2 = labels_mod.label_presence_pairs(labels, taxonomy, "G", negs_per_pos=3, seed=7)
    pd.testing.assert_frame_equal(p1.reset_index(drop=True), p2.reset_index(drop=True))


def test_label_presence_pairs_skips_group_with_fewer_than_two_taxonomy_labels():
    taxonomy = [TaxonomyLabel(code="1", group="G", term="T1")]
    labels = _labels_frame([("A", "T1", "G", "1")])
    pairs = labels_mod.label_presence_pairs(labels, taxonomy, "G", negs_per_pos=5, seed=42)
    assert pairs.empty


def test_label_presence_pairs_skips_group_with_no_train_annotations():
    taxonomy = [TaxonomyLabel(code="1", group="G", term="T1"), TaxonomyLabel(code="2", group="G", term="T2")]
    labels = _labels_frame([("A", "", "", "")])
    pairs = labels_mod.label_presence_pairs(labels, taxonomy, "G", negs_per_pos=5, seed=42)
    assert pairs.empty


def test_label_presence_pairs_uncommon_merges_groups():
    taxonomy = [TaxonomyLabel(code="1", group="G1", term="T1"), TaxonomyLabel(code="2", group="G2", term="T2")]
    labels = _labels_frame([("A", "T1", "G1", "1"), ("B", "T2", "G2", "2")])
    pairs = labels_mod.label_presence_pairs(
        labels, taxonomy, "Uncommon", uncommon_group_names=["G1", "G2"], negs_per_pos=1, seed=1,
    )
    assert set(pairs.loc[pairs["target"] == 1, "case_id"]) == {"A", "B"}


def test_label_presence_pairs_uncommon_requires_group_names():
    taxonomy = [TaxonomyLabel(code="1", group="G1", term="T1"), TaxonomyLabel(code="2", group="G2", term="T2")]
    labels = _labels_frame([("A", "T1", "G1", "1")])
    with pytest.raises(ValueError):
        labels_mod.label_presence_pairs(labels, taxonomy, "Uncommon", negs_per_pos=1, seed=1)


# ---------------------------------------------------------------------------
# contrastive_pairs
# ---------------------------------------------------------------------------


def test_contrastive_pairs_counts_by_min_report_chars():
    labels = _labels_frame([("A", "T1", "G1", "1"), ("B", "", "", "")])
    report_frame = pd.DataFrame({
        "case_id": ["A", "B"],
        "HISTOPATHOLOGICAL SUMMARY": ["a long enough summary text here", ""],
        "FINAL COMMENT": ["a sufficiently long final comment", ""],
        "COMMENT": ["", ""],
        "ANCILLARY TESTS": ["short", ""],  # 5 chars, below the default min_report_chars=10
    })
    pairs = labels_mod.contrastive_pairs(labels, report_frame)

    # Case B has no label -> contributes nothing. Case A: HIST and FINAL_COMMENT+COMMENT
    # clear 10 chars, ANCILLARY ("short") does not -> 2 pairs.
    assert len(pairs) == 2
    assert set(pairs["case_id"]) == {"A"}
    assert set(pairs["label_text"]) == {"T1 G1"}
    assert set(pairs.columns) == {"case_id", "section", "report_text", "label_text", "matched_term", "matched_group"}
