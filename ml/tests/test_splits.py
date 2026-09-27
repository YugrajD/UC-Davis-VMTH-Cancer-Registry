"""generations/splits.py: three-way creation, loading, the md5 half rule."""

from __future__ import annotations

import hashlib

import pytest

from generations.manifest import ManifestError, read_manifest
from generations.splits import (
    CALIBRATION_FILE, TEST_FILE, TRAIN_FILE,
    create_three_way, in_sweep_half, load_split, split_dir,
)

from . import fixtures as fx

TWO_WAY_SPLIT_ID = "two-way"


@pytest.fixture
def two_way_env(tmp_path, monkeypatch):
    fx.point_generations_config_at(monkeypatch, tmp_path)
    fx.make_two_way_split_generation(TWO_WAY_SPLIT_ID, sorted(fx.SPLIT_CASE_IDS[:32]), sorted(fx.SPLIT_CASE_IDS[32:]))
    return tmp_path


def test_load_split_rejects_tampered_partition(two_way_env):
    with open(split_dir(TWO_WAY_SPLIT_ID) / TRAIN_FILE, "a", encoding="utf-8") as file:
        file.write("CASE-0040\n")  # a test case smuggled into train
    with pytest.raises(ManifestError):
        load_split(TWO_WAY_SPLIT_ID)


def test_load_split_rejects_unlisted_calibration_file(two_way_env):
    (split_dir(TWO_WAY_SPLIT_ID) / CALIBRATION_FILE).write_text("CASE-0001\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="not in its manifest"):
        load_split(TWO_WAY_SPLIT_ID)


def test_md5_rule_is_the_legacy_expression():
    for case_id in fx.SPLIT_CASE_IDS:
        assert in_sweep_half(case_id) == (int(hashlib.md5(case_id.encode()).hexdigest(), 16) % 2 == 0)


def test_three_way_split(two_way_env):
    parent = load_split(TWO_WAY_SPLIT_ID)
    manifest = create_three_way(TWO_WAY_SPLIT_ID, "three-way-a")
    split = load_split("three-way-a")

    # Train unchanged, byte-for-byte.
    assert split.train == parent.train
    assert (split_dir("three-way-a") / TRAIN_FILE).read_bytes() == (split_dir(TWO_WAY_SPLIT_ID) / TRAIN_FILE).read_bytes()
    # Calibration / test cover the parent's test exactly, disjointly, by the md5 rule.
    assert split.calibration | split.test == parent.test
    assert not split.calibration & split.test
    assert split.calibration and split.test  # the synthetic IDs land in both halves
    assert all(in_sweep_half(c) for c in split.calibration)
    assert not any(in_sweep_half(c) for c in split.test)

    assert manifest["parent"] == TWO_WAY_SPLIT_ID
    assert manifest["counts"] == {"train": len(parent.train), "calibration": len(split.calibration), "test": len(split.test)}
    assert "hashlib.md5" in manifest["md5_half_rule"]
    assert read_manifest(split_dir("three-way-a")) == manifest


def test_three_way_split_is_deterministic(two_way_env):
    create_three_way(TWO_WAY_SPLIT_ID, "three-way-a")
    create_three_way(TWO_WAY_SPLIT_ID, "three-way-b")
    for name in (TRAIN_FILE, CALIBRATION_FILE, TEST_FILE):
        assert (split_dir("three-way-a") / name).read_bytes() == (split_dir("three-way-b") / name).read_bytes()


def test_three_way_refuses_parent_with_calibration(two_way_env):
    create_three_way(TWO_WAY_SPLIT_ID, "three-way-a")
    with pytest.raises(ValueError):
        create_three_way("three-way-a", "three-way-b")
