"""generations/splits.py: legacy import, three-way creation, loading."""

from __future__ import annotations

import hashlib

import pytest

import config
from generations.manifest import ManifestError, read_manifest
from generations.splits import (
    CALIBRATION_FILE, LEGACY_SPLIT_ID, LEGACY_TEMPORAL_SPLIT_ID, TEST_FILE, TRAIN_FILE,
    create_three_way, import_legacy, in_sweep_half, load_split, split_dir,
)

from . import fixtures as fx


@pytest.fixture
def legacy_env(tmp_path, monkeypatch):
    fx.point_generations_config_at(monkeypatch, tmp_path)
    fx.make_legacy_split_files()
    return tmp_path


def _legacy_ids(path):
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def test_import_legacy_is_byte_identical(legacy_env):
    import_legacy()
    for split_id, train_txt, test_txt in [
        (LEGACY_SPLIT_ID, config.LEGACY_TRAIN_CASES_TXT, config.LEGACY_TEST_CASES_TXT),
        (LEGACY_TEMPORAL_SPLIT_ID, config.LEGACY_TRAIN_CASES_TEMPORAL_TXT, config.LEGACY_TEST_CASES_TEMPORAL_TXT),
    ]:
        directory = split_dir(split_id)
        assert (directory / TRAIN_FILE).read_bytes() == train_txt.read_bytes()
        assert (directory / TEST_FILE).read_bytes() == test_txt.read_bytes()
        assert not (directory / CALIBRATION_FILE).exists()
        split = load_split(split_id)
        assert split.train == _legacy_ids(train_txt) and split.test == _legacy_ids(test_txt)
        assert split.calibration == frozenset()
        assert "\r" not in "".join(split.train | split.test)


def test_import_legacy_manifest_records_provenance(legacy_env):
    random_manifest, temporal_manifest = import_legacy()
    assert random_manifest["split_id"] == LEGACY_SPLIT_ID and temporal_manifest["split_id"] == LEGACY_TEMPORAL_SPLIT_ID
    assert "hashlib.md5(case_id.encode())" in random_manifest["md5_half_rule"]
    notes = " ".join(random_manifest["notes"])
    assert "tail gate" in notes and "whole test set" in notes
    assert random_manifest["counts"] == {"train": 32, "calibration": 0, "test": 8}
    assert temporal_manifest["counts"] == {"train": 35, "calibration": 0, "test": 5}
    assert random_manifest["source_files"][TRAIN_FILE]["path"] == "output/splits/train_cases.txt"


def test_import_legacy_refuses_to_overwrite(legacy_env):
    import_legacy()
    with pytest.raises(FileExistsError):
        import_legacy()


def test_load_split_rejects_tampered_partition(legacy_env):
    import_legacy()
    with open(split_dir(LEGACY_SPLIT_ID) / TRAIN_FILE, "a", encoding="utf-8") as file:
        file.write("CASE-0040\r\n")  # a test case smuggled into train
    with pytest.raises(ManifestError):
        load_split(LEGACY_SPLIT_ID)


def test_load_split_rejects_unlisted_calibration_file(legacy_env):
    import_legacy()
    (split_dir(LEGACY_SPLIT_ID) / CALIBRATION_FILE).write_text("CASE-0001\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="not in its manifest"):
        load_split(LEGACY_SPLIT_ID)


def test_md5_rule_is_the_legacy_expression():
    for case_id in fx.SPLIT_CASE_IDS:
        assert in_sweep_half(case_id) == (int(hashlib.md5(case_id.encode()).hexdigest(), 16) % 2 == 0)


def test_three_way_split(legacy_env):
    import_legacy()
    parent = load_split(LEGACY_SPLIT_ID)
    manifest = create_three_way(LEGACY_SPLIT_ID, "three-way-a")
    split = load_split("three-way-a")

    # Train unchanged, byte-for-byte.
    assert split.train == parent.train
    assert (split_dir("three-way-a") / TRAIN_FILE).read_bytes() == (split_dir(LEGACY_SPLIT_ID) / TRAIN_FILE).read_bytes()
    # Calibration / test cover the parent's test exactly, disjointly, by the md5 rule.
    assert split.calibration | split.test == parent.test
    assert not split.calibration & split.test
    assert split.calibration and split.test  # the synthetic IDs land in both halves
    assert all(in_sweep_half(c) for c in split.calibration)
    assert not any(in_sweep_half(c) for c in split.test)

    assert manifest["parent"] == LEGACY_SPLIT_ID
    assert manifest["counts"] == {"train": len(parent.train), "calibration": len(split.calibration), "test": len(split.test)}
    assert "hashlib.md5" in manifest["md5_half_rule"]
    assert read_manifest(split_dir("three-way-a")) == manifest


def test_three_way_split_is_deterministic(legacy_env):
    import_legacy()
    create_three_way(LEGACY_SPLIT_ID, "three-way-a")
    create_three_way(LEGACY_SPLIT_ID, "three-way-b")
    for name in (TRAIN_FILE, CALIBRATION_FILE, TEST_FILE):
        assert (split_dir("three-way-a") / name).read_bytes() == (split_dir("three-way-b") / name).read_bytes()


def test_three_way_refuses_parent_with_calibration(legacy_env):
    import_legacy()
    create_three_way(LEGACY_SPLIT_ID, "three-way-a")
    with pytest.raises(ValueError):
        create_three_way("three-way-a", "three-way-b")
