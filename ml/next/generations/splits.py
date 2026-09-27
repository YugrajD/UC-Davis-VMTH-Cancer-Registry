"""Split generations: ``config.SPLITS_DIR/<split_id>/{train,calibration,test}_cases.txt`` + manifest.

A split generation is immutable once written: every function here refuses to
write into an existing ``<split_id>/`` directory, and ``load_split`` verifies the
manifest before reading, then reads only partition files the manifest lists.

- ``import_legacy`` copies the old flat split files byte-identical into
  ``legacy-80-20`` and ``legacy-temporal`` (no calibration file).
- ``create_three_way`` derives train / calibration / test from a two-way
  parent: train unchanged, parent test cut in two by the md5 half rule.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

import config
from generations.manifest import MANIFEST_NAME, read_manifest, sha256_file, verify_manifest, write_manifest

LEGACY_SPLIT_ID = "legacy-80-20"
LEGACY_TEMPORAL_SPLIT_ID = "legacy-temporal"

TRAIN_FILE = "train_cases.txt"
CALIBRATION_FILE = "calibration_cases.txt"
TEST_FILE = "test_cases.txt"

# The legacy md5 half rule, verbatim from ml/scripts/sweep_lp_thresholds.py
# (sweep_half). The "sweep" half fitted the legacy LP thresholds and becomes the
# calibration partition; the "eval" half scored the 62.1% baseline and becomes test.
MD5_HALF_RULE = "int(hashlib.md5(case_id.encode()).hexdigest(), 16) % 2 == 0 -> sweep half (calibration); else eval half (test)"


def in_sweep_half(case_id: str) -> bool:
    """True → md5 "sweep" half (calibration); False → "eval" half (test). Deterministic."""
    return int(hashlib.md5(case_id.encode()).hexdigest(), 16) % 2 == 0


@dataclass(frozen=True)
class Split:
    train: frozenset[str]
    calibration: frozenset[str]
    test: frozenset[str]


def split_dir(split_id: str) -> Path:
    return config.SPLITS_DIR / split_id


def _read_ids(path: Path) -> frozenset[str]:
    # strip() also drops the legacy files' CRLF line endings.
    return frozenset(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _new_split_dir(split_id: str) -> Path:
    directory = split_dir(split_id)
    if directory.exists():
        raise FileExistsError(f"split {split_id!r} already exists at {directory}; split generations are immutable")
    directory.mkdir(parents=True)
    return directory


def load_split(split_id: str) -> Split:
    """Verify the split's manifest, then read the partitions it lists.

    A partition the manifest does not list (no calibration file in a legacy
    split) loads as an empty set; a stray unlisted file fails verification.
    """
    directory = split_dir(split_id)
    verify_manifest(directory)
    listed = read_manifest(directory)["files"]

    def partition(name: str) -> frozenset[str]:
        return _read_ids(directory / name) if name in listed else frozenset()

    return Split(train=partition(TRAIN_FILE), calibration=partition(CALIBRATION_FILE), test=partition(TEST_FILE))


def _import_legacy_pair(split_id: str, train_txt: Path, test_txt: Path, method: dict) -> dict:
    directory = _new_split_dir(split_id)
    shutil.copyfile(train_txt, directory / TRAIN_FILE)
    shutil.copyfile(test_txt, directory / TEST_FILE)
    return write_manifest(directory, {
        "split_id": split_id,
        **method,
        "parent": None,
        "source_files": {
            TRAIN_FILE: {"path": train_txt.relative_to(config.ML_ROOT).as_posix(), "sha256": sha256_file(train_txt)},
            TEST_FILE: {"path": test_txt.relative_to(config.ML_ROOT).as_posix(), "sha256": sha256_file(test_txt)},
        },
        "counts": {"train": len(_read_ids(train_txt)), "calibration": 0, "test": len(_read_ids(test_txt))},
    })


def import_legacy() -> list[dict]:
    """Import the old flat split files as ``legacy-80-20`` and ``legacy-temporal``. Returns both manifests."""
    random_split = _import_legacy_pair(
        LEGACY_SPLIT_ID, config.LEGACY_TRAIN_CASES_TXT, config.LEGACY_TEST_CASES_TXT,
        {
            "method": "legacy random split (ml/training/data/create_split.py): cancer cases stratified by "
                      "their first matched_group in annotation.csv, non-cancer report cases split unstratified",
            "seed": 42,
            "stratification": "matched_group (cancer cases only)",
            "fractions": {"train": 0.8, "test": 0.2},
            "md5_half_rule": MD5_HALF_RULE,
            "notes": [
                "No calibration partition. Legacy LP thresholds were fitted on the md5 sweep half of test; "
                "the 62.1% G+S baseline is scored on the md5 eval half.",
                "The tail gate (K=2, gap 0.08) and the 0.80 case-presence gate threshold were fitted on the "
                "whole test set, so eval-half scores on this split are somewhat optimistic.",
            ],
        },
    )
    temporal_split = _import_legacy_pair(
        LEGACY_TEMPORAL_SPLIT_ID, config.LEGACY_TRAIN_CASES_TEMPORAL_TXT, config.LEGACY_TEST_CASES_TEMPORAL_TXT,
        {
            "method": "legacy temporal holdout (ml/training/data/create_split.py --temporal-cutoff-year): "
                      "report year (demographics DtOfRq) >= cutoff -> test, else train; no parseable date -> train",
            "seed": None,
            "stratification": None,
            "cutoff_year": None,  # not recorded by the legacy tooling
            "notes": ["Drift check only; no calibration partition."],
        },
    )
    return [random_split, temporal_split]


def create_three_way(parent_split_id: str, split_id: str) -> dict:
    """Derive train / calibration / test from a two-way parent split. Returns the manifest.

    train = the parent's train file, copied byte-identical; calibration = parent
    test cases in the md5 sweep half; test = the rest (the eval half).
    """
    parent = load_split(parent_split_id)
    if parent.calibration:
        raise ValueError(f"parent split {parent_split_id!r} already has a calibration partition")
    calibration = sorted(c for c in parent.test if in_sweep_half(c))
    test = sorted(c for c in parent.test if not in_sweep_half(c))

    directory = _new_split_dir(split_id)
    parent_dir = split_dir(parent_split_id)
    shutil.copyfile(parent_dir / TRAIN_FILE, directory / TRAIN_FILE)
    (directory / CALIBRATION_FILE).write_text("\n".join(calibration) + "\n", encoding="utf-8", newline="\n")
    (directory / TEST_FILE).write_text("\n".join(test) + "\n", encoding="utf-8", newline="\n")
    return write_manifest(directory, {
        "split_id": split_id,
        "method": "three-way from parent: train = parent train unchanged; parent test split by the md5 half rule",
        "seed": None,
        "stratification": None,
        "parent": parent_split_id,
        "parent_manifest_sha256": sha256_file(parent_dir / MANIFEST_NAME),
        "md5_half_rule": MD5_HALF_RULE,
        "counts": {"train": len(parent.train), "calibration": len(calibration), "test": len(test)},
    })
