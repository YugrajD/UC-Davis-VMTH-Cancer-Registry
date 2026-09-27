"""Leakage guards: pure checks over case-id sets / dataframes plus a ``Split``.

Each guard raises ``GuardViolation`` naming the count and a few example case_ids
(IDs only — never text). ``check_all`` loads whichever stores exist via config
and runs every applicable guard; ``train``, ``calibrate``, ``evaluate gold`` and
``promote`` call it and refuse on any violation.

Every store is read as utf-8 with ``dtype=str, keep_default_na=False``; a
zero-byte store or one missing a required column raises ``GuardViolation``.
case_ids are compared after ``str.strip()``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

import config
import io_utils
from generations.manifest import read_manifest
from generations.splits import Split, load_split, split_dir

GOLD_ORIGINS = {"eval_batch", "review_queue", "random_slice"}
GOLD_EVAL_ORIGINS = {"eval_batch", "random_slice"}  # public: manual_audit imports it
CORRECTED_LABEL_SOURCES = {"gold", "silver"}

_EXAMPLES = 5


class GuardViolation(Exception):
    """A split / leakage guard failed."""


def _fail_if_any(offending: Iterable[str], what: str) -> None:
    offending = sorted(set(offending))
    if offending:
        examples = ", ".join(offending[:_EXAMPLES])
        more = f", ... (+{len(offending) - _EXAMPLES} more)" if len(offending) > _EXAMPLES else ""
        raise GuardViolation(f"{len(offending)} case(s) {what}: {examples}{more}")


def _case_ids(df: pd.DataFrame) -> set[str]:
    return set(df["case_id"].astype(str).str.strip())


def check_disjoint(split: Split) -> None:
    """train, calibration and test are pairwise disjoint."""
    _fail_if_any(split.train & split.calibration, "in both train and calibration")
    _fail_if_any(split.train & split.test, "in both train and test")
    _fail_if_any(split.calibration & split.test, "in both calibration and test")


def check_covers_parent(split: Split, parent: Split) -> None:
    """A derived split keeps the parent's train and partitions the parent's eval side exactly."""
    _fail_if_any(split.train ^ parent.train, "differ between train and the parent's train")
    _fail_if_any((split.calibration | split.test) ^ (parent.calibration | parent.test),
                 "differ between calibration ∪ test and the parent's calibration ∪ test")


def check_gold_origins(gold: pd.DataFrame) -> None:
    """Every gold-store row has a known origin. Expects columns ``case_id, origin``.

    An unknown origin (typo, blank) can't be classed as gold-eval or gold-train,
    so it would silently slip past the guards below.
    """
    _fail_if_any(gold.loc[~gold["origin"].isin(GOLD_ORIGINS), "case_id"].str.strip(),
                 f"have a gold origin outside {sorted(GOLD_ORIGINS)}")


def check_gold_eval_in_test(gold: pd.DataFrame, split: Split) -> None:
    """Gold-eval never sits in train or calibration. Expects columns ``case_id, origin``.

    ``eval_batch`` cases are sampled from test, so they must be ⊆ test.
    ``random_slice`` cases come from uploads and may be in no partition at all;
    they are only a violation when they are in train or calibration.
    """
    eval_batch = _case_ids(gold[gold["origin"] == "eval_batch"])
    _fail_if_any(eval_batch - split.test, "are eval_batch gold but not in the test partition")
    random_slice = _case_ids(gold[gold["origin"] == "random_slice"])
    _fail_if_any(random_slice & (split.train | split.calibration),
                 "are random_slice gold but in train or calibration")


def check_eval_queue_gold_not_trained(gold: pd.DataFrame, labels: pd.DataFrame, split: Split) -> None:
    """Review-queue gold for an eval-side case (calibration ∪ test) never enters training labels.

    Expects ``gold`` columns ``case_id, origin`` and ``labels`` column ``case_id``.
    """
    eval_side_queue = _case_ids(gold[gold["origin"] == "review_queue"]) & (split.calibration | split.test)
    _fail_if_any(eval_side_queue & _case_ids(labels),
                 "have eval-side review-queue gold and appear in training labels")


def check_labels_train_only(labels: pd.DataFrame, split: Split) -> None:
    """Training labels ∩ (calibration ∪ test) = ∅. Expects column ``case_id``.

    ``labels`` is the table a trainer actually fits on, i.e. after any
    partition filtering (corrected annotations are train-only by construction).
    """
    _fail_if_any(_case_ids(labels) & (split.calibration | split.test),
                 "are in training labels but on the eval side (calibration ∪ test)")


def check_calibration_inputs(calibration_case_ids: Iterable[str], split: Split) -> None:
    """Every case a calibration step reads is in the calibration partition."""
    _fail_if_any({c.strip() for c in calibration_case_ids} - split.calibration,
                 "are calibration inputs but not in the calibration partition")


def check_corrected_sources(corrected: pd.DataFrame, gold: pd.DataFrame) -> None:
    """The audit store never feeds corrected annotations.

    Expects ``corrected`` columns ``case_id, label_source`` and ``gold`` column
    ``case_id``. Fails on any ``label_source`` other than gold/silver, and on any
    case labelled ``gold`` without a gold-store row: its only possible evidence
    is then row-level audit judgements, which never count as gold.
    """
    _fail_if_any(corrected.loc[~corrected["label_source"].isin(CORRECTED_LABEL_SOURCES), "case_id"].str.strip(),
                 f"have a corrected-annotation label_source outside {sorted(CORRECTED_LABEL_SOURCES)}")
    _fail_if_any(_case_ids(corrected[corrected["label_source"] == "gold"]) - _case_ids(gold),
                 "are labelled gold in corrected annotations but have no gold-store row")


def _read_store(path: str | Path, required_columns: set[str]) -> pd.DataFrame:
    """Read one of our own (utf-8) stores; a malformed store is a violation, not a crash."""
    try:
        df = io_utils.read_csv(path, encoding="utf-8", dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError:
        raise GuardViolation(f"{path} is empty (no header)")
    missing = required_columns - set(df.columns)
    if missing:
        raise GuardViolation(f"{path} lacks column(s) {sorted(missing)}")
    return df


def check_all(split_id: str, *, labels_csv: str | Path | None = None,
              calibration_case_ids: Iterable[str] | None = None) -> list[str]:
    """Run every guard that applies to ``split_id`` and the stores that exist.

    ``labels_csv``: the labels table a trainer will fit on, after partition
    filtering (must hold train cases only). A trainer that filters a
    whole-universe silver table in memory calls ``check_labels_train_only`` /
    ``check_eval_queue_gold_not_trained`` on the filtered frame instead.
    ``calibration_case_ids``: every case a calibration step reads.

    Collects every violation, then raises one ``GuardViolation`` listing them
    all. Returns the names of the guards that ran and passed.
    """
    split = load_split(split_id)  # verifies the split manifest
    gold_path, corrected_path = config.GOLD_STORE_CSV, config.CORRECTED_ANNOTATIONS_CSV
    gold = _read_store(gold_path, {"case_id", "origin"}) if gold_path.is_file() else None
    corrected = _read_store(corrected_path, {"case_id", "label_source"}) if corrected_path.is_file() else None
    labels = _read_store(labels_csv, {"case_id"}) if labels_csv is not None else None
    parent_id = read_manifest(split_dir(split_id)).get("parent")

    guards = [("disjoint", lambda: check_disjoint(split))]
    if parent_id:
        guards.append(("covers parent", lambda: check_covers_parent(split, load_split(parent_id))))
    if gold is not None:
        guards.append(("gold origins", lambda: check_gold_origins(gold)))
        guards.append(("gold-eval in test", lambda: check_gold_eval_in_test(gold, split)))
    for name, table in (("labels", labels), ("corrected annotations", corrected)):
        if table is not None:
            guards.append((f"{name} train-only", lambda t=table: check_labels_train_only(t, split)))
            if gold is not None:
                guards.append((f"{name}: no eval-side queue gold",
                               lambda t=table: check_eval_queue_gold_not_trained(gold, t, split)))
    if corrected is not None:
        empty_gold = pd.DataFrame({"case_id": pd.Series(dtype=str)})
        guards.append(("corrected sources (no audit)",
                       lambda: check_corrected_sources(corrected, gold if gold is not None else empty_gold)))
    if calibration_case_ids is not None:
        guards.append(("calibration inputs", lambda: check_calibration_inputs(calibration_case_ids, split)))

    passed, failures = [], []
    for name, guard in guards:
        try:
            guard()
            passed.append(name)
        except GuardViolation as violation:
            failures.append(f"[{name}] {violation}")
    if failures:
        raise GuardViolation(f"split {split_id!r}: " + "; ".join(failures))
    return passed
