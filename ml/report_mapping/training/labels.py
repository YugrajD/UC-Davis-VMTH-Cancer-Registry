"""Builds the training targets from any labels table.

A labels table is any dataframe with ``case_id, matched_term, matched_group,
matched_code`` — an empty ``matched_term`` means no cancer. The old silver
``annotation.csv`` qualifies unchanged (it carries a few extra columns, which
are simply ignored); so does a corrected-annotations table.

``select_train`` is the one place that touches the split and the guards: it
filters a labels table down to the split's train partition, then runs
``generations.guards.check_labels_train_only`` and
``check_eval_queue_gold_not_trained`` on the *filtered* frame — per
ml-rewrite-plan.md's WP3 -> WP5 note, trainers must never hand the guards a
whole-universe table (that fails by design). Every target builder below
expects an already-filtered frame.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import config
import io_utils
from diagnosis_mapping.silver import load_silver
from generations import guards
from generations.splits import Split, load_split
from report_mapping.text import clean_text

REQUIRED_COLUMNS = ("case_id", "matched_term", "matched_group", "matched_code")


def load_labels_table(labels: str) -> pd.DataFrame:
    """Load a labels table from a silver_id (``diagnosis_mapping.silver.
    load_silver``) or a CSV path. A path that exists on disk is read
    directly; anything else is tried as a silver_id."""
    path = Path(labels)
    if path.is_file():
        df = io_utils.read_csv(path, encoding="utf-8", dtype=str, keep_default_na=False)
    else:
        df = load_silver(labels)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"labels table {labels!r} is missing column(s) {missing}")
    return df


def lineage(labels_df: pd.DataFrame, labels: str) -> dict:
    """What a labels table was built from, for a candidate manifest's ``parents``
    (read by ``generations.triggers``): ``silver_id``, ``gold_train_snapshot`` and
    ``gold_train_codes``. It records what the candidate trained on, not the gold store now.

    ``silver_id`` comes from the table's ``silver_generation`` column, which silver and corrected
    tables both carry. The gold fields come from a corrected-annotations table; any other table
    trained on no gold."""
    def single(column: str) -> str | None:
        if column not in labels_df.columns:
            return None
        values = sorted(set(labels_df[column]) - {""})
        if len(values) > 1:
            raise ValueError(f"labels table {labels!r} mixes {len(values)} {column} values")
        return values[0] if values else None

    gold_codes = 0
    if "label_source" in labels_df.columns:
        gold_codes = int(((labels_df["label_source"] == "gold") & (labels_df["matched_code"] != "")).sum())
    return {"silver_id": single("silver_generation"), "gold_train_snapshot": single("gold_snapshot"),
            "gold_train_codes": gold_codes}


def _gold_for_guard() -> pd.DataFrame:
    """The gold store, or an empty (case_id, origin) frame when it doesn't
    exist yet — mirrors ``generations.guards.check_all``'s own handling."""
    path = config.GOLD_STORE_CSV
    if not path.is_file():
        return pd.DataFrame({"case_id": pd.Series(dtype=str), "origin": pd.Series(dtype=str)})
    return io_utils.read_csv(path, encoding="utf-8", dtype=str, keep_default_na=False)


def select_train(labels: pd.DataFrame, split_id: str) -> pd.DataFrame:
    """Filter ``labels`` to ``split_id``'s train partition, then guard the result.

    Filtering by ``case_id in split.train`` already excludes calibration/test
    cases in the normal case; the guard calls are a defensive second check —
    e.g. against a corrupted split whose partitions aren't actually disjoint.
    """
    split = load_split(split_id)
    ids = labels["case_id"].astype(str).str.strip()
    filtered = labels.loc[ids.isin(split.train)].copy()
    guards.check_labels_train_only(filtered, split)
    guards.check_eval_queue_gold_not_trained(_gold_for_guard(), filtered, split)
    return filtered


def training_case_ids(labels_train: pd.DataFrame, report_case_ids: list[str], split: Split) -> list[str]:
    """The one case universe the model trains on: cases with a report,
    in ``split``'s train partition, AND with at least one row in the
    (already train-filtered) ``labels_train``.

    A case can have a report and sit in train but have no row at all in
    ``labels_train`` — e.g. a vague-without-gold case that ``coding.
    corrected`` drops from the labels table entirely, rather than recording
    it as confirmed non-cancer. Filtering only by ``split.train`` would
    silently turn such a case into an all-zero ("confirmed non-cancer") gate
    target. Order follows ``report_case_ids``.
    """
    label_ids = set(labels_train["case_id"].astype(str).str.strip())
    return [cid for cid in report_case_ids if cid in split.train and cid in label_ids]


def cancer_case_ids(labels_train: pd.DataFrame) -> set[str]:
    """case_ids with at least one non-empty ``matched_term`` row."""
    terms = labels_train["matched_term"].map(clean_text)
    return set(labels_train.loc[terms != "", "case_id"].astype(str).str.strip())


def gate_targets(labels_train: pd.DataFrame, case_ids: list[str]) -> np.ndarray:
    """(len(case_ids),) float32: 1.0 for a case with >= 1 confirmed cancer
    annotation, else 0.0."""
    positives = cancer_case_ids(labels_train)
    return np.array([1.0 if cid in positives else 0.0 for cid in case_ids], dtype=np.float32)
