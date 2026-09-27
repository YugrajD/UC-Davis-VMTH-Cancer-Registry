"""Builds every stage's training targets from any labels table.

A labels table is any dataframe with ``case_id, matched_term, matched_group,
matched_code`` — an empty ``matched_term`` means no cancer. The old silver
``annotation.csv`` qualifies unchanged (it carries a few extra columns, which
are simply ignored); so will a future corrected-annotations table.

``select_train`` is the one place that touches the split and the guards: it
filters a labels table down to the split's train partition, then runs
``generations.guards.check_labels_train_only`` and
``check_eval_queue_gold_not_trained`` on the *filtered* frame — per
ml-rewrite-plan.md's WP3 -> WP5 note, trainers must never hand the guards a
whole-universe table (that fails by design). Every target builder below
expects an already-filtered frame.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import config
import io_utils
from diagnosis_mapping.silver import load_silver
from generations import guards
from generations.splits import Split, load_split
from report_mapping import sections
from taxonomy.taxonomy import TaxonomyLabel

REQUIRED_COLUMNS = ("case_id", "matched_term", "matched_group", "matched_code")


def _clean(value: object) -> str:
    return sections.clean_text(value)


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


def training_case_ids(labels_train: pd.DataFrame, cache_case_ids: list[str], split: Split) -> list[str]:
    """The one case universe every stage trains on: cases with an embedding,
    in ``split``'s train partition, AND with at least one row in the
    (already train-filtered) ``labels_train``.

    A case can have an embedding and sit in train but have no row at all in
    ``labels_train`` — e.g. a vague-without-gold case that ``coding.
    corrected`` drops from the labels table entirely, rather than recording
    it as confirmed non-cancer. Filtering only by ``split.train`` (as the
    trainers used to) would silently turn such a case into an all-zero
    ("confirmed non-cancer") gate/group target. Every trainer
    (``case_presence.py``, ``group.py``, ``oof.py``) calls this one function
    instead of each re-deriving the case set itself. Order follows
    ``cache_case_ids``.
    """
    label_ids = set(labels_train["case_id"].astype(str).str.strip())
    return [cid for cid in cache_case_ids if cid in split.train and cid in label_ids]


def cancer_case_ids(labels_train: pd.DataFrame) -> set[str]:
    """case_ids with at least one non-empty ``matched_term`` row."""
    terms = labels_train["matched_term"].map(_clean)
    return set(labels_train.loc[terms != "", "case_id"].astype(str).str.strip())


def gate_targets(labels_train: pd.DataFrame, case_ids: list[str]) -> np.ndarray:
    """(len(case_ids),) float32: 1.0 for a case with >= 1 confirmed cancer
    annotation, else 0.0. Mirrors ``build_case_presence_dataset.py``."""
    positives = cancer_case_ids(labels_train)
    return np.array([1.0 if cid in positives else 0.0 for cid in case_ids], dtype=np.float32)


@dataclass(frozen=True)
class GroupTargets:
    case_ids: list[str]
    group_names: list[str]        # common groups (sorted) + "Uncommon" if any group merged
    targets: np.ndarray           # (N, G) float32 multi-hot
    uncommon_groups: list[str]    # groups merged into "Uncommon", sorted
    class_weights: np.ndarray     # (G,) float32, uncapped (negatives / positives per group)


def group_targets(
    labels_train: pd.DataFrame,
    case_ids: list[str],
    *,
    uncommon_threshold: int,
    forced_uncommon: tuple[str, ...],
) -> GroupTargets:
    """Multi-hot group targets over ``case_ids`` (the embedding-cache universe,
    already filtered to train). Mirrors ``build_training_data.py``: groups with
    fewer than ``uncommon_threshold`` cases *within* ``case_ids``, plus every
    name in ``forced_uncommon``, merge into one "Uncommon" column. A case in
    ``labels_train`` but not in ``case_ids`` (no embedding) contributes nothing;
    a case in ``case_ids`` with no cancer row stays an all-zero row."""
    rows = labels_train.copy()
    rows["case_id"] = rows["case_id"].astype(str).str.strip()
    rows["matched_group"] = rows["matched_group"].map(_clean)
    rows = rows[rows["matched_group"] != ""]

    all_groups = sorted(rows["matched_group"].unique())
    case_id_set = set(case_ids)
    in_universe = rows[rows["case_id"].isin(case_id_set)]
    per_group = in_universe["matched_group"].value_counts()
    force = set(forced_uncommon)

    common_groups = sorted(
        g for g in all_groups if per_group.get(g, 0) >= uncommon_threshold and g not in force
    ) if uncommon_threshold > 0 else sorted(g for g in all_groups if g not in force)
    uncommon_group_names = sorted(
        g for g in all_groups if g not in common_groups
    ) if uncommon_threshold > 0 else sorted(force & set(all_groups))

    has_uncommon = len(uncommon_group_names) > 0
    final_groups = common_groups + (["Uncommon"] if has_uncommon else [])
    group_idx = {g: i for i, g in enumerate(final_groups)}
    case_idx = {cid: i for i, cid in enumerate(case_ids)}

    N, G = len(case_ids), len(final_groups)
    targets = np.zeros((N, G), dtype=np.float32)
    uncommon_col = group_idx.get("Uncommon")
    for _, row in in_universe.iterrows():
        ci = case_idx.get(row["case_id"])
        if ci is None:
            continue
        gi = group_idx.get(row["matched_group"], uncommon_col)
        if gi is not None:
            targets[ci, gi] = 1.0

    positive_counts = targets.sum(axis=0)
    negative_counts = N - positive_counts
    class_weights = (negative_counts / np.maximum(positive_counts, 1.0)).astype(np.float32)

    return GroupTargets(
        case_ids=list(case_ids), group_names=final_groups, targets=targets,
        uncommon_groups=uncommon_group_names, class_weights=class_weights,
    )


def label_presence_pairs(
    labels_train: pd.DataFrame,
    taxonomy_labels: list[TaxonomyLabel],
    group_name: str,
    *,
    uncommon_group_names: list[str] | None = None,
    negs_per_pos: int,
    seed: int,
) -> pd.DataFrame:
    """(case_id, label_term, label_group, target) rows for one group's LabelPresence
    head. Mirrors ``build_label_presence_pairs.py``: one positive row per
    train-partition (case, label) annotation in the group, plus
    ``negs_per_pos`` within-group negatives sampled uniformly at random from
    the taxonomy's other labels in the group. Pass ``group_name="Uncommon"``
    with ``uncommon_group_names`` set to build the merged Uncommon head's
    pairs. Returns an empty frame (with the right columns) when the group has
    fewer than 2 taxonomy labels or no train-partition annotations."""
    columns = ["case_id", "label_term", "label_group", "target"]
    if group_name == "Uncommon":
        if not uncommon_group_names:
            raise ValueError("uncommon_group_names must be provided when group_name='Uncommon'")
        target_groups = set(uncommon_group_names)
        labels_in_group = [t for t in taxonomy_labels if t.group in target_groups]
    else:
        labels_in_group = [t for t in taxonomy_labels if t.group == group_name]

    if len(labels_in_group) < 2:
        return pd.DataFrame(columns=columns)

    label_pool = [(t.term, t.group) for t in labels_in_group]

    rows = labels_train.copy()
    rows["matched_group"] = rows["matched_group"].map(_clean)
    rows["matched_term"] = rows["matched_term"].map(_clean)
    rows["case_id"] = rows["case_id"].astype(str).str.strip()
    if group_name == "Uncommon":
        group_rows = rows[rows["matched_group"].isin(set(uncommon_group_names))]
    else:
        group_rows = rows[rows["matched_group"] == group_name]
    group_rows = group_rows[group_rows["matched_term"] != ""]
    if group_rows.empty:
        return pd.DataFrame(columns=columns)

    rng = random.Random(seed)
    out: list[dict] = []
    for _, row in group_rows.iterrows():
        pos_term, pos_group, case_id = row["matched_term"], row["matched_group"], row["case_id"]
        out.append({"case_id": case_id, "label_term": pos_term, "label_group": pos_group, "target": 1})
        neg_pool = [(t, g) for t, g in label_pool if t != pos_term]
        if not neg_pool:
            continue
        n_neg = min(negs_per_pos, len(neg_pool))
        for neg_term, neg_group in rng.sample(neg_pool, n_neg):
            out.append({"case_id": case_id, "label_term": neg_term, "label_group": neg_group, "target": 0})
    return pd.DataFrame(out, columns=columns)


def _section_label(group: tuple[str, ...]) -> str:
    """Readable section name, matching legacy's ``_section_name``: kept for
    continuity with legacy pair CSVs, though the trainer never keys on it."""
    return group[0] if len(group) == 1 else "+".join(group)


def contrastive_pairs(
    labels_train: pd.DataFrame,
    report_frame: pd.DataFrame,
    *,
    min_report_chars: int = 10,
) -> pd.DataFrame:
    """Per-section (report_text, label_text) pairs for backbone adaptation.
    Mirrors ``build_contrastive_dataset.py``: one row per (case, label,
    section) where that section has >= ``min_report_chars`` characters.
    ``report_frame`` is report.csv-shaped (has ``case_id`` + the raw section
    source columns; ``report_mapping.sections.build_section_frame`` derives
    the three section columns here)."""
    columns = ["case_id", "section", "report_text", "label_text", "matched_term", "matched_group"]
    case_to_labels: dict[str, set[tuple[str, str]]] = {}
    for _, row in labels_train.iterrows():
        case_id = str(row["case_id"]).strip()
        term = _clean(row.get("matched_term", ""))
        group = _clean(row.get("matched_group", ""))
        if not case_id or not term or not group:
            continue
        case_to_labels.setdefault(case_id, set()).add((term, group))
    if not case_to_labels:
        return pd.DataFrame(columns=columns)

    sec_frame = sections.build_section_frame(report_frame)
    texts_by_col = sections.section_texts(sec_frame)
    case_ids = sec_frame["case_id"].map(_clean).tolist()

    rows: list[dict] = []
    for i, case_id in enumerate(case_ids):
        labels_for_case = case_to_labels.get(case_id)
        if not labels_for_case:
            continue
        kept = [
            (sections.CONCAT_3_SECTIONS[s], sections.SECTION_COLUMNS[s], texts_by_col[sections.SECTION_COLUMNS[s]][i])
            for s in range(len(sections.SECTION_COLUMNS))
            if len(texts_by_col[sections.SECTION_COLUMNS[s]][i]) >= min_report_chars
        ]
        if not kept:
            continue
        for term, group in labels_for_case:
            label_text = f"{term} {group}"
            for source_group, _col, text in kept:
                rows.append({
                    "case_id": case_id, "section": _section_label(source_group), "report_text": text,
                    "label_text": label_text, "matched_term": term, "matched_group": group,
                })
    return pd.DataFrame(rows, columns=columns)
