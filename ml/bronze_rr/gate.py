"""Case-presence gate helpers: the held-out epoch-selection split, ROC AUC, and the old (WP14 PetBERT) gate.

The ModernBERT gate itself is a ``encoder.PooledClassifier`` with one output; it is trained in ``train.py``
on the same targets as the old gate (``report_mapping.training.labels.gate_targets``: 1 iff the case has
>= 1 non-empty matched_term).
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

import io_utils
from report_mapping.sections import clean_text

HELDOUT_FRACTION = 0.05
HELDOUT_SEED = 1234  # fixed: the epoch-selection set must not move with the training seed


def split_holdout(targets: np.ndarray, fraction: float = HELDOUT_FRACTION, seed: int = HELDOUT_SEED
                  ) -> tuple[list[int], list[int]]:
    """(train indices, held-out indices), both sorted. Stratified on the 0/1 target, deterministic in ``seed``."""
    rng = random.Random(seed)
    held: list[int] = []
    for value in (0.0, 1.0):
        members = [i for i, t in enumerate(targets) if float(t) == value]
        rng.shuffle(members)
        held += members[:round(fraction * len(members))]
    held_set = set(held)
    return [i for i in range(len(targets)) if i not in held_set], sorted(held)


def roc_auc(scores: np.ndarray, positive: np.ndarray) -> float:
    """Area under the ROC curve (Mann-Whitney U with average ranks); NaN when one class is missing."""
    positive = positive.astype(bool)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def load_old_gate_probs(predictions_csv: Path) -> dict[str, float]:
    """{case_id: case_presence_prob} from a report_mapping predictions CSV. The probability is one value per
    case (repeated on each of its rows); a case whose rows disagree is an error. Cases absent from the CSV
    (production wrote no row for an empty-text case) are read as 0 by ``probs_for``."""
    rows = io_utils.read_csv(predictions_csv, encoding="utf-8", dtype=str, keep_default_na=False,
                             usecols=["case_id", "case_presence_prob"])
    rows["case_id"] = rows["case_id"].map(clean_text)
    rows = rows[rows["case_presence_prob"] != ""]
    values = rows.assign(prob=rows["case_presence_prob"].astype(float))
    spread = values.groupby("case_id")["prob"].nunique()
    if (spread > 1).any():
        raise ValueError(f"{int((spread > 1).sum())} case(s) in {predictions_csv} have more than one case_presence_prob")
    return values.groupby("case_id")["prob"].first().to_dict()


def probs_for(case_ids: list[str], by_case: dict[str, float]) -> np.ndarray:
    """Gate probabilities in ``case_ids`` order; a case without one is 0 (rejected at every threshold > 0)."""
    return np.array([by_case.get(c, 0.0) for c in case_ids], dtype=np.float64)
