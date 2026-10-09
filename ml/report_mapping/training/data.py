"""Training data from a labels table and a split.

The cases are the train partition's cases that have report text and at least one label row. The term
vocabulary is the sorted set of terms those cases carry; term targets are multi-hot on cancer cases (a
non-cancer case has an all-zero row and is ignored by the term loss); the gate target is 1 for a cancer case.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd

from generations.splits import load_split
from report_mapping import text
from report_mapping.training import labels as labels_mod


@dataclass(frozen=True)
class TrainingData:
    case_ids: list[str]
    texts: list[str]
    vocab: list[str]
    term_targets: np.ndarray   # (N, len(vocab)) float32
    gate_targets: np.ndarray   # (N,) float32


def load_training_data(labels: pd.DataFrame, split_id: str, limit: int = 0) -> TrainingData:
    """``labels``: a labels table (``labels.load_labels_table``). ``limit`` > 0 keeps only the first N cases
    (smoke); the vocabulary still comes from all of them."""
    reports = text.read_reports()
    labels_train = labels_mod.select_train(labels, split_id)  # also runs the leakage guards
    texts_by_case = {c: t for c, t in zip(reports["case_id"], text.build_report_texts(reports)) if t}
    case_ids = labels_mod.training_case_ids(labels_train, list(texts_by_case), load_split(split_id))

    rows = labels_train.assign(case_id=labels_train["case_id"].astype(str).str.strip(),
                               matched_term=labels_train["matched_term"].map(text.clean_text))
    terms_by_case = rows[rows["matched_term"] != ""].groupby("case_id")["matched_term"].agg(set).to_dict()
    vocab = sorted({t for c in case_ids for t in terms_by_case.get(c, ())})
    term_index = {t: i for i, t in enumerate(vocab)}

    case_ids = case_ids[:limit] if limit else case_ids
    term_targets = np.zeros((len(case_ids), len(vocab)), dtype=np.float32)
    for row, case_id in enumerate(case_ids):
        for term in terms_by_case.get(case_id, ()):
            term_targets[row, term_index[term]] = 1.0
    return TrainingData(case_ids=case_ids, texts=[texts_by_case[c] for c in case_ids], vocab=vocab,
                        term_targets=term_targets, gate_targets=labels_mod.gate_targets(labels_train, case_ids))


def split_holdout(targets: np.ndarray, fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """(train indices, held-out indices), both sorted. Stratified on the 0/1 target, deterministic in ``seed``."""
    rng = random.Random(seed)
    held: list[int] = []
    for value in (0.0, 1.0):
        members = [i for i, t in enumerate(targets) if float(t) == value]
        rng.shuffle(members)
        held += members[:round(fraction * len(members))]
    held_set = set(held)
    return [i for i in range(len(targets)) if i not in held_set], sorted(held)


def assign_folds(targets: np.ndarray, k: int, seed: int) -> np.ndarray:
    """(N,) fold number 0..k-1 per case. Stratified on the 0/1 target: each class is shuffled and dealt round-robin."""
    rng = random.Random(seed)
    folds = np.zeros(len(targets), dtype=int)
    for value in (0.0, 1.0):
        members = [i for i, t in enumerate(targets) if float(t) == value]
        rng.shuffle(members)
        folds[members] = np.arange(len(members)) % k
    return folds
