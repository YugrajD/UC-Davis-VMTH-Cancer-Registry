"""Helpers for auxiliary carcinoma/sarcoma label constraints by anon_id."""

from __future__ import annotations

import csv
import os
import re
from typing import Iterable


_WORD_RE = re.compile(r"[a-z0-9]+")


def normalize_text(text: str) -> str:
    tokens = _WORD_RE.findall((text or "").lower())
    return " ".join(tokens)


def load_anon_ids(csv_path: str, id_col: str = "anon_id") -> set[str]:
    if not csv_path or not os.path.exists(csv_path):
        return set()
    anon_ids: set[str] = set()
    with open(csv_path, newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            return anon_ids
        if id_col not in reader.fieldnames:
            id_col = reader.fieldnames[0]
        for row in reader:
            anon_id = (row.get(id_col) or "").strip()
            if anon_id:
                anon_ids.add(anon_id)
    return anon_ids


def candidate_indices_for_aux_label(
    labels: list[str],
    *,
    aux_label: str,
) -> list[int]:
    needle = normalize_text(aux_label)
    indices = []
    for idx, label in enumerate(labels):
        label_norm = normalize_text(label)
        if needle in label_norm:
            indices.append(idx)
    return indices


def best_index_with_constraint(
    score_row,
    candidate_indices: Iterable[int],
) -> tuple[int, float] | None:
    best_idx = -1
    best_score = -1.0
    for idx in candidate_indices:
        score = float(score_row[idx])
        if score > best_score:
            best_idx = int(idx)
            best_score = score
    if best_idx < 0:
        return None
    return best_idx, best_score
