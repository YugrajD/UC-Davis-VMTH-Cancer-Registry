"""Stage 3a — per-group LabelPresenceClassifier scoring.

Each cancer group has its own LabelPresenceClassifier checkpoint (one ``.pt``
per group, named after the slugified group name). Within a predicted group,
the classifier scores each of the group's labels and returns those clearing
``threshold`` (or the argmax label as a fallback when nothing clears it).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from model.label_presence_classifier import LabelPresenceClassifier
from utils.encoding import safe_filename


def load_label_presence_models(
    classifier_dir: str | None,
    group_names: list[str],
) -> dict[str, LabelPresenceClassifier] | None:
    """Load per-group LabelPresenceClassifier checkpoints from a directory.

    Returns None when classifier_dir is not set. Missing .pt files are silently
    skipped — those groups fall back to keyword-only correction.
    """
    if classifier_dir is None:
        return None
    dir_path = Path(classifier_dir)
    if not dir_path.exists():
        print(f"Warning: label_presence_classifier_dir does not exist: {classifier_dir}")
        return None

    models: dict[str, LabelPresenceClassifier] = {}
    for group_name in group_names:
        pt_path = dir_path / f"{safe_filename(group_name)}.pt"
        if pt_path.exists():
            models[group_name] = LabelPresenceClassifier.load(pt_path)
            models[group_name].eval()

    if models:
        print(f"Loaded {len(models)}/{len(group_names)} LabelPresenceClassifier models from {classifier_dir}")
    else:
        print(f"Warning: no LabelPresenceClassifier models found in {classifier_dir}")
    return models


def score_within_group(
    *,
    case_embedding: np.ndarray,            # (1, 768)
    label_indices: list[int],              # global label indices for this group
    label_embeddings: np.ndarray,          # (M, 768) — full taxonomy
    lp_model: LabelPresenceClassifier,
    threshold: float,
) -> tuple[list[int], dict[int, float]]:
    """Score each in-group label and return (selected_indices, score_map).

    Argmax fallback: if no label clears ``threshold``, return the single
    highest-scoring label so the stage never returns an empty pool.
    """
    case_emb_t = torch.from_numpy(case_embedding)
    group_embs_t = torch.from_numpy(label_embeddings[label_indices])
    lp_probs = lp_model.score_matrix(case_emb_t, group_embs_t)[0].numpy()
    selected_within = [j for j, p in enumerate(lp_probs) if p >= threshold]
    if not selected_within:
        selected_within = [int(np.argmax(lp_probs))]
    selected = [label_indices[j] for j in selected_within]
    score_map = {label_indices[j]: float(lp_probs[j]) for j in selected_within}
    return selected, score_map
