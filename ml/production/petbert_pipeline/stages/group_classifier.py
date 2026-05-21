"""Stage 2 — GroupClassifier.

Predicts which cancer group(s) each case belongs to. Gate-rejected cases
(``presence_gate_mask[i] is False``) have their group probabilities zeroed so
they fall through to Uncategorized in Stage 3.
"""

from __future__ import annotations

import numpy as np
import torch

from model.group_classifier import GroupClassifier


def run_group_classifier(
    *,
    col_emb_concat: np.ndarray,
    classifier_path: str,
    presence_gate_mask: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, list[str]]:
    print(f"Loading group classifier from {classifier_path}...")
    group_clf, group_names = GroupClassifier.load(classifier_path)
    group_clf.to(device)
    group_probs = group_clf.predict_proba(torch.from_numpy(col_emb_concat)).numpy()
    group_clf.cpu()
    del group_clf

    group_probs[~presence_gate_mask] = 0.0
    return group_probs, group_names
