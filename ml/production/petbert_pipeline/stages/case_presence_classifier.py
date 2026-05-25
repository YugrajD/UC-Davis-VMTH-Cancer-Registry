"""Stage 1 — CasePresenceClassifier gate.

Reads mean report embeddings (768-dim) and returns a boolean mask: True for
cases whose cancer probability is above ``threshold``. Cases with False are
predicted Uncategorized without ever reaching the GroupClassifier.

When ``classifier_path`` is None the gate is a no-op — every case passes.
"""

from __future__ import annotations

import numpy as np
import torch

from model.case_presence_classifier import CasePresenceClassifier


def run_case_presence_classifier(
    *,
    embeddings: np.ndarray,
    classifier_path: str | None,
    threshold: float,
    device: torch.device,
) -> np.ndarray:
    n = embeddings.shape[0]
    if classifier_path is None:
        return np.ones(n, dtype=bool)

    print(f"Loading case presence classifier from {classifier_path}...")
    case_clf = CasePresenceClassifier.load(classifier_path)
    case_clf.to(device)
    cancer_probs = case_clf.predict_proba(torch.from_numpy(embeddings)).numpy()
    case_clf.cpu()
    del case_clf

    gate_mask = cancer_probs >= threshold
    print(
        f"  Case presence gate (threshold={threshold:.2f}): "
        f"{int(gate_mask.sum())}/{n} cases pass "
        f"({gate_mask.mean() * 100:.1f}%)"
    )
    return gate_mask
