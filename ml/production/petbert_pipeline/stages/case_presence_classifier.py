"""Stage 1 — CasePresenceClassifier gate.

Reads mean report embeddings (2304-dim, or 2304+D when demographics are active)
and returns a boolean mask: True for cases whose cancer probability is above
``threshold``. Cases with False are predicted Non-Cancer without ever reaching
the GroupClassifier.

When ``classifier_path`` is None the gate is a no-op — every case passes.
"""

from __future__ import annotations

import numpy as np
import torch

import config
from model.case_presence_classifier import CasePresenceClassifier


def run_case_presence_classifier(
    *,
    embeddings: np.ndarray,
    case_ids: list[str],
    classifier_path: str | None,
    threshold: float,
    device: torch.device,
    use_demographics: bool = False,
    demographics_csv: str = config.DEMOGRAPHICS_CSV,
    demographics_encoder_spec: str = config.DEMOGRAPHICS_ENCODER_SPEC,
) -> tuple[np.ndarray, np.ndarray]:
    n = embeddings.shape[0]
    if classifier_path is None:
        return np.ones(n, dtype=bool), np.full(n, np.nan, dtype=np.float32)

    print(f"Loading case presence classifier from {classifier_path}...")
    case_clf = CasePresenceClassifier.load(classifier_path)
    meta = CasePresenceClassifier.load_meta(classifier_path)

    # Append demographics block when checkpoint was trained with it
    if meta["uses_demographics"]:
        from features.demographics import DemographicsEncoder
        enc = DemographicsEncoder().load_spec(demographics_encoder_spec, demographics_csv)
        demo_block = enc.transform(case_ids)
        input_emb = np.hstack([embeddings, demo_block]).astype(np.float32)
        expected_width = embeddings.shape[1] + enc.dim
        assert input_emb.shape[1] == meta["demo_width"] + embeddings.shape[1], (
            f"Demographics width mismatch: checkpoint expects demo_width={meta['demo_width']}, "
            f"encoder produced {enc.dim}. Encoder spec may be stale."
        )
        assert input_emb.shape[1] == case_clf.emb_dim, (
            f"Embedding width mismatch: model expects {case_clf.emb_dim}, got {input_emb.shape[1]}."
        )
    else:
        input_emb = embeddings

    case_clf.to(device)
    cancer_probs = case_clf.predict_proba(torch.from_numpy(input_emb)).numpy()
    case_clf.cpu()
    del case_clf

    gate_mask = cancer_probs >= threshold
    print(
        f"  Case presence gate (threshold={threshold:.2f}): "
        f"{int(gate_mask.sum())}/{n} cases pass "
        f"({gate_mask.mean() * 100:.1f}%)"
    )
    return gate_mask, cancer_probs.astype(np.float32, copy=False)
