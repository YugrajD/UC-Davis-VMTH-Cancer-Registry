"""Stage 2 — GroupClassifier.

Predicts which cancer group(s) each case belongs to. Gate-rejected cases
(``presence_gate_mask[i] is False``) have their group probabilities zeroed so
they fall through to Uncategorized in Stage 3.
"""

from __future__ import annotations

import numpy as np
import torch

import config
from model.group_classifier import GroupClassifier


def run_group_classifier(
    *,
    col_emb_concat: np.ndarray,
    case_ids: list[str],
    classifier_path: str,
    presence_gate_mask: np.ndarray,
    device: torch.device,
    use_demographics: bool = False,
    demographics_csv: str = config.DEMOGRAPHICS_CSV,
    demographics_encoder_spec: str = config.DEMOGRAPHICS_ENCODER_SPEC,
) -> tuple[np.ndarray, list[str]]:
    print(f"Loading group classifier from {classifier_path}...")
    group_clf, group_names = GroupClassifier.load(classifier_path)
    meta = GroupClassifier.load_meta(classifier_path)

    # Append demographics block when checkpoint was trained with it
    if meta["uses_demographics"]:
        from features.demographics import DemographicsEncoder
        enc = DemographicsEncoder().load_spec(demographics_encoder_spec, demographics_csv)
        demo_block = enc.transform(case_ids)
        input_emb = np.hstack([col_emb_concat, demo_block]).astype(np.float32)
        assert input_emb.shape[1] == meta["demo_width"] + col_emb_concat.shape[1], (
            f"Demographics width mismatch: checkpoint expects demo_width={meta['demo_width']}, "
            f"encoder produced {enc.dim}. Encoder spec may be stale."
        )
        assert input_emb.shape[1] == group_clf.emb_dim, (
            f"Embedding width mismatch: model expects {group_clf.emb_dim}, got {input_emb.shape[1]}."
        )
    else:
        input_emb = col_emb_concat

    group_clf.to(device)
    group_probs = group_clf.predict_proba(torch.from_numpy(input_emb)).numpy()
    group_clf.cpu()
    del group_clf

    group_probs[~presence_gate_mask] = 0.0
    return group_probs, group_names
