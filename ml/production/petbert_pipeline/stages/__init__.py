"""Per-stage modules of the 4-stage production pipeline.

Stage 1 — case_presence_classifier:  CasePresenceClassifier gate
Stage 2 — group_classifier:          GroupClassifier
Stage 3a — label_presence_classifier: per-group LabelPresenceClassifier
Stage 3b — keyword_correction:       ICD-O behavior + subtype filter

This package also exposes ``categorize_per_case``: the dispatcher that loops
over cases, dispatches to Stage 3a (when an LP model is loaded for the group)
or directly to Stage 3b, and assembles the final ``CategorizationResult``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from ..embedding import cosine_similarity_matrix
from ..types import CategorizationResult
from .case_presence_classifier import run_case_presence_classifier
from .group_classifier import run_group_classifier
from .keyword_correction import apply_keyword_correction
from .label_presence_classifier import load_label_presence_models, score_within_group

if TYPE_CHECKING:
    from ICD_labels import TaxonomyLabel
    from model.label_presence_classifier import LabelPresenceClassifier


__all__ = [
    "run_case_presence_classifier",
    "run_group_classifier",
    "load_label_presence_models",
    "score_within_group",
    "apply_keyword_correction",
    "categorize_per_case",
]


def categorize_per_case(
    *,
    texts: list[str],
    mean_embeddings: np.ndarray,              # (N, 768)
    label_embeddings: np.ndarray,             # (M, 768)
    taxonomy_labels: list["TaxonomyLabel"],
    labels: list[str],
    group_probs: np.ndarray,                  # (N, num_groups)
    group_names: list[str],
    threshold: float,
    max_predictions: int = 5,
    presence_mask: np.ndarray | None = None,
    uncommon_groups: frozenset[str] = frozenset(),
    fallback_to_argmax: bool = True,
    label_presence_models: dict[str, "LabelPresenceClassifier"] | None = None,
    label_presence_threshold: float = 0.5,
) -> CategorizationResult:
    """Dispatch each case to Stage 3a (LP) and/or Stage 3b (KW), assemble result.

    Cases where no group exceeds the threshold and ``fallback_to_argmax`` is
    False are predicted "Unidentified Cancer" — the classifier never abstains
    on a cancer case. Gate-rejected cases (presence_mask[i]=False) are the
    only ones that fall through to "Uncategorized" (true negatives).
    """
    N = len(texts)
    M = len(labels)

    group_to_label_indices: dict[str, list[int]] = {}
    for j, tl in enumerate(taxonomy_labels):
        group_to_label_indices.setdefault(tl.group, []).append(j)

    uncommon_label_indices: list[int] = [
        j for g in uncommon_groups for j in group_to_label_indices.get(g, [])
    ]

    group_name_to_idx = {g: i for i, g in enumerate(group_names)}
    label_scores = np.zeros((N, M), dtype=np.float32)
    for j, tl in enumerate(taxonomy_labels):
        g_idx = group_name_to_idx.get(tl.group)
        if g_idx is not None:
            label_scores[:, j] = group_probs[:, g_idx]

    final_labels: list[str] = []
    final_indices: list[int] = []
    final_scores: list[float] = []
    methods: list[str] = []
    top_k_indices: list[list[int]] = []
    top_k_scores: list[list[float]] = []
    top_k_methods: list[list[str]] = []
    embedding_labels_list: list[str] = []
    embedding_scores_list: list[float] = []

    for i, text in enumerate(texts):
        if not text:
            final_labels.append("")
            final_indices.append(-1)
            final_scores.append(0.0)
            methods.append("empty")
            top_k_indices.append([])
            top_k_scores.append([])
            top_k_methods.append([])
            embedding_labels_list.append("")
            embedding_scores_list.append(0.0)
            continue

        case_probs = group_probs[i]
        top_group_idx = int(np.argmax(case_probs))
        embedding_scores_list.append(float(case_probs[top_group_idx]))

        predicted = sorted(
            [g for g in range(len(group_names)) if case_probs[g] >= threshold],
            key=lambda g: -case_probs[g],
        )

        if not predicted:
            gate_passed = presence_mask is None or bool(presence_mask[i])
            if fallback_to_argmax and gate_passed:
                predicted = [top_group_idx]
            else:
                label_str = "Unidentified Cancer" if gate_passed else "Uncategorized"
                method_str = "unidentified_cancer" if gate_passed else "low_confidence"
                best_idxs = group_to_label_indices.get(group_names[top_group_idx], [])
                if best_idxs:
                    emb_sims = cosine_similarity_matrix(
                        mean_embeddings[i : i + 1], label_embeddings[best_idxs]
                    )[0]
                    best_within = int(np.argmax(emb_sims))
                    embedding_labels_list.append(labels[best_idxs[best_within]])
                else:
                    embedding_labels_list.append(label_str)
                final_labels.append(label_str)
                final_indices.append(-1)
                final_scores.append(float(case_probs[top_group_idx]))
                methods.append(method_str)
                top_k_indices.append([-1])
                top_k_scores.append([float(case_probs[top_group_idx])])
                top_k_methods.append([method_str])
                continue

        k_idxs: list[int] = []
        k_scores: list[float] = []
        k_meths: list[str] = []
        seen_winners: set[int] = set()

        for g_idx in predicted[:max_predictions]:
            group_name = group_names[g_idx]
            if group_name == "Uncommon":
                label_idxs = uncommon_label_indices
            else:
                label_idxs = group_to_label_indices.get(group_name, [])
            if not label_idxs:
                continue

            lp_model = (label_presence_models or {}).get(group_name)
            if lp_model is not None:
                # Stage 3a: LabelPresenceClassifier picks within the group.
                lp_pool, lp_score_map = score_within_group(
                    case_embedding=mean_embeddings[i : i + 1],
                    label_indices=label_idxs,
                    label_embeddings=label_embeddings,
                    lp_model=lp_model,
                    threshold=label_presence_threshold,
                )
                # Stage 3b: keyword post-filter on the LP-selected pool.
                pool = apply_keyword_correction(
                    text=text,
                    pool=lp_pool,
                    taxonomy_labels=taxonomy_labels,
                    labels=labels,
                    group_name=group_name,
                )
                for best_label_idx in pool:
                    if best_label_idx in seen_winners:
                        continue
                    seen_winners.add(best_label_idx)
                    k_idxs.append(best_label_idx)
                    k_scores.append(lp_score_map.get(best_label_idx, float(case_probs[g_idx])))
                    k_meths.append("label_presence")
            else:
                # Stage 3b standalone: keyword filter + cosine similarity.
                pool = apply_keyword_correction(
                    text=text,
                    pool=label_idxs,
                    taxonomy_labels=taxonomy_labels,
                    labels=labels,
                    group_name=group_name,
                )
                if not pool:
                    continue
                pool_embs = label_embeddings[pool]
                pool_sims = cosine_similarity_matrix(
                    mean_embeddings[i : i + 1], pool_embs
                )[0]
                best_within = int(np.argmax(pool_sims))
                best_label_idx = pool[best_within]
                if best_label_idx in seen_winners:
                    continue
                seen_winners.add(best_label_idx)
                k_idxs.append(best_label_idx)
                k_scores.append(float(case_probs[g_idx]))
                k_meths.append("embedding")

        if not k_idxs:
            gate_passed = presence_mask is None or bool(presence_mask[i])
            label_str = "Unidentified Cancer" if gate_passed else "Uncategorized"
            method_str = "unidentified_cancer" if gate_passed else "low_confidence"
            final_labels.append(label_str)
            final_indices.append(-1)
            final_scores.append(0.0)
            methods.append(method_str)
            embedding_labels_list.append(label_str)
            top_k_indices.append([-1])
            top_k_scores.append([0.0])
            top_k_methods.append([method_str])
            continue

        final_labels.append(labels[k_idxs[0]])
        final_indices.append(k_idxs[0])
        final_scores.append(k_scores[0])
        methods.append("embedding")
        embedding_labels_list.append(labels[k_idxs[0]])
        top_k_indices.append(k_idxs)
        top_k_scores.append(k_scores)
        top_k_methods.append(k_meths)

    embedding_labels = np.array(embedding_labels_list, dtype=object)
    embedding_scores = np.array(embedding_scores_list, dtype=np.float32)

    return CategorizationResult(
        final_labels=final_labels,
        final_indices=final_indices,
        final_scores=final_scores,
        methods=methods,
        embedding_labels=embedding_labels,
        embedding_scores=embedding_scores,
        label_scores=label_scores,
        labels=labels,
        top_k_indices=top_k_indices,
        top_k_scores=top_k_scores,
        top_k_methods=top_k_methods,
    )
