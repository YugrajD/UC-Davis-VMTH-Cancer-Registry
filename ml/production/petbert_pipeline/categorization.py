"""Embedding-based categorization: cosine similarity against taxonomy label embeddings.

Two categorization strategies are available:

1. run_categorization() — binary-classifier path used during training cycles:
   PresenceClassifier scores across all ~857 labels, argmax selects winner.
   Used by run_cycle.py for intermediate per-cycle scoring.

2. run_categorization_group() — production 3-stage pipeline:
   Stage 1: CasePresenceClassifier gates non-cancer cases.
   Stage 2: GroupClassifier predicts which cancer group(s) a report belongs to.
   Stage 3: ICD-O behavior keyword matching selects the specific term within each group.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ICD_labels import ranked_behaviors
from .embedding import cosine_similarity_matrix


def _behavior_digit(code: str) -> str:
    """Return the ICD-O behavior digit from a code string like '8000/3' → '3'."""
    parts = code.split("/")
    return parts[-1][0] if len(parts) > 1 and parts[-1] else ""


if TYPE_CHECKING:
    from ICD_labels import TaxonomyLabel


@dataclass(frozen=True)
class CategorizationResult:
    final_labels: list[str]          # chosen taxonomy term, "Uncategorized", or ""
    final_indices: list[int]         # index into labels (-1 if empty)
    final_scores: list[float]        # cosine similarity of chosen label
    methods: list[str]               # "embedding", "low_confidence", or "empty"
    embedding_labels: np.ndarray     # top-1 label before thresholding (N,)
    embedding_scores: np.ndarray     # top-1 score before thresholding (N,)
    label_scores: np.ndarray         # full similarity matrix (N, M)
    labels: list[str]                # all taxonomy term strings
    top_k_indices: list[list[int]]   # per row: up to max_predictions label indices
    top_k_scores: list[list[float]]  # per row: corresponding scores
    top_k_methods: list[list[str]]   # per row: "embedding" or "low_confidence"


def run_categorization(
    *,
    texts: list[str],
    text_embeddings: list[np.ndarray],
    label_embeddings: np.ndarray,
    labels: list[str],
    embedding_min_sim: float,
    col_has_content: list[np.ndarray] | None = None,
    max_predictions: int = 5,
    score_matrix: np.ndarray | None = None,
) -> CategorizationResult:
    """Categorize each report using PresenceClassifier scores (training cycle path).

    Used by run_cycle.py for intermediate per-cycle scoring. In production,
    run_categorization_group() is used instead.

    When ``score_matrix`` is provided it is used directly instead of computing
    cosine similarity — the presence classifier has already produced a (N, M)
    probability matrix.
    """
    if score_matrix is not None:
        sims = score_matrix
    elif isinstance(text_embeddings, list):
        sim_matrices: list[np.ndarray] = []
        for i, emb in enumerate(text_embeddings):
            sim = cosine_similarity_matrix(emb, label_embeddings)  # (N, M)
            if col_has_content is not None:
                empty_rows = ~col_has_content[i]
                sim[empty_rows, :] = -np.inf
            sim_matrices.append(sim)
        sims = np.stack(sim_matrices, axis=0).max(axis=0)  # (N, M)
    else:
        sims = cosine_similarity_matrix(text_embeddings, label_embeddings)

    finite_mask = np.isfinite(sims)
    label_means = (
        np.where(finite_mask, sims, 0.0).sum(axis=0)
        / np.maximum(finite_mask.sum(axis=0), 1)
    )
    sims = sims - label_means[np.newaxis, :]

    top_idx = np.argmax(sims, axis=1)
    top_scores = sims[np.arange(len(top_idx)), top_idx].astype(np.float32, copy=False)
    top_labels = np.array([labels[i] for i in top_idx], dtype=object)

    final_labels: list[str] = []
    final_indices: list[int] = []
    final_scores: list[float] = []
    methods: list[str] = []
    top_k_indices: list[list[int]] = []
    top_k_scores: list[list[float]] = []
    top_k_methods: list[list[str]] = []

    for i, text in enumerate(texts):
        if not text:
            final_labels.append("")
            final_indices.append(-1)
            final_scores.append(0.0)
            methods.append("empty")
            top_k_indices.append([])
            top_k_scores.append([])
            top_k_methods.append([])
        elif float(top_scores[i]) >= embedding_min_sim:
            final_labels.append(str(top_labels[i]))
            final_indices.append(int(top_idx[i]))
            final_scores.append(float(top_scores[i]))
            methods.append("embedding")
            sorted_idx = np.argsort(-sims[i])
            k_idxs, k_scores, k_meths = [], [], []
            for rank_idx in sorted_idx:
                score = float(sims[i, rank_idx])
                if score < embedding_min_sim:
                    break
                k_idxs.append(int(rank_idx))
                k_scores.append(score)
                k_meths.append("embedding")
                if len(k_idxs) >= max_predictions:
                    break
            top_k_indices.append(k_idxs)
            top_k_scores.append(k_scores)
            top_k_methods.append(k_meths)
        else:
            final_labels.append("Uncategorized")
            final_indices.append(int(top_idx[i]))
            final_scores.append(float(top_scores[i]))
            methods.append("low_confidence")
            top_k_indices.append([int(top_idx[i])])
            top_k_scores.append([float(top_scores[i])])
            top_k_methods.append(["low_confidence"])

    return CategorizationResult(
        final_labels=final_labels,
        final_indices=final_indices,
        final_scores=final_scores,
        methods=methods,
        embedding_labels=top_labels,
        embedding_scores=top_scores,
        label_scores=sims,
        labels=labels,
        top_k_indices=top_k_indices,
        top_k_scores=top_k_scores,
        top_k_methods=top_k_methods,
    )


def run_categorization_group(
    *,
    texts: list[str],
    mean_embeddings: np.ndarray,              # (N, 768) — mean report embedding per case
    label_embeddings: np.ndarray,             # (M, 768) — all taxonomy label embeddings
    taxonomy_labels: list[TaxonomyLabel],     # length M
    labels: list[str],                        # length M — term strings for display
    group_probs: np.ndarray,                  # (N, num_groups) — GroupClassifier output
    group_names: list[str],                   # length num_groups
    threshold: float,
    max_predictions: int = 5,
    presence_mask: np.ndarray | None = None,  # (N,) bool — True if case passed presence gate
) -> CategorizationResult:
    """Production 3-stage categorization: CasePresenceClassifier → GroupClassifier → KW.

    Stage 1 — CasePresenceClassifier gates non-cancer cases (applied by the caller;
               gate-rejected cases have group_probs zeroed to 0 before this call).
    Stage 2 — GroupClassifier decides which cancer group(s) a report belongs to.
               Groups compete explicitly during training, eliminating the CO floor.
    Stage 3 — Within each predicted group, ICD-O behavior keyword matching selects
               the specific term. This is a much easier sub-problem (fewer candidates,
               tighter cluster).

    Cases where no group exceeds the threshold are predicted as "Unidentified Cancer"
    (method "unidentified_cancer") — the classifier never abstains on a cancer case.
    Gate-rejected cases (presence_mask[i]=False) are the only ones that fall through
    to "Uncategorized" (true negatives).
    """
    N = len(texts)
    M = len(labels)

    group_to_label_indices: dict[str, list[int]] = {}
    for j, tl in enumerate(taxonomy_labels):
        group_to_label_indices.setdefault(tl.group, []).append(j)

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

        case_probs = group_probs[i]  # (num_groups,)
        top_group_idx = int(np.argmax(case_probs))
        embedding_scores_list.append(float(case_probs[top_group_idx]))

        predicted = sorted(
            [g for g in range(len(group_names)) if case_probs[g] >= threshold],
            key=lambda g: -case_probs[g],
        )

        if not predicted:
            gate_passed = presence_mask is None or bool(presence_mask[i])
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

        behavior_rank = ranked_behaviors(text)
        k_idxs: list[int] = []
        k_scores: list[float] = []
        k_meths: list[str] = []
        seen_winners: set[int] = set()

        for g_idx in predicted[:max_predictions]:
            group_name = group_names[g_idx]
            label_idxs = group_to_label_indices.get(group_name, [])
            if not label_idxs:
                continue
            pool = label_idxs
            for b in behavior_rank:
                filtered = [j for j in label_idxs if _behavior_digit(taxonomy_labels[j].code) == b]
                if filtered:
                    pool = filtered
                    break
            pool_embs = label_embeddings[pool]  # (k, 768)
            pool_sims = cosine_similarity_matrix(
                mean_embeddings[i : i + 1], pool_embs
            )[0]  # (k,)
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
