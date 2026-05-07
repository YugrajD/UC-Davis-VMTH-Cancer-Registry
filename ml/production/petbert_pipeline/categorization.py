"""Embedding-based categorization: cosine similarity against taxonomy label embeddings.

Three categorization strategies are available:

1. run_categorization() — original approach: cosine similarity (or binary PresenceClassifier
   scores) across all ~857 labels, argmax selects winner. Suffers from a ~42% completely-off
   floor because labels compete implicitly.

2. run_categorization_group() — two-stage approach: GroupClassifier predicts which cancer
   group(s) a report belongs to (explicit multi-label competition), then cosine similarity
   selects the best term within each predicted group. Eliminates the completely-off floor.

3. run_categorization_group_keyword() — two-stage approach requiring only a PresenceClassifier:
   Stage 1 uses the top-scoring term's group as the predicted group (free — no separate model).
   Stage 2 uses ICD-O behavior keyword matching to select the specific term within that group,
   converting "slightly off" (right group, wrong term) predictions into "good" predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ICD_labels import best_behavior
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
    text_embeddings: np.ndarray | list[np.ndarray],
    label_embeddings: np.ndarray,
    labels: list[str],
    embedding_min_sim: float,
    col_has_content: list[np.ndarray] | None = None,
    max_predictions: int = 5,
    score_matrix: np.ndarray | None = None,
) -> CategorizationResult:
    """Categorize each diagnosis by similarity to taxonomy label embeddings.

    When ``text_embeddings`` is a list of per-column embedding arrays, each
    column independently computes its similarity scores and the label with the
    highest score across *any* column wins.  Empty cells (tracked via
    ``col_has_content``) are masked out so they cannot influence the result.

    If ``score_matrix`` is provided (shape N×M), it is used directly instead of
    computing cosine similarity — e.g. when the presence classifier has already
    produced a pre-scored (N, M) probability matrix.
    """
    if score_matrix is not None:
        sims = score_matrix
    elif isinstance(text_embeddings, list):
        sim_matrices: list[np.ndarray] = []
        for i, emb in enumerate(text_embeddings):
            sim = cosine_similarity_matrix(emb, label_embeddings)  # (N, M)
            if col_has_content is not None:
                # Rows where this column is empty cannot win — mask with -inf.
                empty_rows = ~col_has_content[i]  # (N,) True where cell is empty
                sim[empty_rows, :] = -np.inf
            sim_matrices.append(sim)
        # Element-wise max across columns: highest similarity for each (row, label) pair.
        sims = np.stack(sim_matrices, axis=0).max(axis=0)  # (N, M)
    else:
        sims = cosine_similarity_matrix(text_embeddings, label_embeddings)

    # Subtract per-label mean so universally-high labels (e.g. "Pyogenic granuloma")
    # don't dominate argmax for every case.  After this shift, embedding_min_sim
    # compares centered scores: 0.0 = average similarity, positive = above average.
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
            # Collect all labels above the threshold, ranked by score (up to max_predictions)
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
    mean_embeddings: np.ndarray,          # (N, 768) — mean report embedding per case
    label_embeddings: np.ndarray,         # (M, 768) — all taxonomy label embeddings
    taxonomy_labels: list[TaxonomyLabel], # length M
    labels: list[str],                    # length M — term strings for display
    group_probs: np.ndarray,              # (N, num_groups) — GroupClassifier output
    group_names: list[str],               # length num_groups
    threshold: float,
    max_predictions: int = 5,
) -> CategorizationResult:
    """Two-stage categorization: group classifier → term selection within group.

    Stage 1 — GroupClassifier decides which cancer group(s) a report belongs to.
               Groups compete explicitly during training, eliminating the ~42% CO floor.
    Stage 2 — Within each predicted group, cosine similarity selects the best term.
               This is a much easier sub-problem (fewer candidates, tighter cluster).

    Cases where no group exceeds the threshold are predicted as Uncategorized.
    """
    N = len(texts)
    M = len(labels)

    # Build group → [label_indices] mapping (static, from taxonomy)
    group_to_label_indices: dict[str, list[int]] = {}
    for j, tl in enumerate(taxonomy_labels):
        group_to_label_indices.setdefault(tl.group, []).append(j)

    # Map group probabilities to term level for label_scores (N, M) compatibility.
    # Each term receives the probability of its group — preserves output format
    # for downstream similarity CSV writers.
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

        # Find groups above threshold, sorted by probability descending
        predicted = sorted(
            [g for g in range(len(group_names)) if case_probs[g] >= threshold],
            key=lambda g: -case_probs[g],
        )

        if not predicted:
            # No group confident enough → Uncategorized
            # Record top-1 group term as embedding_label for provenance
            best_idxs = group_to_label_indices.get(group_names[top_group_idx], [])
            if best_idxs:
                emb_sims = cosine_similarity_matrix(
                    mean_embeddings[i : i + 1], label_embeddings[best_idxs]
                )[0]
                best_within = int(np.argmax(emb_sims))
                embedding_labels_list.append(labels[best_idxs[best_within]])
            else:
                embedding_labels_list.append("Uncategorized")
            final_labels.append("Uncategorized")
            final_indices.append(-1)
            final_scores.append(float(case_probs[top_group_idx]))
            methods.append("low_confidence")
            top_k_indices.append([-1])
            top_k_scores.append([float(case_probs[top_group_idx])])
            top_k_methods.append(["low_confidence"])
            continue

        # For each predicted group, pick the best term by behavior keyword filtering
        # then cosine similarity within the filtered pool (same approach as group-keyword mode).
        behavior = best_behavior(text)
        k_idxs: list[int] = []
        k_scores: list[float] = []
        k_meths: list[str] = []

        for g_idx in predicted[:max_predictions]:
            group_name = group_names[g_idx]
            label_idxs = group_to_label_indices.get(group_name, [])
            if not label_idxs:
                continue
            filtered: list[int] = []
            if behavior:
                filtered = [j for j in label_idxs if _behavior_digit(taxonomy_labels[j].code) == behavior]
            pool = filtered if filtered else label_idxs
            pool_embs = label_embeddings[pool]  # (k, 768)
            pool_sims = cosine_similarity_matrix(
                mean_embeddings[i : i + 1], pool_embs
            )[0]  # (k,)
            best_within = int(np.argmax(pool_sims))
            best_label_idx = pool[best_within]
            k_idxs.append(best_label_idx)
            k_scores.append(float(case_probs[g_idx]))
            k_meths.append("embedding")

        if not k_idxs:
            final_labels.append("Uncategorized")
            final_indices.append(-1)
            final_scores.append(0.0)
            methods.append("low_confidence")
            embedding_labels_list.append("Uncategorized")
            top_k_indices.append([-1])
            top_k_scores.append([0.0])
            top_k_methods.append(["low_confidence"])
            continue

        # Top-1 prediction for final_labels / final_indices
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


def run_categorization_group_keyword(
    *,
    texts: list[str],
    score_matrix: np.ndarray,               # (N, M) PresenceClassifier probabilities
    taxonomy_labels: list["TaxonomyLabel"], # length M
    labels: list[str],                      # length M — term strings for display
    embedding_min_sim: float,
    max_predictions: int = 5,
) -> CategorizationResult:
    """Two-stage categorization using PresenceClassifier + behavior keyword matching.

    Stage 1 — Identical to run_categorization(): mean-center scores, find argmax,
               apply threshold. This determines whether a prediction is made at all
               (and thus CO, FP, FN are unchanged vs the default mode).
    Stage 2 — Only runs when Stage 1 would have made a prediction. Within the
               Stage 1 group, ICD-O behavior keyword matching on the report text
               selects the specific term. The behavior digit (0=benign, 1=borderline,
               2=in situ, 3=malignant, 6=metastatic) is inferred from clinical
               vocabulary; candidates are filtered to that digit before taking the
               highest raw PresenceClassifier score as the winner.
               If no keyword signal is found, all candidates in the group compete.

    CO, FP, and FN are identical to default mode — only Good/Slight can change.
    """
    # Stage 1: center scores, find top-1, check threshold — exactly as run_categorization().
    label_means = score_matrix.mean(axis=0)
    sims = score_matrix - label_means[np.newaxis, :]

    # Build group → [label_indices] mapping (static, from taxonomy).
    group_to_label_indices: dict[str, list[int]] = {}
    for j, tl in enumerate(taxonomy_labels):
        group_to_label_indices.setdefault(tl.group, []).append(j)

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

        # Stage 1: identical decision to default mode.
        top_global_idx = int(np.argmax(sims[i]))
        top_global_score = float(sims[i, top_global_idx])
        embedding_scores_list.append(top_global_score)
        embedding_labels_list.append(labels[top_global_idx])

        if top_global_score < embedding_min_sim:
            # Below threshold — same "Uncategorized" decision as default mode.
            final_labels.append("Uncategorized")
            final_indices.append(top_global_idx)
            final_scores.append(top_global_score)
            methods.append("low_confidence")
            top_k_indices.append([top_global_idx])
            top_k_scores.append([top_global_score])
            top_k_methods.append(["low_confidence"])
            continue

        # Build the same default top-k as run_categorization(): all labels above
        # threshold, sorted by centered score descending, up to max_predictions.
        sorted_idx = np.argsort(-sims[i])
        default_topk: list[int] = []
        for rank_idx in sorted_idx:
            if float(sims[i, rank_idx]) < embedding_min_sim:
                break
            default_topk.append(int(rank_idx))
            if len(default_topk) >= max_predictions:
                break

        # Stage 2: apply behavior keyword filtering independently to each top-k row.
        # For each Stage-1 term, find its group, then pick the best keyword-matching
        # term within that group by raw PresenceClassifier score.
        # Raw scores are used within-group because centering penalizes common labels.
        behavior = best_behavior(text)
        k_idxs: list[int] = []
        k_scores: list[float] = []
        for stage1_idx in default_topk:
            group = taxonomy_labels[stage1_idx].group
            cands = group_to_label_indices.get(group, [])
            if not cands:
                # Degenerate: no labels in this group — keep the Stage 1 term.
                k_idxs.append(stage1_idx)
                k_scores.append(float(sims[i, stage1_idx]))
                continue
            filtered: list[int] = []
            if behavior:
                filtered = [j for j in cands if _behavior_digit(taxonomy_labels[j].code) == behavior]
            pool = filtered if filtered else cands
            pool_raw = score_matrix[i, pool]
            winner = pool[int(np.argmax(pool_raw))]
            k_idxs.append(winner)
            k_scores.append(float(sims[i, stage1_idx]))  # preserve Stage 1 score for ordering

        final_labels.append(labels[k_idxs[0]])
        final_indices.append(k_idxs[0])
        final_scores.append(top_global_score)  # report Stage 1 centered score as confidence
        methods.append("embedding")
        top_k_indices.append(k_idxs)
        top_k_scores.append(k_scores)
        top_k_methods.append(["embedding"] * len(k_idxs))

    embedding_labels = np.array(embedding_labels_list, dtype=object)
    embedding_scores = np.array(embedding_scores_list, dtype=np.float32)

    return CategorizationResult(
        final_labels=final_labels,
        final_indices=final_indices,
        final_scores=final_scores,
        methods=methods,
        embedding_labels=embedding_labels,
        embedding_scores=embedding_scores,
        label_scores=score_matrix,
        labels=labels,
        top_k_indices=top_k_indices,
        top_k_scores=top_k_scores,
        top_k_methods=top_k_methods,
    )
