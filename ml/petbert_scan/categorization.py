"""Embedding-based categorization: cosine similarity against taxonomy label embeddings.

For each diagnosis:
  - Pick the top-1 label by cosine similarity.
  - If the best score is below the threshold, mark as "low_confidence".
  - If the text was empty, mark as "empty".
"""

from dataclasses import dataclass

import numpy as np

from .embedding import cosine_similarity_matrix


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
