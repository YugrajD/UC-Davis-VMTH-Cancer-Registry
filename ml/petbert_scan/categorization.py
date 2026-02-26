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
    final_labels: list[str]       # chosen taxonomy term, "Uncategorized", or ""
    final_indices: list[int]      # index into labels (-1 if empty)
    final_scores: list[float]     # cosine similarity of chosen label
    methods: list[str]            # "embedding", "low_confidence", or "empty"
    embedding_labels: np.ndarray  # top-1 label before thresholding (N,)
    embedding_scores: np.ndarray  # top-1 score before thresholding (N,)
    label_scores: np.ndarray      # full similarity matrix (N, M)
    labels: list[str]             # all taxonomy term strings


def run_categorization(
    *,
    texts: list[str],
    text_embeddings: np.ndarray | list[np.ndarray],
    label_embeddings: np.ndarray,
    labels: list[str],
    embedding_min_sim: float,
    col_has_content: list[np.ndarray] | None = None,
) -> CategorizationResult:
    """Categorize each diagnosis by cosine similarity to taxonomy label embeddings.

    When ``text_embeddings`` is a list of per-column embedding arrays, each
    column independently computes its similarity scores and the label with the
    highest score across *any* column wins.  Empty cells (tracked via
    ``col_has_content``) are masked out so they cannot influence the result.
    """
    if isinstance(text_embeddings, list):
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
    top_idx = np.argmax(sims, axis=1)
    top_scores = sims[np.arange(len(top_idx)), top_idx].astype(np.float32, copy=False)
    top_labels = np.array([labels[i] for i in top_idx], dtype=object)

    final_labels: list[str] = []
    final_indices: list[int] = []
    final_scores: list[float] = []
    methods: list[str] = []

    for i, text in enumerate(texts):
        if not text:
            final_labels.append("")
            final_indices.append(-1)
            final_scores.append(0.0)
            methods.append("empty")
        elif float(top_scores[i]) >= embedding_min_sim:
            final_labels.append(str(top_labels[i]))
            final_indices.append(int(top_idx[i]))
            final_scores.append(float(top_scores[i]))
            methods.append("embedding")
        else:
            final_labels.append("Uncategorized")
            final_indices.append(int(top_idx[i]))
            final_scores.append(float(top_scores[i]))
            methods.append("low_confidence")

    return CategorizationResult(
        final_labels=final_labels,
        final_indices=final_indices,
        final_scores=final_scores,
        methods=methods,
        embedding_labels=top_labels,
        embedding_scores=top_scores,
        label_scores=sims,
        labels=labels,
    )
