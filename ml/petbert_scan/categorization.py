"""Embedding-based categorization core and result container.

This module is responsible for the central decision logic:
  1. Compute cosine similarity between each diagnosis embedding and every
     taxonomy label embedding.
  2. For each diagnosis, pick the highest-scoring label.
  3. Apply a confidence threshold -- if the best score is too low, mark the
     row as "Uncategorized" rather than guessing.
"""

from dataclasses import dataclass

import numpy as np

from .embedding import cosine_similarity_matrix


@dataclass(frozen=True)
class CategorizationResult:
    """Container for all per-row categorization outputs.

    Attributes:
        final_labels:     The chosen taxonomy term per row (or "Uncategorized" / "").
        final_indices:    Index into the taxonomy list for the chosen label (-1 if empty).
        final_scores:     Cosine similarity score of the chosen label.
        methods:          How each row was classified: "embedding", "low_confidence", or "empty".
        keyword_labels:   Reserved for future keyword-based classification (always empty).
        keyword_scores:   Reserved for future keyword scores (always 0.0).
        embedding_labels: The raw top-1 label from cosine similarity (before thresholding).
        embedding_scores: The raw top-1 cosine score (before thresholding).
        label_scores:     Full (num_texts, num_labels) similarity matrix.
        labels:           List of all taxonomy term strings.
    """
    final_labels: list[str]
    final_indices: list[int]
    final_scores: list[float]
    methods: list[str]
    keyword_labels: list[str]
    keyword_scores: list[float]
    embedding_labels: np.ndarray
    embedding_scores: np.ndarray
    label_scores: np.ndarray
    labels: list[str]


def categorize_embeddings(
    text_embeddings: np.ndarray, label_embeddings: np.ndarray, labels: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute cosine similarities and pick the best label for each text.

    Returns:
        pred_labels:  (N,) array of label name strings (the top-1 match).
        pred_scores:  (N,) array of similarity scores for the top-1 match.
        pred_idx:     (N,) array of integer indices into ``labels``.
        sims:         (N, M) full similarity matrix (texts x labels).
    """
    # sims[i, j] = cosine_similarity(text_embeddings[i], label_embeddings[j])
    sims = cosine_similarity_matrix(text_embeddings, label_embeddings)

    # For each text, find the label with the highest cosine similarity.
    pred_idx = np.argmax(sims, axis=1)
    pred_scores = sims[np.arange(len(pred_idx)), pred_idx].astype(np.float32, copy=False)
    pred_labels = np.array([labels[i] for i in pred_idx], dtype=object)
    return pred_labels, pred_scores, pred_idx, sims


def run_hybrid_categorization(
    *,
    texts: list[str],
    text_embeddings: np.ndarray,
    label_embeddings: np.ndarray,
    labels: list[str],
    embedding_min_sim: float,
) -> CategorizationResult:
    """Categorize each diagnosis text against the taxonomy using cosine similarity.

    For each row:
      - If the text is empty -> method="empty", label="".
      - If best cosine score >= embedding_min_sim (default 0.6) -> method="embedding",
        use the best-matching taxonomy term.
      - If best cosine score < threshold -> method="low_confidence",
        label="Uncategorized" (but the closest label index is still recorded).

    The function is named "hybrid" because it was designed to support both
    keyword-based and embedding-based approaches. Currently only embedding-based
    categorization is active.
    """
    embedding_labels, embedding_scores, embedding_idx, label_scores = categorize_embeddings(
        text_embeddings, label_embeddings, labels
    )
    embedding_scores = embedding_scores.astype(np.float32, copy=False)

    final_labels: list[str] = []
    final_indices: list[int] = []
    final_scores: list[float] = []
    methods: list[str] = []
    keyword_labels: list[str] = []
    keyword_scores: list[float] = []

    for idx, text in enumerate(texts):
        # Empty diagnosis text -- nothing to classify.
        if not text:
            final_labels.append("")
            final_indices.append(-1)
            final_scores.append(0.0)
            methods.append("empty")
            keyword_labels.append("")
            keyword_scores.append(0.0)
            continue

        # Keyword path is unused; always empty.
        keyword_labels.append("")
        keyword_scores.append(0.0)

        # Apply the confidence threshold to decide whether to accept the
        # embedding-based prediction or mark it as low-confidence.
        if float(embedding_scores[idx]) >= embedding_min_sim:
            final_labels.append(str(embedding_labels[idx]))
            final_indices.append(int(embedding_idx[idx]))
            final_scores.append(float(embedding_scores[idx]))
            methods.append("embedding")
        else:
            final_labels.append("Uncategorized")
            final_indices.append(int(embedding_idx[idx]))
            final_scores.append(float(embedding_scores[idx]))
            methods.append("low_confidence")

    return CategorizationResult(
        final_labels=final_labels,
        final_indices=final_indices,
        final_scores=final_scores,
        methods=methods,
        keyword_labels=keyword_labels,
        keyword_scores=keyword_scores,
        embedding_labels=embedding_labels,
        embedding_scores=embedding_scores,
        label_scores=label_scores,
        labels=labels,
    )
