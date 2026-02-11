"""Embedding-based categorization core and result container."""

from dataclasses import dataclass

import numpy as np

from .embedding import cosine_similarity_matrix


@dataclass(frozen=True)
class CategorizationResult:
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
    sims = cosine_similarity_matrix(text_embeddings, label_embeddings)
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
        if not text:
            final_labels.append("Uncategorized")
            final_indices.append(-1)
            final_scores.append(0.0)
            methods.append("empty")
            keyword_labels.append("")
            keyword_scores.append(0.0)
            continue

        keyword_labels.append("")
        keyword_scores.append(0.0)

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
