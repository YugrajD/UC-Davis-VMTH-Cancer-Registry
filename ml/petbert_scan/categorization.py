from dataclasses import dataclass

import numpy as np

from .constants import DEFAULT_LABELS, LABEL_ANCHORS
from .embedding import cosine_similarity_matrix

try:
    from model.classifier import CANCER_LABELS, VetBERTClassifier
except Exception:
    CANCER_LABELS = None
    VetBERTClassifier = None


@dataclass(frozen=True)
class CategorizationResult:
    final_labels: list[str]
    final_scores: list[float]
    methods: list[str]
    keyword_labels: list[str]
    keyword_scores: list[float]
    embedding_labels: np.ndarray
    embedding_scores: np.ndarray
    label_scores: np.ndarray
    labels: list[str]


def build_label_texts(labels: list[str]) -> list[str]:
    return [f"Veterinary diagnosis of {label.lower()} in a pathology or clinical report." for label in labels]


def resolve_labels() -> list[str]:
    return list(CANCER_LABELS) if CANCER_LABELS else DEFAULT_LABELS


def categorize_embeddings(
    text_embeddings: np.ndarray, label_embeddings: np.ndarray, labels: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sims = cosine_similarity_matrix(text_embeddings, label_embeddings)
    pred_idx = np.argmax(sims, axis=1)
    pred_scores = sims[np.arange(len(pred_idx)), pred_idx].astype(np.float32, copy=False)
    pred_labels = np.array([labels[i] for i in pred_idx], dtype=object)
    return pred_labels, pred_scores, sims


def run_hybrid_categorization(
    *,
    texts: list[str],
    text_embeddings: np.ndarray,
    label_embeddings: np.ndarray,
    labels: list[str],
    keyword_min_conf: float,
    embedding_min_sim: float,
) -> CategorizationResult:
    embedding_labels, embedding_scores, label_scores = categorize_embeddings(
        text_embeddings, label_embeddings, labels
    )
    embedding_scores = embedding_scores.astype(np.float32, copy=False)

    keyword_classifier = VetBERTClassifier() if VetBERTClassifier is not None else None
    final_labels: list[str] = []
    final_scores: list[float] = []
    methods: list[str] = []
    keyword_labels: list[str] = []
    keyword_scores: list[float] = []

    for idx, text in enumerate(texts):
        if not text:
            final_labels.append("Uncategorized")
            final_scores.append(0.0)
            methods.append("empty")
            keyword_labels.append("")
            keyword_scores.append(0.0)
            continue

        if keyword_classifier is None:
            kw_label = ""
            kw_score = 0.0
        else:
            keyword_result = keyword_classifier.predict(text)
            kw_label = keyword_result["predicted_label"]
            kw_score = float(keyword_result["confidence"])

        keyword_labels.append(kw_label)
        keyword_scores.append(kw_score)

        anchor_pattern = LABEL_ANCHORS.get(kw_label)
        has_anchor = bool(anchor_pattern.search(text)) if anchor_pattern else False
        if kw_score >= keyword_min_conf and has_anchor:
            final_labels.append(kw_label)
            final_scores.append(kw_score)
            methods.append("keyword")
            continue

        if float(embedding_scores[idx]) >= embedding_min_sim:
            final_labels.append(str(embedding_labels[idx]))
            final_scores.append(float(embedding_scores[idx]))
            methods.append("embedding")
        else:
            final_labels.append("Uncategorized")
            final_scores.append(float(max(kw_score, embedding_scores[idx])))
            methods.append("low_confidence")

    return CategorizationResult(
        final_labels=final_labels,
        final_scores=final_scores,
        methods=methods,
        keyword_labels=keyword_labels,
        keyword_scores=keyword_scores,
        embedding_labels=embedding_labels,
        embedding_scores=embedding_scores,
        label_scores=label_scores,
        labels=labels,
    )

