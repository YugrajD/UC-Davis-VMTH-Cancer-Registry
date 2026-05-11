"""Shared dataclasses and type aliases used by the scan pipeline."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

import config as _config


TaskMode = Literal["categorize", "neighbors", "both"]


@dataclass(frozen=True)
class ScanConfig:
    csv_path: str
    id_col: str
    text_cols: tuple[str, ...]  # columns to embed independently; empty = TF-IDF selection
    model_name: str
    local_only: bool
    out_dir: str
    max_rows: int | None
    batch_size: int
    max_length: int
    neighbors_k: int
    task: TaskMode
    embedding_min_sim: float
    device: str
    labels_csv_path: str
    tfidf_vectorizer_path: str = _config.TFIDF_VECTORIZER_PATH
    embedding_cache_path: str | None = None
    group_classifier_path: str | None = None
    group_classifier_threshold: float = 0.3
    group_classifier_fallback_to_argmax: bool = True
    case_presence_classifier_path: str | None = None
    case_presence_threshold: float = 0.5
    label_presence_classifier_dir: str | None = None
    label_presence_threshold: float = 0.5
    label_presence_thresholds_json: str | None = None
    tail_max_predictions: int = 2
    tail_max_group_prob_gap: float = 0.08
    uncommon_groups_path: str = _config.UNCOMMON_GROUPS_TXT


@dataclass(frozen=True)
class ScanOutputs:
    predictions_csv: str
    provenance_csv: str
    similarity_csv: str
    visualization_csv: str
    neighbors_csv: str | None
    npz: str
    summary_json: str


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
