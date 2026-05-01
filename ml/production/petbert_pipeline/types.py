"""Shared dataclasses and type aliases used by the scan pipeline."""

from dataclasses import dataclass
from typing import Literal

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
    presence_classifier_path: str | None  # used by training cycle (run_cycle.py); None in production
    tfidf_vectorizer_path: str = _config.TFIDF_VECTORIZER_PATH
    embedding_cache_path: str | None = None
    group_classifier_path: str | None = None
    group_classifier_threshold: float = 0.3
    case_presence_classifier_path: str | None = None
    case_presence_threshold: float = 0.5


@dataclass(frozen=True)
class ScanOutputs:
    predictions_csv: str
    provenance_csv: str
    similarity_csv: str
    visualization_csv: str
    neighbors_csv: str | None
    npz: str
    summary_json: str
