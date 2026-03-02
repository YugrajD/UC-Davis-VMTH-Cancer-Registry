"""Shared dataclasses and type aliases used by the scan pipeline."""

from dataclasses import dataclass
from typing import Literal


TaskMode = Literal["categorize", "neighbors", "both"]


@dataclass(frozen=True)
class ScanConfig:
    csv_path: str
    id_col: str
    text_cols: tuple[str, ...]  # columns to embed independently; best-match label wins
    col_weights: dict[str, float]  # reserved; no longer used for scoring
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
    presence_classifier_path: str | None


@dataclass(frozen=True)
class ScanOutputs:
    predictions_csv: str
    provenance_csv: str
    similarity_csv: str
    visualization_csv: str
    column_scores_csv: str
    neighbors_csv: str | None
    npz: str
    summary_json: str
