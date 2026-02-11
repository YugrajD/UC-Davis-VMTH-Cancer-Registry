from dataclasses import dataclass
from typing import Literal


TaskMode = Literal["categorize", "neighbors", "both"]


@dataclass(frozen=True)
class ScanConfig:
    csv_path: str
    id_col: str
    text_col: str
    model_name: str
    local_only: bool
    out_dir: str
    max_rows: int | None
    batch_size: int
    max_length: int
    neighbors_k: int
    task: TaskMode
    keyword_min_conf: float
    embedding_min_sim: float
    device: str


@dataclass(frozen=True)
class ScanOutputs:
    rows_csv: str
    categories_csv: str
    neighbors_csv: str | None
    npz: str
    summary_json: str

