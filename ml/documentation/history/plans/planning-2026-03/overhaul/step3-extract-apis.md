# Step 3 — Extract API Functions

Four training scripts currently expose only `main()` with argparse.
Each needs a typed API function so orchestrators can call them directly.

Already have clean APIs (no changes needed):
- `binary/evaluate.py`: `evaluate(petbert_csv, keyword_csv, out_dir)`
- `group/build_training_data.py`: `build_training_data(cache_path, keyword_csv_path, out_path)`
- `group/train.py`: `train(training_data_path, out_path, epochs, ...)`

---

## `training/binary/build_training_pairs.py`

Add:
```python
def build_pairs(
    *,
    report_csv: str = "ml/data/report.csv",
    keyword_csv: str = "ml/output/diagnoses/keyword_predictions.csv",
    evaluation_csv: str = "ml/output/evaluation/evaluation.csv",
    labels_csv: str = "ml/labels/labels.csv",
    out: str = "ml/data/training_pairs.csv",
    easy_neg_per_pos: int = 3,
    fp_neg_per_case: int = 10,
    co_neg_per_case: int = 3,
    co_neg_extra_csv: str = "",
    co_neg_bank_csv: str = "",
    seed: int = 42,
    max_pos_per_group: int = 0,
) -> None:
    ...  # current body of main() after argparse
```
`main()` becomes: `parse args → build_pairs(...)`.

---

## `training/binary/train.py`

Add:
```python
def train(
    *,
    pairs_csv: str = "ml/data/training_pairs.csv",
    embedding_cache: str | None = None,
    report_csv: str = "ml/data/report.csv",
    labels_csv: str = "ml/labels/labels.csv",
    out_dir: str = "ml/model/checkpoints",
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    hidden_dim: int = 256,
    dropout: float = 0.3,
    val_split: float = 0.15,
    device: str = "auto",
    seed: int = 42,
    pos_weight: float = 1.0,
    recall_weight: float = 0.5,
) -> int:
    ...  # current body of main() after argparse
```

---

## `training/binary/update_co_bank.py`

Add:
```python
def update_co_bank(
    evaluation_csv: str = "ml/output/evaluation/evaluation.csv",
    bank_csv: str = "ml/output/evaluation/evaluation_co_bank.csv",
) -> int:
    ...  # current body of main() after argparse
```

---

## `training/binary/log_evaluation.py`

Add:
```python
def log_evaluation(
    summary: str = "ml/output/evaluation/evaluation_summary.csv",
    history: str = "ml/output/evaluation/evaluation_history.csv",
    label: str = "",
) -> int:
    ...  # current body of main() after argparse
```
