# Step 4 — Update Imports + Refactor run_cycle.py

## `training/binary/run_cycle.py`

**Import changes:**
```python
# Before                                              After
from petbert_scan.pipeline import run_scan         → from petbert_pipeline.pipeline import run_scan
from petbert_scan.types import ScanConfig          → from petbert_pipeline.types import ScanConfig
from training.build_training_pairs import main … → from training.binary.build_training_pairs import build_pairs
from training.evaluate_predictions import evaluate → from training.binary.evaluate import evaluate
from training.log_evaluation import main …        → from training.binary.log_evaluation import log_evaluation
from training.train_classifier import main …      → from training.binary.train import train
from training.update_co_bank import main …        → from training.binary.update_co_bank import update_co_bank
```

**Remove `_call()` helper** — replace each call with direct API:
```python
# Before
_call("Step 1/5 — Build training pairs", _build_pairs_main, build_argv)

# After
_print_banner("Step 1/5 — Build training pairs")
build_pairs(
    co_neg_per_case=args.co_neg_per_case,
    fp_neg_per_case=args.fp_neg_per_case,
    max_pos_per_group=args.max_pos_per_group,
    co_neg_extra_csv=args.co_neg_extra_csv,
    co_neg_bank_csv=args.co_neg_bank_csv,
)
```
Same pattern for `train(...)`, `update_co_bank(...)`, `log_evaluation(...)`.

**Update hardcoded output path:**
```python
# Before
petbert_csv=Path("ml/output/report/petbert_scan_predictions.csv")
# After
petbert_csv=Path("ml/output/report/petbert_predictions.csv")
```

---

## `training/binary/build_training_pairs.py`

```python
from petbert_scan.utils import clean_text, merge_report_columns
# →
from petbert_pipeline.utils import clean_text, merge_report_columns
```

---

## `training/binary/train.py`

```python
from petbert_scan.utils import device_from_arg       → from petbert_pipeline.utils import device_from_arg
from petbert_scan.embedding_cache import load_cache  → from petbert_pipeline.embedding_cache import load_cache
# Update error message: "python -m petbert_scan" → "python -m petbert_pipeline"
```

---

## `training/group/build_training_data.py`

Update error message strings only (no import changes):
- `"petbert_scan"` → `"petbert_pipeline"`
- `"keyword_scan"` → `"keyword_pipeline"`

---

## `training/group/train.py`

- Update any error message strings referencing old package names
- Verify `import sys` is present (file calls `sys.exit()` in `train()`)
