# Step 6 — Dry-Run Verification

**Rule**: Do not overwrite classifier checkpoints. Only the production pipeline output directory is temporarily modified (and restored after verification).

---

## 6a. Help / import checks

```bash
ml/.venv/bin/python3 -m petbert_pipeline --help
ml/.venv/bin/python3 -m keyword_pipeline --help
ml/.venv/bin/python3 ml/scripts/run_pipeline.py --help
ml/.venv/bin/python3 ml/scripts/run_training.py --help
```

## 6b. Training import verification (no training run)

```bash
env PYTHONPATH=ml ml/.venv/bin/python3 -c "
from training.binary.build_training_pairs import build_pairs
from training.binary.train import train
from training.binary.evaluate import evaluate
from training.binary.update_co_bank import update_co_bank
from training.binary.log_evaluation import log_evaluation
from training.binary.run_cycle import main as run_cycle
from training.group.build_training_data import build_training_data
from training.group.train import train as train_group
print('All training imports OK')
"
```

## 6c. Production pipeline dry run (100 rows, embedding cache, no PetBERT call)

1. Back up `ml/output/report/`
2. Run: `ml/.venv/bin/python3 ml/scripts/run_pipeline.py --max-rows 100 --local-only`
3. Verify `ml/output/report/petbert_predictions.csv` exists and has ~100 rows
4. Restore backup
