# Step 5 — Create ml/scripts/

## `scripts/run_pipeline.py`

Entry point for production inference. Sets `sys.path` internally so no `env PYTHONPATH=ml` is needed.

```
python ml/scripts/run_pipeline.py
python ml/scripts/run_pipeline.py --max-rows 100 --local-only
python ml/scripts/run_pipeline.py --presence-classifier ml/model/checkpoints/presence_classifier_best.pt
```

Defaults:
- `--embedding-cache ml/data/embedding_cache.npz`
- `--presence-classifier ml/model/checkpoints/presence_classifier_best.pt` (if it exists)
- `--local-only True`

All other flags pass through to `petbert_pipeline` CLI unchanged.

---

## `scripts/run_training.py`

End-to-end training pipeline. Sets `sys.path` internally.

```
python ml/scripts/run_training.py --mode binary --label "v12"
python ml/scripts/run_training.py --mode group  --epochs 50
python ml/scripts/run_training.py --skip-keyword-scan --mode binary
```

Flow:
1. (unless `--skip-keyword-scan`) Run `keyword_pipeline` → `ml/output/diagnoses/keyword_predictions.csv`
2. If `--mode binary`: Run `training/binary/run_cycle.py` (full binary training cycle)
3. If `--mode group`: Run `training/group/build_training_data.py` → `training/group/train.py`
