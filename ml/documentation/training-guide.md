# Training Guide

How to run each training approach, from prerequisites through to a trained checkpoint.
For architectural details, approach comparisons, and pros/cons see [model-training.md](model-training.md).

> **Prerequisites for all approaches:**
> - `ml/output/annotation/keyword/keyword_annotation.csv` must exist (run the keyword pipeline first if not)
> - `ml/data/report.csv` must exist
> - Use `ml/.venv/Scripts/python.exe` (Windows) or `ml/.venv/bin/python3` (macOS/Linux)

---

## Binary PresenceClassifier (iterative)

The recommended approach. Trains an MLP on cached PetBERT embeddings using a rolling
bank of hard negatives. Each cycle takes ~10 minutes; run 6–7 cycles to reach plateau.

### First run (cold start)

A cold start is required after any architecture change, new keyword data, or if no
embedding cache exists. Deletes the stale cache, bank, and checkpoint:

```bash
rm -f ml/data/embedding_cache.npz
rm -f ml/output/training/binary/evaluation_co_bank.csv
rm -f ml/model/checkpoints/presence_classifier_current.pt
```

Then run c1 — Step 0 will rebuild the embedding cache automatically (takes several minutes):

```bash
ml/.venv/Scripts/python.exe ml/scripts/run_training.py \
  --mode binary \
  --label "cold-start c1" \
  --co-neg-per-case 5 \
  --fp-neg-per-case 10 \
  --embedding-min-sim 0.05 \
  --epochs 25 \
  --recall-weight 0.25 \
  --hidden-dim 512 \
  --device xpu \
  --local-only
```

### Subsequent cycles

Continue with the same command, updating `--label` each time:

```bash
ml/.venv/Scripts/python.exe ml/scripts/run_training.py \
  --mode binary \
  --label "c2" \
  --co-neg-per-case 5 \
  --fp-neg-per-case 10 \
  --embedding-min-sim 0.05 \
  --epochs 25 \
  --recall-weight 0.25 \
  --hidden-dim 512 \
  --device xpu \
  --local-only
```

Do **not** raise `--co-neg-per-case` to 10 — causes regression with the per-column architecture.

### What each cycle does

1. Build embedding cache (Step 0 — skipped if cache exists)
2. Build `training_pairs.csv` from positives + CO bank + FP negatives
3. Train `PresenceClassifier` for 25 epochs; save best checkpoint by validation F1
4. Run pipeline with new checkpoint → `petbert_predictions.csv`
5. Evaluate predictions → `evaluation.csv` (good/slightly_off/completely_off/fp/fn)
6. Update CO bank with this cycle's completely-off predictions
7. Log results to `evaluation_history.csv`; auto-promote if new best

### Expected trajectory (5,788 cancer cases)

| Cycle | Expected Good+Slight | Notes |
|-------|---------------------|-------|
| c1 (cold start) | ~28–30% | Cache rebuilt; bank ~15k |
| c2 | ~26% | May dip — continue |
| c3–c4 | ~38–39% | Large jump |
| c5–c6 | ~39–40% | Plateau |
| c7+ | may regress | Stop here |

### Key parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `--co-neg-per-case` | `5` | Do NOT raise to 10 |
| `--fp-neg-per-case` | `10` | Keep at 10 |
| `--recall-weight` | `0.25` | Prevents degenerate checkpoints winning |
| `--epochs` | `25` | Beyond 25 shows diminishing returns |
| `--embedding-min-sim` | `0.05` | Scores are mean-subtracted — not raw cosine |

---

## GroupClassifier (one-shot)

Trains a multi-label MLP that predicts cancer group(s) per report. One-shot — run once,
no iterative cycles. Re-run whenever keyword coverage improves.

```bash
ml/.venv/Scripts/python.exe ml/scripts/run_training.py --mode group --device xpu
```

This builds training data from the embedding cache and `keyword_annotation.csv`, trains
for the configured number of epochs, and saves to `ml/model/checkpoints/group_classifier_best.pt`.

> **Note:** GroupClassifier is not yet competitive at 5,788 cases (21.9% vs binary 41.9%).
> Re-train when keyword coverage reaches ~10,000 confirmed cases.

### Options

```bash
ml/.venv/Scripts/python.exe ml/scripts/run_training.py \
  --mode group \
  --device xpu \
  --epochs 50 \
  --lr 1e-3 \
  --threshold 0.3
```

---

## Fine-tuned PetBERT (WIP)

Fine-tunes PetBERT end-to-end as a group classifier. Not yet benchmarked — resolve the
known code issues in `petbert-pipeline.md` before running.

### Step 1 — Build dataset

```bash
ml/.venv/Scripts/python.exe ml/training/finetune/build_dataset.py \
  --reports-csv ml/data/report.csv \
  --predictions-csv ml/output/annotation/keyword/keyword_annotation.csv \
  --labels-csv ml/labels/labels.csv \
  --out-dir ml/data/finetune_dataset
```

Reads report text and keyword labels, tokenizes with the PetBERT tokenizer, saves a
HuggingFace `DatasetDict` with computed class weights to `ml/data/finetune_dataset/`.

### Step 2 — Fine-tune

```bash
ml/.venv/Scripts/python.exe ml/training/finetune/train.py \
  --dataset ml/data/finetune_dataset \
  --out-dir ml/model/checkpoints/petbert_finetuned \
  --epochs 5 \
  --batch-size 16 \
  --lr 2e-5
```

Saves the full model + tokenizer to `ml/model/checkpoints/petbert_finetuned/` in
HuggingFace format (loadable with `--finetuned-model-path` in the production pipeline).

### Step 3 — Run pipeline

```bash
ml/.venv/Scripts/python.exe ml/scripts/run_production.py \
  --finetuned-model-path ml/model/checkpoints/petbert_finetuned \
  --local-only
```

---

## Running the Production Pipeline

After training, run inference with the best checkpoint:

```bash
# With binary PresenceClassifier (current best)
ml/.venv/Scripts/python.exe ml/scripts/run_production.py \
  --presence-classifier ml/model/checkpoints/presence_classifier_best.pt \
  --embedding-cache ml/data/embedding_cache.npz \
  --embedding-min-sim 0.05 \
  --local-only

# With GroupClassifier
ml/.venv/Scripts/python.exe ml/scripts/run_production.py \
  --group-classifier ml/model/checkpoints/group_classifier_best.pt \
  --embedding-cache ml/data/embedding_cache.npz \
  --embedding-min-sim 0.05 \
  --local-only
```

---

## What Triggers a Cold Start

| Change | Cache valid? | Bank valid? | Checkpoint valid? |
|--------|-------------|-------------|-------------------|
| New keyword data | No | No | No — full cold start |
| Architecture change (e.g. `n_cols`) | No | No | No — full cold start |
| Hyperparameter change (`--hidden-dim`, `--epochs`) | Yes | Yes | No — retrain only |
| New training cycle (same architecture) | Yes | Yes | Overwritten each cycle |
