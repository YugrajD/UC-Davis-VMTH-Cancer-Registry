# Training Guide

Step-by-step instructions for cold-start training and retraining cycles. For architecture rationale see [model-training.md](model-training.md); for the head-by-head reference see [classifiers.md](classifiers.md).

## Prerequisites

- **Python venv:** `ml/.venv/Scripts/python.exe` (Windows) / `ml/.venv/bin/python3` (macOS/Linux). Every script in `ml/scripts/` adds `ml/` to `sys.path`, so no `PYTHONPATH` is needed.
- **Device:** `--device cuda` recommended (PyTorch 2.6+ on CUDA 12.8 supports Blackwell sm_120). `--device auto` picks the best available (cuda → mps → xpu → cpu).
- **Annotations:** `ml/output/annotation/annotation.csv` must exist — this is the canonical training-supervision file (`config.ANNOTATION_CSV`, the default for every training and evaluation script). If missing, run `python ml/scripts/run_annotation.py` to generate `ml/output/annotation/llm_annotation_cleaned.csv`, then promote it (`Copy-Item ml/output/annotation/llm_annotation_cleaned.csv ml/output/annotation/annotation.csv`). Training scripts error out if the canonical file is missing.
- **Reports:** `ml/data/report.csv` must exist.
- **Train/test split:** `ml/output/splits/train_cases.txt` and `test_cases.txt` must exist. Generate once with `ml/.venv/Scripts/python.exe ml/training/data/create_split.py`. Do NOT regenerate between training runs — a new seed invalidates comparisons.

## Cold-start protocol

A cold start is required whenever embeddings change (backbone retraining or a change in the section grouping fed to PetBERT). Run the steps in this order; each step's outputs feed the next.

### Step 1 — Annotate (skip if `annotation.csv` already exists)

```bash
ml/.venv/Scripts/python.exe ml/scripts/run_annotation.py
```
Produces `ml/output/annotation/llm_annotation_cleaned.csv` (cleanup pass included by default). Promote it to the canonical training-supervision path that every training script reads:
```bash
Copy-Item ml/output/annotation/llm_annotation_cleaned.csv ml/output/annotation/annotation.csv
```
`run_training.py` checks for `annotation.csv` at startup and errors out (with a hint) if it isn't there. Runtime: 30–60 minutes plus cleanup on a full corpus.

### Step 2 — Adapt the PetBERT backbone

```bash
ml/.venv/Scripts/python.exe ml/scripts/run_training.py \
  --mode adapt-backbone \
  --epochs 3 --batch-size 32 --lr 2e-5 --temperature 0.07 \
  --device cuda --local-only
```
Builds per-section `(report_text, label_text)` pairs from `llm_annotation.csv` + `report.csv`, then InfoNCE-fine-tunes PetBERT. Saves the full HuggingFace checkpoint to `ml/output/checkpoints/contrastive/`. Use `--skip-pair-build` to reuse an existing pairs CSV. Runtime: ~30 min on cuda.

### Step 3 — Invalidate the embedding cache

The backbone's embedding space changed, so the cache is stale:
```bash
rm -f ml/output/training/embedding_cache.npz
```

### Step 4 — Rebuild the embedding cache

```bash
ml/.venv/Scripts/python.exe ml/scripts/run_production.py --embed-only --device cuda
```
`--embed-only` runs the production pipeline up to and including embedding, then stops. Populates `ml/output/training/embedding_cache.npz` with `concat_3`, the three per-section views, the masked-mean, and label embeddings. Runtime: ~25 min on cuda. Subsequent training and inference reuse this cache.

### Step 5 — Train the CasePresenceClassifier (Stage 1 gate)

```bash
ml/.venv/Scripts/python.exe ml/scripts/run_training.py \
  --mode train-case-presence \
  --epochs 20 --case-presence-recall-weight 0.7 \
  --device cuda --local-only \
  --train-cases ml/output/splits/train_cases.txt
```
Output: `ml/output/checkpoints/case_presence/case_presence_classifier.pt`. Runtime: a few minutes.

### Step 6 — Train the GroupClassifier (Stage 2)

```bash
ml/.venv/Scripts/python.exe ml/scripts/run_training.py \
  --mode train-groups \
  --epochs 300 --lr 5e-5 --dropout 0.1 \
  --max-class-weight 50 --weight-decay 1e-3 \
  --device cuda --local-only \
  --train-cases ml/output/splits/train_cases.txt
```
Critical hyperparameters (all required):
- `--max-class-weight 50` — caps per-group BCE `pos_weight` (rare-group weights would otherwise reach >3000×).
- `--weight-decay 1e-3` — prevents the degenerate "predict every group on every case" solution.
- `--dropout 0.1` — current production setting; 0.3 over-regularises.
- `--epochs 300` — best epoch typically lands in the 200–280 range.

Output: `ml/output/checkpoints/group/group_classifier_best.pt` (only overwritten when val macro F1 beats the previous best). `group_classifier_current.pt` is overwritten each run. Runtime: ~15 min on cuda.

### Step 7 — Train the per-group LabelPresenceClassifiers (Stage 3a)

```bash
ml/.venv/Scripts/python.exe ml/scripts/run_training.py \
  --mode train-label-presence \
  --label-presence-epochs 25 --label-presence-negs-per-pos 5 \
  --label-presence-recall-weight 0.5 \
  --label-presence-n-cols 3 --label-presence-col-pair-mode --label-presence-col-combine learned \
  --device cuda --local-only \
  --train-cases ml/output/splits/train_cases.txt
```
Defaults for the `--label-presence-*` flags match production. Trains one model per common group plus the Uncommon head. Outputs `ml/output/checkpoints/label_presence/{safe_group_name}.pt`. Runtime: ~10–20 min total on cuda.

### Step 8 — Calibrate per-LP thresholds

First produce per-(case, label) scores on the test split:
```bash
ml/.venv/Scripts/python.exe ml/scripts/run_evaluation.py \
  --stage label-presence \
  --test-cases ml/output/splits/test_cases.txt \
  --out-dir ml/output/evaluation/label_presence \
  --label "lp eval (baseline t=0.5)"
```

Then sweep per-LP thresholds on the sweep half; eval half stays unbiased:
```bash
ml/.venv/Scripts/python.exe ml/scripts/sweep_lp_thresholds.py \
  --eval-csv ml/output/evaluation/label_presence/label_presence_evaluation.csv \
  --baseline-threshold 0.5 --grid 0.05,0.95,0.01 \
  --out-json ml/output/checkpoints/label_presence/lp_thresholds.json
```
`run_production.py` auto-loads the resulting JSON next run.

### Step 9 — Recalibrate the Stage-2 tail gate (optional)

```bash
ml/.venv/Scripts/python.exe ml/scripts/sweep_tail_gate.py
```
Runs production + evaluation for a small grid of `(K, gap)` pairs against `test_cases.txt`. The current production defaults (`K=2, gap=0.08`) were calibrated 2026-05-11; rerun this if the GroupClassifier has changed.

### Step 10 — Score the held-out test set

```bash
ml/.venv/Scripts/python.exe ml/scripts/run_evaluation.py \
  --test-cases ml/output/splits/test_cases.txt \
  --out-dir ml/output/evaluation/contrastive_test \
  --label "cold-start cycle"
```
Add `--stage all` to also write per-stage metrics for Stage 1/2/3 plus case-based and common-labels evaluations:
```bash
ml/.venv/Scripts/python.exe ml/scripts/run_evaluation.py \
  --stage all \
  --test-cases ml/output/splits/test_cases.txt \
  --out-dir ml/output/evaluation/contrastive_test \
  --label "cold-start cycle"
```

## Embedding & classifier versioning

From `CLAUDE.md`: embeddings change when (a) the text fed to PetBERT changes (not applicable currently — concat-3 is the only path) or (b) PetBERT weights change (i.e. a backbone retrain). When embeddings change, every downstream classifier (Group, CasePresence, LabelPresence) is invalidated. Stale classifiers load silently and produce wrong results — no error.

Before retraining the backbone:
1. Move the full old generation (embeddings + backbone + all classifiers) to `ml/output/archive/YYYY-MM-DD_<short-description>/` (sibling of `checkpoints/`, never referenced by `config.py`).
2. Delete originals from production paths.
3. Run the cold-start protocol above.

## Retraining a single head

When only one head changes (e.g. you retrained GroupClassifier with a new hyperparameter), the cache and backbone are still valid — skip Steps 1–4 and rerun only the affected step plus the calibration steps that depend on it. The LP thresholds depend on which LPs exist, and the tail-gate calibration depends on the GroupClassifier — recalibrate them if the corresponding head changed.

## Troubleshooting

**Windows torch DLL load order.** `production/petbert_pipeline/pipeline.py` imports `torch` at module top (with `# noqa: F401`) before `pandas` / `numpy` / `sklearn`. Originally needed because the XPU torch wheel's c10.dll search path conflicted with sklearn-loaded MKL DLLs. With CUDA wheels on Windows a similar collision is possible; leave the early import in place.

**Embedding cache invalidation.** `production/petbert_pipeline/embedding_cache.py` validates the cache against the model-name string, report-CSV mtime, labels-CSV mtime, and the expected section column names. On any mismatch the cache is treated as invalid and a fresh embed pass runs. If you intentionally renamed the model directory, expect the next run to rebuild the cache (~25 min on cuda).

**`emb_dim` mismatches.** GroupClassifier and CasePresenceClassifier serialize `emb_dim` in their checkpoints and validate against the cache shape at load time. A 768-dim head against a 2304-dim cache (or vice-versa) is a sign the backbone or embedding pipeline changed without retraining — start a fresh cold-start cycle.

**Annotation file missing.** `run_training.py` errors out and asks you to run `python ml/scripts/run_annotation.py` first. There is no auto-keyword fallback.

**GroupClassifier diverges to all-1s.** You forgot `--weight-decay 1e-3` or `--max-class-weight 50`. Both are required guards.
