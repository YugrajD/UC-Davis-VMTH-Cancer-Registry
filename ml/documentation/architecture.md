# ML Directory — Architecture & File Organization

## Overview

The `ml/` directory contains two distinct pipelines with separate roles:

| Pipeline | Purpose | When it runs |
|---|---|---|
| **`petbert_pipeline/`** | Production — embed reports, predict cancer labels | Inference + each training cycle |
| **`keyword_pipeline/`** | Ground-truth labeling — keyword matching on diagnosis text | Separately, by the domain expert |

Training is an iterative loop that uses `keyword_pipeline` output as ground truth to supervise `petbert_pipeline`.

---

## Directory Structure

```
ml/
├── data/                   Input data and generated intermediate files
├── documentation/          Design docs and training history
├── keyword_pipeline/       Ground-truth labeling pipeline (training only)
├── labels/                 Vet-ICD-O taxonomy — loading, embedding, projection
├── model/                  Neural network architectures and shared constants
│   └── checkpoints/        Saved model weights (.pt files)
├── output/                 All generated outputs (predictions, evaluation, etc.)
│   ├── diagnoses/          keyword_pipeline results (ground truth)
│   ├── evaluation/         Cycle-by-cycle evaluation results and history
│   └── report/             petbert_pipeline predictions and supporting files
├── petbert_pipeline/       Production scan pipeline (PetBERT + classifier)
├── scripts/                Top-level entry points (no PYTHONPATH needed)
│   ├── run_pipeline.py     Production inference entry point
│   └── run_training.py     Full training cycle entry point (binary or group)
├── training/               Training scripts, organized by mode
│   ├── binary/             Binary PresenceClassifier training cycle
│   ├── group/              GroupClassifier training (one-shot)
│   └── finetune/           PetBERT fine-tuning (WIP — see classifier.md)
├── requirements.txt        Pinned Python dependencies
└── .venv/                  Python virtual environment
```

---

## Packages

### `petbert_pipeline/` — Production pipeline

Invoked as `python -m petbert_pipeline`. Takes clinical report text, produces cancer predictions.

| File | Role |
|---|---|
| `types.py` | `ScanConfig` and `ScanOutputs` dataclasses — all pipeline config in one place |
| `pipeline.py` | Top-level orchestration: load → embed → categorize → write. Entry point: `run_scan(config)` |
| `embedding.py` | PetBERT loading, mean-pooled embedding, cosine similarity |
| `embedding_cache.py` | Save/load cached embeddings to `.npz` — avoids re-running PetBERT every cycle |
| `categorization.py` | Two strategies: cosine argmax (baseline) and two-stage GroupClassifier |
| `io.py` | Write all output files (CSV, NPZ, JSON) |
| `cli.py` | `argparse` CLI; `build_config()` maps args → `ScanConfig` |
| `utils.py` | Text cleaning, device selection, column merging |

### `keyword_pipeline/` — Ground-truth labeling

Keyword-based matching against a curated dictionary. Not part of production inference — used by the domain expert to generate `keyword_predictions.csv`, which becomes the training ground truth.

| File | Role |
|---|---|
| `pipeline.py` | Scan diagnosis text against keyword dictionary |
| `cli.py` | CLI entry point |

### `labels/` — Vet-ICD-O taxonomy

Shared by both pipelines. Loads and embeds the taxonomy used for prediction targets.

| File | Role |
|---|---|
| `taxonomy.py` | Parse `labels.csv` into `TaxonomyLabel(code, group, term)` records |
| `catalog.py` | Build a `LabelCatalog` (label strings + embedding texts) from `ScanConfig` |
| `projection.py` | Map predicted label indices → `(term, group, code)` output fields |
| `enrichment.py` | Blend label embeddings toward the centroid of confirmed cases (experimental) |
| `labels.csv` | Vet-ICD-O-canine-1 taxonomy: ~857 terms across 44 cancer groups |

### `model/` — Neural network architectures

| File | Role |
|---|---|
| `constants.py` | `PETBERT_EMB_DIM=768`, `DEFAULT_HIDDEN_DIM=256`, `DEFAULT_DROPOUT=0.3` |
| `presence_classifier.py` | Binary MLP: `[report_emb ‖ label_emb] → present/absent` probability |
| `group_classifier.py` | Multi-label MLP: `report_emb → per-group sigmoid probabilities` |

### `training/binary/` — Binary PresenceClassifier training loop

All scripts for the iterative training cycle. Orchestrated by `ml/scripts/run_training.py --mode binary`.

| File | Role |
|---|---|
| `run_cycle.py` | **Orchestrator** — runs the full cycle end-to-end (see flow below) |
| `build_training_pairs.py` | Assembles `training_pairs.csv` from positives, CO negatives, FP negatives |
| `train.py` | Trains `PresenceClassifier` on cached embeddings; saves checkpoint |
| `evaluate.py` | Scores petbert_pipeline predictions vs. keyword ground truth → verdicts |
| `log_evaluation.py` | Appends cycle results to `evaluation_history.csv`; prints trend table |
| `update_co_bank.py` | Maintains rolling bank of completely-off negatives across cycles |

### `training/group/` — GroupClassifier training (one-shot)

| File | Role |
|---|---|
| `build_training_data.py` | Builds multi-hot targets for `GroupClassifier` from cached embeddings |
| `train.py` | Trains `GroupClassifier` (one-shot, not iterative) |

### `training/finetune/` — PetBERT fine-tuning (WIP)

End-to-end fine-tuning of PetBERT as a sequence classifier on the group prediction task. Uses keyword pipeline output as training labels; produces a self-contained HuggingFace checkpoint used via `--finetuned-model-path` in the production pipeline. See `documentation/classifier.md` for full details and known issues.

| File | Role |
|---|---|
| `build_dataset.py` | Build HuggingFace `DatasetDict` from report text + keyword labels; compute class weights |
| `train.py` | Fine-tune `AutoModelForSequenceClassification` on top of PetBERT with weighted CrossEntropyLoss |

---

## Data Flow

```
report.csv  (clinical reports, 12,620 cases)
    │
    ▼ petbert_pipeline (Step 0 — first run only)
embedding_cache.npz  (768-dim PetBERT embeddings, cached)
    │
    ├──► [Step 1] build_training_pairs  ◄── keyword_predictions.csv (ground truth)
    │         └── training_pairs.csv       ◄── evaluation_co_bank.csv (rolling negatives)
    │
    ├──► [Step 2] train
    │         └── presence_classifier_current.pt
    │
    ├──► [Step 3] petbert_pipeline  (with trained classifier)
    │         └── output/report/petbert_predictions.csv
    │
    ├──► [Step 4] evaluate
    │         └── output/evaluation/evaluation.csv  (verdicts: good/slightly_off/off/fp/fn)
    │
    ├──► [Step 4.5] update_co_bank
    │         └── output/evaluation/evaluation_co_bank.csv  (accumulated negatives)
    │
    ├──► [Step 5] log_evaluation
    │         └── output/evaluation/evaluation_history.csv
    │
    └──► [Step 5.5] auto-promote best checkpoint
              If current cycle's Good+Slight ≥ all-time best →
              presence_classifier_current.pt → presence_classifier_best.pt
```

All steps run in-process (no subprocess spawning). Steps 0 and 3 call `run_scan(ScanConfig(...))` directly from `petbert_pipeline.pipeline`.

---

## Output Files

| Path | Contents |
|---|---|
| `output/report/petbert_predictions.csv` | Top-5 predicted labels per case (term, group, ICD code, score) |
| `output/report/petbert_column_scores.csv` | Per-column similarity breakdown — which report section drove each prediction |
| `output/report/petbert_provenance.csv` | Per-case traceability (column selected, token counts) |
| `output/report/petbert_similarity_scores.csv` | Full score matrix (N cases × M labels) — presence probabilities or cosine similarities |
| `output/report/petbert_visualization.csv` | PCA 2-D coordinates per case (for scatter plots) |
| `output/report/petbert_embeddings.npz` | Raw 768-dim report embedding vectors |
| `output/report/petbert_summary.json` | Run metadata and aggregate prediction counts |
| `output/diagnoses/keyword_predictions.csv` | Ground-truth labels from keyword_pipeline (case_id, matched_term, matched_group) |
| `output/evaluation/evaluation.csv` | Predictions + verdicts for the latest cycle |
| `output/evaluation/evaluation_summary.csv` | Aggregate counts and percentages |
| `output/evaluation/evaluation_history.csv` | One row per training cycle — tracks Good+Slight%, CO%, FP%, FN% over time |
| `output/evaluation/evaluation_co_bank.csv` | Rolling bank of completely-off negatives (accumulated across cycles) |
| `data/embedding_cache.npz` | Cached PetBERT embeddings (avoids re-running PetBERT each cycle) |
| `data/training_pairs.csv` | Generated (case, label, target) pairs for classifier training |

---

## Running

> **Note:** Use `ml/.venv/bin/python3`. Scripts in `ml/scripts/` inject `ml/` into `sys.path` automatically — no `env PYTHONPATH=ml` needed.

**Full binary training cycle (iterative):**
```bash
ml/.venv/bin/python3 ml/scripts/run_training.py \
  --mode binary \
  --label "phase 12" \
  --co-neg-per-case 10 \
  --fp-neg-per-case 10 \
  --embedding-min-sim 0.05 \
  --epochs 25 \
  --recall-weight 0.25 \
  --device mps \
  --local-only
```

**GroupClassifier training (one-shot, run after keyword coverage improves):**
```bash
ml/.venv/bin/python3 ml/scripts/run_training.py --mode group --device mps
```

**Production inference only:**
```bash
ml/.venv/bin/python3 ml/scripts/run_pipeline.py \
  --presence-classifier ml/model/checkpoints/presence_classifier_best.pt \
  --embedding-cache ml/data/embedding_cache.npz \
  --local-only
```

---

## Classifier Strategy

Two architectures exist; the binary classifier is currently best:

| Classifier | Input | Output | Current result |
|---|---|---|---|
| `PresenceClassifier` | `[report_emb ‖ label_emb]` (1536-dim) | present/absent per pair | ~33% Good+Slight ✓ |
| `GroupClassifier` | `report_emb` (768-dim) | per-group probability | ~22% Good+Slight (needs more data) |

`GroupClassifier` uses two-stage inference: predict group(s) → cosine selects best term within group. Designed to eliminate the ~33% completely-off floor, but needs ~10k confirmed cases to generalize.

See `documentation/classifier.md` for the full development history and per-phase results.
