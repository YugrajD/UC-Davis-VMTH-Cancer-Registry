# ML Directory — Overview

## What is this?

A machine learning system that maps veterinary pathology report text to standardized
Vet-ICD-O-canine-1 cancer labels (term, group, ICD code).

Three pipelines with distinct roles:

| Pipeline | Purpose | When it runs |
|---|---|---|
| **`production/`** | Embed report text, predict cancer labels | Inference and each training cycle |
| **`evaluation/`** | Generate ground-truth labels, score predictions | Before training and after each cycle |
| **`training/`** | Train and retrain classifiers | On demand |

The evaluation pipeline produces `keyword_predictions.csv` (and in future, LLM-based labels),
which becomes the training ground truth. It does not run in production — in production,
only report text is available, not the diagnosis field.

---

## Directory Structure

```
ml/
├── data/                   Input data and generated intermediate files
├── documentation/          Design docs and training history
├── labels/                 Vet-ICD-O taxonomy — loading, embedding, projection
├── model/                  Neural network architectures and shared constants
│   └── checkpoints/        Saved model weights (.pt files)
├── output/                 All generated outputs (predictions, evaluation, etc.)
│   ├── evaluation/         Keyword pipeline results (ground truth) + cycle evaluation
│   ├── production/         petbert_pipeline predictions and supporting files
│   └── training/           Training-specific artifacts (e.g. group_training_data.npz)
├── production/             Production inference pipeline
│   └── petbert_pipeline/   PetBERT embedding + classifier scoring
├── evaluation/             Ground-truth generation + model assessment
│   ├── keyword_pipeline/   Keyword-based ground-truth labeling
│   ├── llm_pipeline/       LLM-based ground-truth labeling (in development)
│   ├── evaluate.py         Score predictions against ground truth → verdicts
│   └── log_evaluation.py   Append cycle results to evaluation_history.csv
├── training/               Training scripts, organized by mode
│   ├── run_cycle.py        Orchestrates a full binary training cycle
│   ├── binary/             Binary PresenceClassifier training
│   ├── group/              GroupClassifier training (one-shot)
│   └── finetune/           PetBERT fine-tuning (WIP)
├── scripts/                Top-level entry points (no PYTHONPATH needed)
│   ├── run_production.py   Production inference entry point
│   ├── run_evaluation.py   Standalone evaluation entry point
│   └── run_training.py     Full training cycle entry point (binary or group)
├── requirements.txt        Pinned Python dependencies
└── .venv/                  Python virtual environment
```

---

## Packages

### `production/petbert_pipeline/` — Production pipeline

Invoked via `run_production.py`. Takes clinical report text, produces cancer predictions.

| File | Role |
|---|---|
| `types.py` | `ScanConfig` and `ScanOutputs` dataclasses — all pipeline config in one place |
| `pipeline.py` | Top-level orchestration: load → embed → categorize → write |
| `embedding.py` | PetBERT loading, per-column mean-pooled embedding, fine-tuned classifier support |
| `embedding_cache.py` | Save/load cached embeddings to `.npz` — avoids re-running PetBERT every cycle |
| `categorization.py` | Two strategies: cosine argmax (baseline) and two-stage GroupClassifier |
| `io.py` | Write all output files (CSV, NPZ, JSON) |
| `cli.py` | `argparse` CLI; `build_config()` maps args → `ScanConfig` |
| `utils.py` | Text cleaning, device selection, column merging |

### `evaluation/keyword_pipeline/` — Keyword ground-truth labeling

Keyword-based matching against a curated dictionary. Not part of production inference.

| File | Role |
|---|---|
| `pipeline.py` | Scan diagnosis text against keyword dictionary |
| `cli.py` | CLI entry point |

### `evaluation/llm_pipeline/` — LLM ground-truth labeling (in development)

Future replacement/supplement for the keyword pipeline using an LLM to generate
higher-coverage ground-truth labels from diagnosis text.

### `evaluation/evaluate.py` — Verdict scoring

Scores production predictions against ground-truth labels (good / slightly_off /
completely_off / false_positive / false_negative). Used by both the training cycle
and standalone evaluation runs.

### `evaluation/log_evaluation.py` — Evaluation history

Appends cycle results to `evaluation_history.csv` and prints a trend table.

### `labels/` — Vet-ICD-O taxonomy

Shared by all pipelines. Loads and embeds the taxonomy used for prediction targets.

| File | Role |
|---|---|
| `taxonomy.py` | Parse `labels.csv` into `TaxonomyLabel(code, group, term)` records |
| `catalog.py` | Build a `LabelCatalog` (label strings + embedding texts) from a labels CSV path |
| `projection.py` | Map predicted label indices → `(term, group, code)` output fields |
| `enrichment.py` | Blend label embeddings toward the centroid of confirmed cases (experimental) |
| `labels.csv` | Vet-ICD-O-canine-1 taxonomy: ~857 terms across 44 cancer groups |

### `model/` — Neural network architectures

| File | Role |
|---|---|
| `constants.py` | `PETBERT_EMB_DIM=768`, `DEFAULT_HIDDEN_DIM=256`, `DEFAULT_DROPOUT=0.3` |
| `presence_classifier.py` | Binary MLP: `[col_embs ‖ label_emb] → present/absent` probability |
| `group_classifier.py` | Multi-label MLP: `report_emb → per-group sigmoid probabilities` |

### `training/binary/` — Binary PresenceClassifier training

| File | Role |
|---|---|
| `build_training_pairs.py` | Assembles `training_pairs.csv` from positives, CO negatives, FP negatives |
| `train.py` | Trains `PresenceClassifier` on cached embeddings; saves checkpoint |
| `update_co_bank.py` | Maintains rolling bank of completely-off negatives across cycles |

### `training/group/` — GroupClassifier training

| File | Role |
|---|---|
| `build_training_data.py` | Builds multi-hot targets for `GroupClassifier` from cached embeddings |
| `train.py` | Trains `GroupClassifier` (one-shot, not iterative) |

### `training/finetune/` — PetBERT fine-tuning (WIP)

| File | Role |
|---|---|
| `build_dataset.py` | Build HuggingFace `DatasetDict` from report text + keyword labels; compute class weights |
| `train.py` | Fine-tune `AutoModelForSequenceClassification` on top of PetBERT |

---

## Data Flow

```
ml/data/diagnoses.csv  (diagnosis text from database)
    │
    ▼ evaluation/keyword_pipeline  (or llm_pipeline in future)
output/evaluation/keyword_predictions.csv  (ground truth labels)
    │
ml/data/report.csv  (clinical reports, 12,620 cases)
    │
    ▼ production/petbert_pipeline (Step 0 — first run only)
ml/data/embedding_cache.npz  (768-dim PetBERT embeddings, cached)
    │
    ├──► [Step 1] build_training_pairs  ◄── output/evaluation/keyword_predictions.csv
    │         └── data/training_pairs.csv  ◄── output/evaluation/evaluation_co_bank.csv
    │
    ├──► [Step 2] train
    │         └── model/checkpoints/presence_classifier_current.pt
    │
    ├──► [Step 3] production/petbert_pipeline  (with trained classifier)
    │         └── output/production/petbert_predictions.csv
    │
    ├──► [Step 4] evaluation/evaluate
    │         └── output/evaluation/evaluation.csv
    │
    ├──► [Step 4.5] training/binary/update_co_bank
    │         └── output/evaluation/evaluation_co_bank.csv
    │
    ├──► [Step 5] evaluation/log_evaluation
    │         └── output/evaluation/evaluation_history.csv
    │
    └──► [Step 5.5] auto-promote best checkpoint
              If current cycle's Good+Slight ≥ all-time best:
              presence_classifier_current.pt → presence_classifier_best.pt
```

---

## Output Files

| Path | Contents |
|---|---|
| `output/production/petbert_predictions.csv` | Top-5 predicted labels per case (term, group, ICD code, score) |
| `output/production/petbert_column_scores.csv` | Per-column similarity breakdown |
| `output/production/petbert_provenance.csv` | Per-case traceability (column selected, token counts) |
| `output/production/petbert_similarity_scores.csv` | Full score matrix (N cases × M labels) |
| `output/production/petbert_visualization.csv` | PCA 2-D coordinates per case |
| `output/production/petbert_embeddings.npz` | Raw 768-dim report embedding vectors |
| `output/production/petbert_summary.json` | Run metadata and aggregate prediction counts |
| `output/evaluation/keyword_predictions.csv` | Ground-truth labels from keyword_pipeline |
| `output/evaluation/evaluation.csv` | Predictions + verdicts for the latest cycle |
| `output/evaluation/evaluation_summary.csv` | Aggregate counts and percentages |
| `output/evaluation/evaluation_history.csv` | One row per training cycle |
| `output/evaluation/evaluation_co_bank.csv` | Rolling bank of completely-off negatives |
| `output/training/group_training_data.npz` | Cached multi-hot training targets for GroupClassifier |
| `data/embedding_cache.npz` | Cached PetBERT embeddings |
| `data/training_pairs.csv` | Generated (case, label, target) pairs for classifier training |

---

## Quick Start

> Use `ml/.venv/Scripts/python.exe` (Windows) or `ml/.venv/bin/python3` (macOS/Linux).
> Scripts in `ml/scripts/` inject `ml/` into `sys.path` automatically — no `PYTHONPATH` needed.

**Production inference:**
```bash
ml/.venv/Scripts/python.exe ml/scripts/run_production.py --local-only
```

**Binary classifier training cycle:**
```bash
ml/.venv/Scripts/python.exe ml/scripts/run_training.py \
  --mode binary --label "cycle 1" \
  --co-neg-per-case 5 --fp-neg-per-case 10 \
  --embedding-min-sim 0.05 --epochs 25 \
  --recall-weight 0.25 --device xpu --local-only
```

**Standalone evaluation:**
```bash
ml/.venv/Scripts/python.exe ml/scripts/run_evaluation.py --label "manual check"
```

**GroupClassifier training (one-shot):**
```bash
ml/.venv/Scripts/python.exe ml/scripts/run_training.py --mode group --device xpu
```

---

## Documentation

| File | Contents |
|---|---|
| `README.md` | This file — project overview, structure, quick start |
| `petbert-pipeline.md` | Production pipeline: how it works, CLI reference, output formats, WIP fine-tuning mode |
| `keyword-pipeline.md` | Keyword pipeline: how it works, CLI reference, output formats, known limitations |
| `classifiers.md` | All three approaches: architecture, flowcharts, advantages/disadvantages, constraints, evaluation results, comparison |
| `training-guide.md` | Practical how-to: cold start steps, run commands, parameters, what triggers a cold start |
| `training-log.md` | Index — links to per-method training logs |
| `training-log/training-log-binary.md` | Binary PresenceClassifier — phase-by-phase history, Phases 1–16 |
| `training-log/training-log-group.md` | GroupClassifier — experiments and results |
| `training-log/training-log-finetune.md` | Fine-tuned PetBERT — prerequisite checklist, no runs yet |
