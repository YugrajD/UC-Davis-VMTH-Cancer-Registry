# ML Directory — Overview

## What is this?

A machine learning system that maps veterinary pathology report text to standardized
Vet-ICD-O-canine-1 cancer labels (term, group, ICD code).

Two pipelines with distinct roles:

| Pipeline | Purpose | When it runs |
|---|---|---|
| **`petbert_pipeline/`** | Production — embed report text, predict cancer labels | Inference and each training cycle |
| **`keyword_pipeline/`** | Ground-truth labeling — keyword matching on diagnosis text | Separately, run by the domain expert |

The keyword pipeline produces `keyword_predictions.csv`, which becomes the training
ground truth for the PetBERT pipeline. It does not run in production — in production,
only report text is available, not the diagnosis field.

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
│   └── finetune/           PetBERT fine-tuning (WIP)
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
| `pipeline.py` | Top-level orchestration: load → embed → categorize → write |
| `embedding.py` | PetBERT loading, per-column mean-pooled embedding, fine-tuned classifier support |
| `embedding_cache.py` | Save/load cached embeddings to `.npz` — avoids re-running PetBERT every cycle |
| `categorization.py` | Two strategies: cosine argmax (baseline) and two-stage GroupClassifier |
| `io.py` | Write all output files (CSV, NPZ, JSON) |
| `cli.py` | `argparse` CLI; `build_config()` maps args → `ScanConfig` |
| `utils.py` | Text cleaning, device selection, column merging |

### `keyword_pipeline/` — Ground-truth labeling

Keyword-based matching against a curated dictionary. Not part of production inference.

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
| `presence_classifier.py` | Binary MLP: `[col_embs ‖ label_emb] → present/absent` probability |
| `group_classifier.py` | Multi-label MLP: `report_emb → per-group sigmoid probabilities` |

### `training/binary/` — Binary PresenceClassifier training

| File | Role |
|---|---|
| `build_training_pairs.py` | Assembles `training_pairs.csv` from positives, CO negatives, FP negatives |
| `train.py` | Orchestrates the full cycle; trains `PresenceClassifier` on cached embeddings; saves checkpoint |
| `evaluate.py` | Scores petbert_pipeline predictions vs. keyword ground truth → verdicts |
| `log_evaluation.py` | Appends cycle results to `evaluation_history.csv`; prints trend table |
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
    │         └── output/evaluation/evaluation.csv
    │
    ├──► [Step 4.5] update_co_bank
    │         └── output/evaluation/evaluation_co_bank.csv
    │
    ├──► [Step 5] log_evaluation
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
| `output/report/petbert_predictions.csv` | Top-5 predicted labels per case (term, group, ICD code, score) |
| `output/report/petbert_column_scores.csv` | Per-column similarity breakdown |
| `output/report/petbert_provenance.csv` | Per-case traceability (column selected, token counts) |
| `output/report/petbert_similarity_scores.csv` | Full score matrix (N cases × M labels) |
| `output/report/petbert_visualization.csv` | PCA 2-D coordinates per case |
| `output/report/petbert_embeddings.npz` | Raw 768-dim report embedding vectors |
| `output/report/petbert_summary.json` | Run metadata and aggregate prediction counts |
| `output/diagnoses/keyword_predictions.csv` | Ground-truth labels from keyword_pipeline |
| `output/evaluation/evaluation.csv` | Predictions + verdicts for the latest cycle |
| `output/evaluation/evaluation_summary.csv` | Aggregate counts and percentages |
| `output/evaluation/evaluation_history.csv` | One row per training cycle |
| `output/evaluation/evaluation_co_bank.csv` | Rolling bank of completely-off negatives |
| `data/embedding_cache.npz` | Cached PetBERT embeddings |
| `data/training_pairs.csv` | Generated (case, label, target) pairs for classifier training |

---

## Quick Start

> Use `ml/.venv/bin/python3` (macOS/Linux) or `ml/.venv/Scripts/python.exe` (Windows).
> Scripts in `ml/scripts/` inject `ml/` into `sys.path` automatically — no `PYTHONPATH` needed.

**Production inference:**
```bash
ml/.venv/bin/python3 ml/scripts/run_pipeline.py --local-only
```

**Binary classifier training cycle:**
```bash
ml/.venv/bin/python3 ml/scripts/run_training.py \
  --mode binary --label "cycle 1" \
  --co-neg-per-case 5 --fp-neg-per-case 10 \
  --embedding-min-sim 0.05 --epochs 25 \
  --recall-weight 0.25 --device mps --local-only
```

**GroupClassifier training (one-shot):**
```bash
ml/.venv/bin/python3 ml/scripts/run_training.py --mode group --device mps
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
| `training-log/training-log-binary.md` | Binary PresenceClassifier — phase-by-phase history, Phases 1–13 |
| `training-log/training-log-group.md` | GroupClassifier — experiments and results |
| `training-log/training-log-finetune.md` | Fine-tuned PetBERT — prerequisite checklist, no runs yet |
