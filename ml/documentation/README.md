# ML Directory — Overview

## What is this?

A machine learning system that maps veterinary pathology report text to standardized
Vet-ICD-O-canine-1 cancer labels (term, group, ICD code).

Four pipelines with distinct roles:

| Pipeline | Purpose | When it runs |
|---|---|---|
| **`production/`** | Embed report text, predict cancer labels | Inference and each training cycle |
| **`annotation/`** | Annotate diagnosis text with verified labels | Before training; user picks keyword or LLM method |
| **`evaluation/`** | Score predictions against verified labels | After each training cycle |
| **`training/`** | Train and retrain classifiers | On demand |

The annotation pipeline produces labels (`keyword_annotation.csv` or `llm_annotation.csv`)
which become the training supervision. It does not run in production — in production, only
report text is available, not the diagnosis field. The keyword method is a fast
option for testing when the Ollama server is unavailable, but the LLM method is
the authoritative source: it handles negation, hedged language, and abbreviations that
keyword matching cannot.

---

## Directory Structure

```
ml/
├── data/                   Input data and generated intermediate files
├── documentation/          Design docs and training history
├── ICD_labels/             Vet-ICD-O taxonomy — loading, embedding, projection
├── model/                  Neural network architectures and shared constants
├── output/                 All generated outputs (predictions, evaluation, etc.)
│   ├── annotation/         Annotation pipeline outputs
│   │   ├── keyword/        keyword_annotation.csv + keyword_summary.json
│   │   └── llm/            llm_annotation.csv + llm_summary.json/md
│   ├── checkpoints/        Trained model weights (.pt files) — gitignored, back up separately
│   │   ├── binary/         Frozen-backbone PresenceClassifier checkpoints
│   │   ├── contrastive/    Adapted backbone + PresenceClassifier checkpoints (production)
│   │   ├── group/          GroupClassifier checkpoints
│   │   └── knn_selector/   KNN group selector .npz
│   ├── evaluation/         Cycle evaluation results
│   ├── production/         Scoring pipeline predictions and supporting files
│   └── training/           Training-specific artifacts (e.g. group_training_data.npz)
├── annotation/             Annotation pipeline — maps diagnosis text to ICD-O labels
│   ├── __init__.py         Programmatic API: annotate_keyword()
│   ├── cli.py              CLI dispatch: --method keyword|llm (requires PYTHONPATH=ml)
│   ├── keyword_pipeline/   Rule-based annotation (keyword method)
│   └── llm_pipeline/       LLM-assisted annotation (authoritative method)
├── production/             Production scoring pipeline
│   └── petbert_pipeline/   PetBERT embedding + classifier scoring
├── evaluation/             Model assessment
│   ├── evaluate.py         Score predictions against verified labels → verdicts
│   └── log_evaluation.py   Append cycle results to evaluation_history.csv
├── training/               Training scripts, organized by mode
│   ├── binary/             Label presence classifier training + cycle orchestration
│   ├── group/              Group classifier training (one-shot)
│   ├── contrastive/        Embedding backbone adaptation (production best)
│   └── finetune/           End-to-end PetBERT fine-tuning (WIP)
├── scripts/                Top-level entry points (no PYTHONPATH needed)
│   ├── run_annotation.py   Annotate diagnoses with cancer labels
│   ├── run_training.py     Train classifiers (modes: train-classifier, train-groups, adapt-backbone, build-knn)
│   ├── run_evaluation.py   Score the latest predictions and record to history
│   └── run_production.py   Score all reports with the best available classifier
├── config.py               Project-wide path defaults and shared helpers
├── requirements.txt        Pinned Python dependencies
└── .venv/                  Python virtual environment
```

---

## Packages

### `production/petbert_pipeline/` — Production scoring pipeline

Invoked via `run_production.py`. Takes clinical report text, produces cancer label predictions.

| File | Role |
|---|---|
| `__init__.py` | Package public API — import `run_scan`, `ScanConfig`, `build_config`, `build_parser` from here |
| `types.py` | `ScanConfig` and `ScanOutputs` dataclasses (internal — access via `__init__.py`) |
| `pipeline.py` | Top-level orchestration: load → embed → score → write |
| `embedding.py` | PetBERT loading, per-column mean-pooled embedding |
| `embedding_cache.py` | Save/load cached embeddings — avoids re-running PetBERT every cycle |
| `categorization.py` | Four strategies: cosine argmax (default), two-stage GroupClassifier, KNN hybrid, and group-keyword (behavior keyword term selection) |
| `io.py` | Write all output files (CSV, NPZ, JSON) |
| `cli.py` | `argparse` CLI; `build_config()` maps args → `ScanConfig` |
| `utils.py` | Text cleaning, device selection, column merging |

### `annotation/keyword_pipeline/` — Rule-based label annotation

Keyword matching against a curated dictionary. Fast, no model required.

| File | Role |
|---|---|
| `__init__.py` | `annotate_with_defaults(csv_path, labels_csv_path, out_dir)` — project-level convenience wrapper |
| `pipeline.py` | Match diagnosis text against the keyword dictionary |
| `cli.py` | CLI entry point |

### `annotation/llm_pipeline/` — LLM-assisted label annotation (authoritative)

Uses a three-tier cascade (exact match → fuzzy → Ollama LLM) to annotate
diagnoses. The LLM is called only for rows that contain a cancer signal term (~15% of
rows). Handles negation, hedged language, and abbreviations.

| File | Role |
|---|---|
| `pipeline.py` | Tiers 1–3, prompt builders, summary writer |
| `client.py` | Ollama HTTP client |
| `cli.py` | CLI with `--list-models`, `--compare-models`, `--model` |
| `.env` | `TAILSCALE_IP`, `API_PORT`, `OLLAMA_MODEL` |

### `evaluation/evaluate.py` — Verdict scoring

Scores predictions against verified annotation labels (good / slightly_off /
completely_off / false_positive / false_negative).

### `evaluation/log_evaluation.py` — Evaluation history

Appends cycle results to `evaluation_history.csv` and prints a trend table.

### `ICD_labels/` — Vet-ICD-O taxonomy

Shared by all pipelines. Loads and embeds the taxonomy used for prediction targets.

| File | Role |
|---|---|
| `__init__.py` | Re-exports `load_labels_taxonomy` and `TaxonomyLabel` — import from here, not `taxonomy.py` |
| `taxonomy.py` | Parse `labels.csv` into `TaxonomyLabel(code, group, term)` records |
| `catalog.py` | Build a `LabelCatalog` (label strings + embedding texts) |
| `projection.py` | Map predicted label indices → `(term, group, code)` output fields |
| `behavior_keywords.py` | ICD-O behavior code keyword lists and scorer — used by `--categorization-mode group-keyword` |
| `enrichment.py` | Blend label embeddings toward confirmed-case centroids (experimental) |
| `labels.csv` | Vet-ICD-O-canine-1 taxonomy: ~857 terms across 44 cancer groups |

### `model/` — Neural network architectures

| File | Role |
|---|---|
| `constants.py` | `PETBERT_EMB_DIM=768`, `DEFAULT_HIDDEN_DIM=512`, `DEFAULT_DROPOUT=0.3` |
| `presence_classifier.py` | Binary MLP: `[col_embs ‖ label_emb] → present/absent` |
| `group_classifier.py` | Multi-label MLP: `report_emb → per-group sigmoid probabilities` |

### `training/binary/` — Label presence classifier training

| File | Role |
|---|---|
| `run_cycle.py` | Orchestrate one full label-classifier training cycle (Steps 0–5) |
| `build_training_pairs.py` | Assemble `training_pairs.csv` from positives, wrong-group negatives, FP negatives |
| `train.py` | Train `PresenceClassifier` on cached embeddings; save checkpoint |
| `update_co_bank.py` | Maintain rolling bank of wrong-group predictions across cycles |

### `training/group/` — Group classifier training

| File | Role |
|---|---|
| `build_training_data.py` | Build multi-hot targets for `GroupClassifier` from cached embeddings |
| `train.py` | Train `GroupClassifier` (one-shot, not iterative) |

### `training/contrastive/` — Embedding backbone adaptation (production best)

| File | Role |
|---|---|
| `build_contrastive_dataset.py` | Build `(report_text, label_text)` pairs from annotations + report CSV |
| `train_contrastive.py` | Adapt PetBERT backbone using contrastive loss; save checkpoint |

### `training/finetune/` — End-to-end PetBERT fine-tuning (WIP)

| File | Role |
|---|---|
| `build_dataset.py` | Build HuggingFace `DatasetDict` for end-to-end group classification |
| `train.py` | Fine-tune `AutoModelForSequenceClassification` (WIP, blocked) |

---

## Data Flow

```
ml/data/diagnoses.csv  (diagnosis text from database)
    │
    ▼ run_annotation.py --method llm  (authoritative)
    ▼ run_annotation.py --method keyword  (fast fallback)
output/annotation/llm/llm_annotation.csv
output/annotation/keyword/keyword_annotation.csv
    │
ml/data/report.csv  (clinical reports, 12,620 cases)
    │
    ▼ run_production.py  (Step 0 — first run only: embed all reports)
ml/data/embedding_cache.npz
    │
    ├──► [Step 1] assemble training data  ◄── annotation + feedback bank
    │         └── data/training_pairs.csv
    │
    ├──► [Step 2] train label presence classifier
    │         └── output/checkpoints/{contrastive,binary}/presence_classifier_current.pt
    │
    ├──► [Step 3] score all reports with updated classifier
    │         └── output/production/{contrastive,binary}/petbert_predictions.csv
    │
    ├──► [Step 4] score predictions against verified labels
    │         └── output/evaluation/{contrastive,binary}/evaluation.csv
    │
    ├──► [Step 4.5] record wrong-group predictions in feedback bank
    │         └── output/training/{contrastive,binary}/evaluation_co_bank.csv
    │
    ├──► [Step 5] record cycle results to history
    │         └── output/evaluation/{contrastive,binary}/evaluation_history.csv
    │
    └──► [Step 5.5] promote checkpoint if new best
              presence_classifier_current.pt → presence_classifier_best.pt
```

---

## Output Files

| Path | Contents |
|---|---|
| `output/production/{contrastive,binary}/petbert_predictions.csv` | Top-5 predicted labels per case (term, group, ICD code, score) |
| `output/production/{contrastive,binary}/petbert_column_scores.csv` | Per-column similarity breakdown |
| `output/production/{contrastive,binary}/petbert_provenance.csv` | Per-case traceability (column selected, token counts) |
| `output/production/{contrastive,binary}/petbert_similarity_scores.csv` | Full score matrix (N cases × M labels) |
| `output/production/{contrastive,binary}/petbert_visualization.csv` | PCA 2-D coordinates per case |
| `output/production/{contrastive,binary}/petbert_embeddings.npz` | Raw 768-dim report embedding vectors |
| `output/production/{contrastive,binary}/petbert_summary.json` | Run metadata and aggregate prediction counts |
| `output/annotation/keyword/keyword_annotation.csv` | Annotation labels from keyword method |
| `output/annotation/keyword/keyword_summary.json` | Run statistics (coverage, group distribution) |
| `output/annotation/llm/llm_annotation.csv` | Annotation labels from LLM method |
| `output/annotation/llm/llm_summary.json` | Run statistics (coverage, group distribution) |
| `output/annotation/llm/llm_summary.md` | Human-readable version of llm_summary.json |
| `output/evaluation/{contrastive,binary}/evaluation.csv` | Predictions + verdicts for the latest cycle |
| `output/evaluation/{contrastive,binary}/evaluation_summary.csv` | Aggregate counts and percentages |
| `output/evaluation/{contrastive,binary}/evaluation_history.csv` | One row per training cycle |
| `output/training/{contrastive,binary}/evaluation_co_bank.csv` | Rolling bank of wrong-group predictions |
| `output/training/group/group_training_data.npz` | Cached multi-hot training targets for group classifier |
| `data/embedding_cache.npz` | Cached PetBERT embeddings |
| `data/training_pairs.csv` | Generated (case, label, target) pairs for classifier training |

---

## Quick Start

> Use `ml/.venv/Scripts/python.exe` (Windows) or `ml/.venv/bin/python3` (macOS/Linux).
> Scripts in `ml/scripts/` inject `ml/` into `sys.path` automatically — no `PYTHONPATH` needed.

**Score all reports with the best available classifier:**
```bash
ml/.venv/Scripts/python.exe ml/scripts/run_production.py --local-only
```

**Run a label-classifier training cycle:**
```bash
ml/.venv/Scripts/python.exe ml/scripts/run_training.py \
  --mode train-classifier --label "c1" \
  --model ml/output/checkpoints/contrastive \
  --co-neg-per-case 5 --fp-neg-per-case 10 \
  --embedding-min-sim 0.05 --epochs 25 \
  --recall-weight 0.25 --device xpu --local-only
```

**Score predictions and record to history:**
```bash
ml/.venv/Scripts/python.exe ml/scripts/run_evaluation.py --label "after c3"
```

**Annotate diagnoses — LLM method (authoritative):**
```bash
ml/.venv/Scripts/python.exe ml/scripts/run_annotation.py --method llm
```

**Annotate diagnoses — keyword method (fast, no model):**
```bash
ml/.venv/Scripts/python.exe ml/scripts/run_annotation.py --method keyword
```

**Annotate diagnoses — LLM method, quick test:**
```bash
ml/.venv/Scripts/python.exe ml/scripts/run_annotation.py --method llm --max-rows 100
```

**Adapt the embedding backbone (run once before train-classifier cycles):**
```bash
ml/.venv/Scripts/python.exe ml/scripts/run_training.py \
  --mode adapt-backbone \
  --epochs 3 --device xpu --local-only
```

**Train group classifier (one-shot):**
```bash
ml/.venv/Scripts/python.exe ml/scripts/run_training.py --mode train-groups --device xpu
```

**Build KNN group lookup:**
```bash
ml/.venv/Scripts/python.exe ml/scripts/run_training.py --mode build-knn
```

---

## Documentation

| File | Contents |
|---|---|
| `README.md` | This file — project overview, structure, quick start |
| `petbert-pipeline.md` | Production scoring pipeline: how it works, CLI reference, output formats |
| `label-annotation.md` | Both annotation methods: keyword and LLM — how they work, coverage comparison, known limitations |
| `classifiers.md` | All classifier approaches: architecture, advantages/disadvantages, evaluation results |
| `model-training.md` | Comparison table and architectural decisions |
| `training-guide.md` | Step-by-step how-to: cold start, run commands, expected trajectory |
| `training-log/training-log-binary.md` | Label presence classifier — phase-by-phase history (Phases 1–16) |
| `training-log/training-log-group.md` | Group classifier — experiments and results |
| `training-log/training-log-finetune.md` | Backbone adaptation — contrastive approach (Phase 17) and end-to-end WIP |
