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

The annotation pipeline produces `llm_annotation.csv`, which becomes the training
supervision. It does not run in production — in production, only report text is
available, not the diagnosis field. The pipeline uses a three-tier cascade
(exact keyword match → fuzzy token overlap → LLM) with negation masking and
behavior-code awareness so that ground-truth labels respect hedging, exclusion,
and benign-vs-malignant distinctions.

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
│   │   └── llm/            llm_annotation.csv + llm_summary.json/md
│   ├── checkpoints/        Trained model weights (.pt files) — gitignored, back up separately
│   │   ├── contrastive/    Adapted PetBERT backbone (production) + legacy LabelPresenceClassifier
│   │   ├── group/          GroupClassifier checkpoints
│   │   ├── case_presence/  CasePresenceClassifier checkpoint (Stage 1 gate)
│   │   └── label_presence/ Per-group LabelPresenceClassifier checkpoints (Stage 3a, Phase 28+)
│   ├── data_analysis/      Annotation coverage stats (run_data_analysis.py): combined .txt + per-section .csv + .png
│   ├── evaluation/         Cycle evaluation results
│   ├── production/         Scoring pipeline predictions and supporting files
│   ├── splits/             Train/test case ID lists (generated once by create_split.py)
│   └── training/           Training-specific artifacts (group_training_data.npz, etc.)
├── analysis/               Standalone analysis utilities (annotation coverage stats)
├── annotation/             Annotation pipeline — maps diagnosis text to ICD-O labels
│   ├── __init__.py         Programmatic API: llm_main()
│   └── llm_pipeline/       LLM-assisted annotation (three-tier cascade + ensemble cleanup)
│       ├── pipeline.py                  Cascade implementation
│       ├── cleanup.py                   Ensemble verification pass
│       ├── client.py                    LM Studio HTTP client
│       ├── cli.py                       Cascade-then-cleanup CLI (called by run_annotation.py)
│       ├── audit.py                     90-row noise-audit harness
│       ├── run_annotation_cleanup.py    Standalone re-run of the cleanup pass
│       └── compare_llm_models.py        Bake-off harness for comparing LM Studio models on Tier-3
├── production/             Production scoring pipelines
│   └── petbert_pipeline/   3/4-stage pipeline (CasePresence → Group → LabelPresence → KW)
├── evaluation/             Model assessment
│   ├── evaluate.py         Score predictions against verified labels → verdicts
│   └── log_evaluation.py   Append cycle results to evaluation_history.csv
├── training/               Training scripts, organized by mode
│   ├── binary/             Legacy LabelPresenceClassifier cycle orchestration + CasePresenceClassifier dataset/training
│   ├── group/              GroupClassifier training (one-shot)
│   ├── label_presence/     Per-group LabelPresenceClassifier training (Stage 3a, one-shot per group)
│   ├── contrastive/        Embedding backbone adaptation (production best)
│   └── data/               Data utilities (train/test split generation)
├── scripts/                Top-level entry points (no PYTHONPATH needed)
│   ├── run_annotation.py            Annotate diagnoses with cancer labels (cascade + cleanup pass; --skip-cleanup to disable)
│   ├── run_data_analysis.py         Annotation coverage statistics
│   ├── run_training.py              4-stage training (modes: train-groups, adapt-backbone, train-case-presence, train-label-presence)
│   ├── run_evaluation.py            Score the latest predictions and record to history
│   └── run_production.py            4-stage production inference (default)
├── config.py               Project-wide path defaults and shared helpers
├── utils/                  Shared helpers (encoding/safe_filename, csv_io/strip_bom, …)
├── text_selection/         Multi-column TF-IDF text selector — used by both production and training
├── requirements.txt        Pinned Python dependencies
└── .venv/                  Python virtual environment
```

---

## Packages

### `production/petbert_pipeline/` — 4-stage production scoring pipeline

Invoked via `run_production.py`. Takes clinical report text, produces cancer label predictions
through CasePresenceClassifier → GroupClassifier → per-group LabelPresenceClassifier → KW correction.

| File | Role |
|---|---|
| `__init__.py` | Package public API — import `run_scan`, `ScanConfig`, `build_config`, `build_parser` from here |
| `types.py` | `ScanConfig`, `ScanOutputs`, and `CategorizationResult` dataclasses |
| `pipeline.py` | Thin orchestrator: load → select text → embed → call each stage → write outputs |
| `embedding.py` | PetBERT loading, per-column mean-pooled embedding |
| `embedding_cache.py` | Save/load cached embeddings — avoids re-running PetBERT every cycle |
| `stages/case_presence_classifier.py` | Stage 1 — CasePresenceClassifier gate |
| `stages/group_classifier.py` | Stage 2 — GroupClassifier |
| `stages/label_presence_classifier.py` | Stage 3a — per-group LabelPresenceClassifier loader + within-group scorer |
| `stages/keyword_correction.py` | Stage 3b — ICD-O behavior + subtype keyword filter |
| `stages/__init__.py` | Per-case dispatcher (`categorize_per_case`) that drives Stage 3a → Stage 3b |
| `io.py` | Write all output files (CSV, NPZ, JSON) |
| `cli.py` | `argparse` CLI; `build_config()` maps args → `ScanConfig` |
| `utils.py` | Text cleaning, device selection, column merging |

### `annotation/llm_pipeline/` — Diagnosis-to-label annotation

Uses a three-tier cascade (exact match → fuzzy → LLM) followed by an ensemble
verification cleanup pass. Tier 1/2 run negation masking and behavior-code-aware
fuzzy matching; Tier 3 calls a local LM-Studio-hosted LLM only for rows with a
cancer-signal term (~15% of rows). The cleanup pass then re-asks two diverse
models per confirmed match and demotes non-unanimous results to `Uncertain`.
Handles negation, hedging, abbreviations, and anatomic-site disambiguation.

| File | Role |
|---|---|
| `pipeline.py` | Tiers 1–3, normalization + negation masking, prompt builder, summary writer |
| `cleanup.py` | Ensemble verification pass — re-asks 2+ models per confirmed row and rewrites `llm_annotation_cleaned.csv` |
| `client.py` | OpenAI-compatible HTTP client (LM Studio) |
| `cli.py` | CLI orchestrating cascade + cleanup; `--skip-cleanup`, `--cleanup-models`, `--list-models` |
| `audit.py` | Reproduces the 90-row stratified noise-audit sample and prints a before/after diff |
| `run_annotation_cleanup.py` | Standalone re-run of the cleanup pass over an existing `llm_annotation.csv` |
| `compare_llm_models.py` | Bake-off harness — runs N LM Studio models on a Tier-3 stratified sample and reports latency / agreement / disagreements |
| `.env` | `LLM_HOST`, `API_PORT`, `LLM_MODEL` |

### `evaluation/evaluate.py` — End-to-end verdict scoring

Scores final pipeline predictions against verified annotation labels
(good / slightly_off / completely_off / false_positive / false_negative).

### `evaluation/evaluate_case_presence.py` — Stage 1 metrics

Binary cancer-vs-non-cancer evaluation of `CasePresenceClassifier` on test
cases. Reports precision, recall, F1, accuracy, AUC.

### `evaluation/evaluate_groups.py` — Stage 2 metrics

Multi-label group evaluation of `GroupClassifier` on cancer test cases only
(isolates Stage 2 from Stage 1 errors). Reports per-group + macro/micro P/R/F1
and top-1/3/5 accuracy.

### `evaluation/evaluate_label_presence.py` — Stage 3 metrics

Per-LP binary evaluation of every `LabelPresenceClassifier` checkpoint on
in-scope cases (cases whose annotation matches that LP's group). Reports
per-LP + macro/micro P/R/F1.

### `evaluation/log_evaluation.py` — Evaluation history

Appends end-to-end cycle results to `evaluation_history.csv` and prints a
trend table. The per-stage modules write their own `*_history.csv` files
directly.

### `ICD_labels/` — Vet-ICD-O taxonomy

Shared by all pipelines. Loads and embeds the taxonomy used for prediction targets.

| File | Role |
|---|---|
| `__init__.py` | Re-exports `load_labels_taxonomy` and `TaxonomyLabel` — import from here, not `taxonomy.py` |
| `taxonomy.py` | Parse `labels.csv` into `TaxonomyLabel(code, group, term)` records |
| `catalog.py` | Build a `LabelCatalog` (label strings + embedding texts) |
| `projection.py` | Map predicted label indices → `(term, group, code)` output fields |
| `behavior_keywords.py` | ICD-O behavior code keyword lists and scorer — applied in Stage 4 (KW correction) by default |
| `subtype_keywords.py` | Histologic/topographic subtype keyword discriminators — applied in Stage 4 KW correction for 6 groups (Phase 27) |
| `labels.csv` | Vet-ICD-O-canine-1 taxonomy: ~857 terms across 44 cancer groups |

### `model/` — Neural network architectures

| File | Role |
|---|---|
| `constants.py` | `PETBERT_EMB_DIM=768`, `DEFAULT_HIDDEN_DIM=512`, `DEFAULT_DROPOUT=0.3`, `DEFAULT_TEXT_COLS` (experiment-path columns) |
| `label_presence_classifier.py` | Binary MLP: `[report_emb ‖ label_emb] → present/absent`. Used as the per-group Stage 3a classifier in the 4-stage pipeline |
| `group_classifier.py` | Multi-label MLP: `report_emb → per-group sigmoid probabilities` |
| `case_presence_classifier.py` | Binary MLP: `report_emb → cancer probability` (case-level gate, Stage 1) |

### `training/binary/` — CasePresenceClassifier (Stage 1) training

| File | Role |
|---|---|
| `build_case_presence_dataset.py` | Build case-level (mean_emb, cancer/no-cancer) dataset for `CasePresenceClassifier` |
| `train_case_presence.py` | Train `CasePresenceClassifier` (Stage 1 gate) — recall-weighted, one-shot |

### `training/group/` — GroupClassifier training

| File | Role |
|---|---|
| `build_training_data.py` | Build multi-hot targets for `GroupClassifier` from cached embeddings (with common/uncommon bucketing via `--uncommon-threshold`) |
| `train.py` | Train `GroupClassifier` (one-shot, not iterative). Supports BCE / Focal / ASL losses and cosine LR schedule |

### `training/label_presence/` — Per-group LabelPresenceClassifier training (Stage 3a)

| File | Role |
|---|---|
| `build_training_pairs.py` | Build within-group `(case, label, target)` pairs for one group; positives are annotation matches, negatives are other labels in the same group. Handles the "Uncommon" bucket as a union of merged groups |
| `train.py` | Train one `LabelPresenceClassifier` (n_cols=1, concat) per group on cached `tfidf_selected` embeddings; save to `{safe_group_name}.pt` |

### `training/contrastive/` — Embedding backbone adaptation (production best)

| File | Role |
|---|---|
| `build_contrastive_dataset.py` | Build `(report_text, label_text)` pairs from annotations + report CSV, using TF-IDF-selected text |
| `train_contrastive.py` | Adapt PetBERT backbone using contrastive loss; save checkpoint |
| `fit_text_selector.py` | Fit and save the TF-IDF vectorizer on the full report corpus — run once before backbone adaptation |

### `training/data/` — Data utilities

| File | Role |
|---|---|
| `create_split.py` | Generate case-level train/test split files (run once; stratified by label group) |

### `analysis/` — Standalone analysis utilities

| File | Role |
|---|---|
| `annotation_stats.py` | Compute annotation coverage stats: cases-per-group, diagnoses-per-case, groups-per-case, same-group collisions. Emits one CSV + one PNG per section (matplotlib, headless `Agg` backend) plus a combined `annotation_distribution.txt`. Invoked by `run_data_analysis.py`; pass `--no-plots` to skip PNGs |

---

## Data Flow (4-stage production pipeline)

```
ml/data/diagnoses.csv  (diagnosis text from database)
    │
    ▼ run_annotation.py
output/annotation/llm/llm_annotation.csv
    │
ml/data/report.csv  (clinical reports)
    │
    ▼ run_training.py --mode adapt-backbone   (one-shot — InfoNCE + hard-neg)
    ▼ run_production.py                       (first run — embed all reports)
output/checkpoints/contrastive/  (adapted PetBERT backbone)
output/training/embedding_cache.npz  (cached PetBERT embeddings, keyed by tfidf_selected text)
    │
    ├──► run_training.py --mode train-case-presence
    │         └── output/checkpoints/case_presence/case_presence_classifier.pt
    │
    ├──► run_training.py --mode train-groups
    │         └── output/checkpoints/group/group_classifier_best.pt
    │
    ├──► run_training.py --mode train-label-presence  (one model per ICD group)
    │         └── output/checkpoints/label_presence/{safe_group_name}.pt
    │
    ▼ run_production.py
output/production/contrastive/petbert_predictions.csv
    │
    ▼ run_evaluation.py --test-cases ml/output/splits/test_cases.txt
output/evaluation/contrastive_test/evaluation.csv
output/evaluation/contrastive_test/evaluation_history.csv
```

---

## Output Files

| Path | Contents |
|---|---|
| `output/production/contrastive/petbert_predictions.csv` | Top-5 predicted labels per case (term, group, ICD code, score) |
| `output/production/contrastive/petbert_column_scores.csv` | Per-column similarity breakdown |
| `output/production/contrastive/petbert_provenance.csv` | Per-case traceability (column selected, token counts) |
| `output/production/contrastive/petbert_similarity_scores.csv` | Full score matrix (N cases × M labels) |
| `output/production/contrastive/petbert_visualization.csv` | PCA 2-D coordinates per case |
| `output/production/contrastive/petbert_embeddings.npz` | Raw 768-dim report embedding vectors |
| `output/production/contrastive/petbert_summary.json` | Run metadata and aggregate prediction counts |
| `output/annotation/llm/llm_annotation.csv` | Annotation labels from LLM method |
| `output/annotation/llm/llm_summary.json` | Run statistics (coverage, group distribution) |
| `output/annotation/llm/llm_summary.md` | Human-readable version of llm_summary.json |
| `output/data_analysis/annotation_distribution.txt` | Annotation coverage statistics report |
| `output/evaluation/contrastive/evaluation.csv` | Predictions + verdicts for the latest cycle (train cases only when split active) |
| `output/evaluation/contrastive/evaluation_summary.csv` | Aggregate counts and percentages |
| `output/evaluation/contrastive/evaluation_history.csv` | One row per training cycle |
| `output/evaluation/contrastive_test/evaluation.csv` | Held-out test set verdicts (written by `run_evaluation.py --test-cases`) |
| `output/evaluation/contrastive_test/evaluation_history.csv` | Test set evaluation history |
| `output/evaluation/{subdir}/case_presence_evaluation{,_summary,_history}.csv` | Stage 1 — per-case + summary + history metrics (P, R, F1, AUC) |
| `output/evaluation/{subdir}/groups_evaluation{,_summary,_history}.csv` | Stage 2 — per (case, group) + per-group + macro/micro + top-k metrics |
| `output/evaluation/{subdir}/label_presence_evaluation{,_summary,_history}.csv` | Stage 3 — per (case, label) + per-LP + macro/micro metrics |
| `output/training/contrastive/evaluation_co_bank.csv` | Rolling bank of wrong-group predictions (legacy training-cycle path) |
| `output/training/contrastive/contrastive_pairs.csv` | (report_text, label_text) pairs for backbone adaptation |
| `output/training/contrastive/hard_neg_pairs.csv` | (report, correct_label, wrong_label) triplets for hard-neg loss |
| `output/training/group/group_training_data.npz` | Cached multi-hot training targets for group classifier |
| `output/training/group/uncommon_groups.txt` | Group names merged into the "Uncommon" bucket |
| `output/training/binary/case_presence_dataset.npz` | Case-level cancer/no-cancer dataset for CasePresenceClassifier |
| `output/training/binary/training_pairs.csv` | Generated (case, label, target) pairs for legacy classifier training |
| `output/training/label_presence/{safe_group_name}_pairs.csv` | Within-group training pairs per ICD group |
| `output/splits/train_cases.txt` | Case IDs reserved for training (generated once by `create_split.py`) |
| `output/splits/test_cases.txt` | Case IDs held out for evaluation (generated once by `create_split.py`) |
| `output/training/embedding_cache.npz` | Cached PetBERT embeddings |
| `output/training/tfidf_selector.joblib` | Fitted per-column TF-IDF vectorizers for multi-column text selection |

---

## Quick Start

> Use `ml/.venv/Scripts/python.exe` (Windows) or `ml/.venv/bin/python3` (macOS/Linux).
> Scripts in `ml/scripts/` inject `ml/` into `sys.path` automatically — no `PYTHONPATH` needed.

**Generate train/test split (run once before first training run):**
```bash
ml/.venv/Scripts/python.exe ml/training/data/create_split.py
```

**Fit TF-IDF vectorizer (run once, before backbone adaptation or cold start):**
```bash
ml/.venv/Scripts/python.exe ml/training/contrastive/fit_text_selector.py
```

**Score all reports with the 4-stage production pipeline (default):**
```bash
ml/.venv/Scripts/python.exe ml/scripts/run_production.py \
  --group-classifier-threshold 0.85 \
  --label-presence-threshold 0.5 \
  --device xpu --local-only
```

**Evaluate on held-out test set (after training):**
```bash
ml/.venv/Scripts/python.exe ml/scripts/run_evaluation.py \
  --test-cases ml/output/splits/test_cases.txt \
  --out-dir ml/output/evaluation/contrastive_test \
  --label "held-out test"
```

**Per-stage evaluation — isolate which classifier moved between training cycles:**
```bash
# All four (end-to-end + Stage 1 + Stage 2 + Stage 3)
ml/.venv/Scripts/python.exe ml/scripts/run_evaluation.py \
  --stage all \
  --test-cases ml/output/splits/test_cases.txt \
  --out-dir ml/output/evaluation/contrastive_test \
  --label "phase XX stage breakdown"

# Or one stage at a time
ml/.venv/Scripts/python.exe ml/scripts/run_evaluation.py --stage case-presence  --test-cases ml/output/splits/test_cases.txt
ml/.venv/Scripts/python.exe ml/scripts/run_evaluation.py --stage groups         --test-cases ml/output/splits/test_cases.txt
ml/.venv/Scripts/python.exe ml/scripts/run_evaluation.py --stage label-presence --test-cases ml/output/splits/test_cases.txt
```

**Annotate diagnoses:**
```bash
ml/.venv/Scripts/python.exe ml/scripts/run_annotation.py
```

**Annotate diagnoses — quick test on first 100 rows:**
```bash
ml/.venv/Scripts/python.exe ml/scripts/run_annotation.py --max-rows 100
```

**Adapt the embedding backbone (run once before train-groups, train-case-presence, or train-label-presence):**
```bash
ml/.venv/Scripts/python.exe ml/scripts/run_training.py \
  --mode adapt-backbone \
  --epochs 3 --device xpu --local-only
```

**Train CasePresenceClassifier (Stage 1 gate, one-shot):**
```bash
ml/.venv/Scripts/python.exe ml/scripts/run_training.py \
  --mode train-case-presence --epochs 20 \
  --case-presence-recall-weight 0.85 \
  --device xpu --local-only \
  --train-cases ml/output/splits/train_cases.txt \
  --annotation-csv ml/output/annotation/llm/llm_annotation.csv
```

**Train GroupClassifier (Stage 2, one-shot):**
```bash
ml/.venv/Scripts/python.exe ml/scripts/run_training.py \
  --mode train-groups --epochs 300 --lr 5e-5 --dropout 0.1 \
  --max-class-weight 50 --weight-decay 1e-3 \
  --device xpu --local-only \
  --train-cases ml/output/splits/train_cases.txt \
  --annotation-csv ml/output/annotation/llm/llm_annotation.csv
```

**Train per-group LabelPresenceClassifiers (Stage 3a, one model per ICD group):**
```bash
ml/.venv/Scripts/python.exe ml/scripts/run_training.py \
  --mode train-label-presence --label-presence-epochs 25 \
  --label-presence-recall-weight 0.5 \
  --label-presence-negs-per-pos 5 \
  --group-classifier-path ml/output/checkpoints/group/group_classifier_best.pt \
  --device xpu --local-only \
  --train-cases ml/output/splits/train_cases.txt \
  --annotation-csv ml/output/annotation/llm/llm_annotation.csv
```

**Annotation coverage statistics:**
```bash
ml/.venv/Scripts/python.exe ml/scripts/run_data_analysis.py
# Writes to ml/output/data_analysis/: annotation_distribution.txt + 4 CSV/PNG pairs
# (cases_per_group, diagnoses_per_case, groups_per_case, collisions).
# Pass --no-plots to skip PNGs.
```

---

## Documentation

| File | Contents |
|---|---|
| `README.md` | This file — project overview, structure, quick start |
| `production-pipeline.md` | Authoritative implementation-based walkthrough of the code path used by `run_production.py` today |
| `label-annotation.md` | Both annotation methods: keyword and LLM — how they work, coverage comparison, known limitations |
| `classifiers.md` | All classifier approaches: architecture, advantages/disadvantages, evaluation results |
| `model-training.md` | Comparison table and architectural decisions |
| `training-guide.md` | Step-by-step how-to: cold start, run commands, expected trajectory |
| `training-log/training-log-binary.md` | Label presence classifier — phase-by-phase history (Phases 1–16) |
| `training-log/training-log-group.md` | Group classifier — experiments and results |
| `training-log/training-log-finetune.md` | Backbone adaptation — contrastive approach (Phase 17) and end-to-end attempt (Approach B, 2026-05, abandoned) |
| `training-log/training-log-label-presence.md` | Per-group LabelPresenceClassifier (Stage 3a) — Phase 28+ |
