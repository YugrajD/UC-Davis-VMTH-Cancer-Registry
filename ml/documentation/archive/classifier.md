# Classifier Development Log

## Architecture Overview

The system has **two pipelines with distinct roles**:

### Production Pipeline — PetBERT Scan

Maps **full pathology report text → cancer group + term + ICD code**. This is the only pipeline that runs in production.

| Stage | Input | Output |
|-------|-------|--------|
| Embedding | Report text columns (HISTOPATHOLOGICAL SUMMARY, FINAL COMMENT, ANCILLARY TESTS) | 768-dim embedding per column via frozen PetBERT; columns kept separate (not mean-pooled) |
| Scoring | Concatenated per-column report embeddings vs. taxonomy label embeddings | Score per (case, label) pair — cosine similarity, or learned probability when a classifier is present |
| Classification | Scores across all labels | Top-k predictions with confidence threshold → term + group + ICD code |

The current scorer is a binary `PresenceClassifier` MLP (`model/presence_classifier.py`) trained on `([col1_emb ‖ col2_emb ‖ col3_emb ‖ label_emb] → present/absent)` — 3072-dim input (Fix 10). It replaces raw cosine similarity scores but is evaluated one pair at a time, which introduces a group-assignment ceiling. See [multiclass-classifier-plan.md](multiclass-classifier-plan.md) for the planned replacement.

### Training Pipeline — Keyword Scan

Maps **diagnosis field text → cancer label**, for the **sole purpose of generating ground-truth training labels for the PetBERT pipeline**. Does not run in production.

The keyword scan matches structured diagnosis strings (e.g. `"SKIN DORSUM: SQUAMOUS CELL CARCINOMA"`) against a curated keyword dictionary to assign Vet-ICD-O labels. It currently covers ~19.2% of diagnosis rows (~5,788 unique cases across 44 cancer groups). The keyword pipeline is actively being improved by a domain expert.

**Ground-truth assumption:** Cases not matched by the keyword scan are treated as **non-cancer (Uncategorized)**. This is valid for a general veterinary clinic population where ~18% cancer prevalence is expected. As keyword coverage improves, training label quality will improve accordingly.

### Training data sources (`build_training_pairs.py`) — binary PresenceClassifier only

These training sources apply to the current binary `PresenceClassifier`. They will not be used by the planned multi-class group classifier (see [multiclass-classifier-plan.md](multiclass-classifier-plan.md)).

| Source | Description |
|--------|-------------|
| `positive` | Keyword-confirmed (case, term) pairs — from the keyword scan (training pipeline only) |
| `hard_negative` | False-positive predictions from previous eval cycle |
| `fp_extra_negative` | Additional random labels sampled for FP cases |
| `co_negative` | Completely-off predictions from the rolling CO bank — the specific wrong-group (case, label) pairs that fool cosine similarity |
| `easy_negative` | Random wrong labels for keyword-confirmed cases |

### Evaluation verdicts (`training/binary/evaluate.py`)

| Verdict | Meaning |
|---------|---------|
| `good` | Predicted term exactly matches a keyword-matched term |
| `slightly_off` | No exact term match but predicted group matches a keyword group |
| `completely_off` | Neither term nor group matches any keyword label for this case |
| `false_positive` | Case has no keyword labels at all (should be Uncategorized) |
| `false_negative` | Confirmed cancer case with no good/slightly_off prediction |

---

## How to Run

> **Note (Windows):** Use `ml/.venv/Scripts/python.exe` on Windows (not `ml/.venv/bin/python3`). Adjust `--device` to match your hardware (`xpu`, `cuda`, `mps`, `cpu`).

**Current recommended command** (bank exists at `ml/output/evaluation/evaluation_co_bank.csv`):

```bash
ml/.venv/Scripts/python.exe ml/scripts/run_training.py \
  --label "..." \
  --co-neg-per-case 5 \
  --fp-neg-per-case 10 \
  --embedding-min-sim 0.05 \
  --epochs 25 \
  --recall-weight 0.25 \
  --device xpu \
  --local-only
```

> **Note (Phase 13):** With the per-column embedding architecture, `--co-neg-per-case 5` outperforms `--co-neg-per-case 10` consistently. Using co=10 with a large bank floods training with hard CO negatives and causes regression. Keep co=5 throughout — no need to switch to co=10 at 20k pairs.

---

## Cold Start (after resetting embeddings or classifier)

A cold start is required any time the embedding space changes — e.g. after updating PetBERT, changing label enrichment logic, or switching `--enrich-labels-csv`. Old bank pairs are anchored to the old cosine space and will add noise; the cache is no longer valid either (see Fix 7).

### Prerequisites

1. `ml/output/diagnoses/keyword_predictions.csv` must exist. If not, run the keyword scan first:
   ```bash
   ml/.venv/bin/python3 -m keyword_pipeline
   ```

2. `ml/data/report.csv` must exist (the input data).

### Files to delete

```bash
rm -f ml/data/embedding_cache.npz                          # stale cache — rebuilt on first cycle
rm -f ml/output/evaluation/evaluation_co_bank.csv          # old-space bank — must start fresh
rm -f ml/model/checkpoints/presence_classifier_current.pt     # old checkpoint
```

### Warm-up phase

The first cycle's Step 0 detects the missing cache and runs PetBERT on all reports and labels. This is the only time PetBERT runs — all subsequent cycles load from cache. It takes several minutes.

```bash
ml/.venv/Scripts/python.exe ml/scripts/run_training.py \
  --label "cold-start c1" \
  --co-neg-per-case 5 \
  --fp-neg-per-case 10 \
  --embedding-min-sim 0.05 \
  --epochs 25 \
  --recall-weight 0.25 \
  --device xpu \
  --local-only
```

Continue all subsequent cycles with `--co-neg-per-case 5`. Do **not** switch to co=10 — with the per-column architecture, co=10 causes regression (Phase 13 c2 experience).

### Expected trajectory (5,788 cancer cases, per-column architecture)

| Cycles completed | Expected Good+Slight | Notes |
|-----------------|---------------------|-------|
| c1 (co=5, cold start) | ~28–30% | Cache rebuilt; bank ~15k |
| c2 (co=5) | ~26% | May dip slightly — continue |
| c3–c4 (co=5) | ~38–39% | Large jump |
| c5–c6 (co=5) | ~39–40% | Plateau; CO floor ~30% |
| c7+ | may regress | Confirm plateau and stop |

---

## Key Parameters

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `--embedding-min-sim` | `0.05` | After Fix 1 (mean subtraction), scores are centered — use 0.05, not 0.5 |
| `--co-neg-per-case` | `10` (bank >20k) / `5` (bank <20k) | Raising to 10 once the bank exceeds ~20k pairs was the key unlock: Good+Slight jumped 13.6% → 21.0% |
| `--fp-neg-per-case` | `10` | Keep at 10; reducing to 5 weakens FP rejection |
| `--epochs` | `25` | Beyond 25 shows diminishing returns |
| `--pos-weight` | `1.0` | Do not increase; the sampler already balances training |
| `--recall-weight` | `0.25` | Score = `(1-rw)·P + rw·R`. At rw=0.5, epoch-1 degenerate checkpoints (R≈0.95, P≈0.10) could win and produced bad cycles. At rw=0.25 they score ~0.31 vs mid-training balanced checkpoints at ~0.39 — they can no longer win. Do not raise above 0.5. |
| `--max-pos-per-group` | `0` (no cap) | Do not cap — removes signal from already-good groups |

---

## Training Log

Per-phase results, fix descriptions, and cycle-by-cycle tables are in [binary-classifier-training-log.md](binary-classifier-training-log.md).

## Known Limitations of the Binary PresenceClassifier

- **Completely-off floor (~30%)**: with 5,788 training cases and per-column embeddings (Phase 13), the CO floor sits at ~30–32%. Further reduction requires either more keyword-confirmed cases or a group-level architecture (GroupClassifier).
- **GroupClassifier still overfits at 5,788 cases**: needs ~10,000+ confirmed cases across 44 groups to generalize reliably. Re-train whenever keyword coverage improves.
- **co=10 regression with per-column architecture**: flooding training with CO negatives at 10/case causes the per-column classifier to over-correct. Always use co=5 with the current architecture.
- **hidden_dim bottleneck (untested)**: `hidden_dim=256` compresses 3072-dim input at a 12:1 ratio; trying 512 or 768 may recover 1–3%. See [presence-classifier-optimizations.md](../../planning/presence-classifier-optimizations.md) for the full list of potential improvements.

The planned multi-class group classifier (see [multiclass-classifier-plan.md](multiclass-classifier-plan.md)) directly addresses the CO floor by replacing pair-wise binary scoring with a single global group decision per report.

---

## Work in Progress — Fine-tuned PetBERT Classifier

### Motivation

In production there will be no diagnosis text field available — only the raw report text columns. The binary `PresenceClassifier` and `GroupClassifier` both operate on frozen PetBERT embeddings, so PetBERT itself never learns veterinary diagnostic language from this task. Fine-tuning PetBERT end-to-end on the group classification objective should allow it to directly use report text to predict cancer groups, without needing a separate embedding + classifier head pipeline.

### Approach

The fine-tuned model replaces the frozen-embedding + GroupClassifier two-stage approach with a single sequence classification model:

```
Training:   report text + (diagnosis → keyword pipeline → group label) → fine-tune PetBERT
Inference:  report text → fine-tuned PetBERT → group probabilities → cosine within group → ICD term
```

The keyword pipeline is used **only to generate training labels** from the existing diagnosis text. At inference, the diagnosis field is not needed — the fine-tuned model has learned to predict groups directly from report text. Within the predicted group, cosine similarity against taxonomy label embeddings (using the base unfinetuned PetBERT) still selects the best specific ICD term.

### Scripts

| Script | Role |
|--------|------|
| `ml/training/finetune/build_dataset.py` | Build HuggingFace `DatasetDict` from `report.csv` + `keyword_predictions.csv`; compute inverse-frequency class weights |
| `ml/training/finetune/train.py` | Fine-tune `AutoModelForSequenceClassification` on top of PetBERT using `WeightedTrainer` (custom HF Trainer with class-weighted CrossEntropyLoss) |

### Usage

```bash
# Step 1 — build the fine-tuning dataset
ml/.venv/bin/python3 ml/training/finetune/build_dataset.py \
  --reports-csv database/data/output/report.csv \
  --predictions-csv ml/output/diagnoses/keyword_predictions.csv \
  --labels-csv ml/labels/labels.csv \
  --out-dir ml/data/finetune_dataset

# Step 2 — fine-tune PetBERT
ml/.venv/bin/python3 ml/training/finetune/train.py \
  --dataset ml/data/finetune_dataset \
  --out-dir ml/model/checkpoints/petbert_finetuned \
  --epochs 5

# Step 3 — run the pipeline with the fine-tuned model
ml/.venv/bin/python3 ml/scripts/run_pipeline.py \
  --finetuned-model-path ml/model/checkpoints/petbert_finetuned \
  --local-only
```

### Known Issues (WIP)

- **`WeightedTrainer` constructor order**: `__init__(self, class_weights=None, *args, **kwargs)` is fragile — if `model` is passed positionally it binds to `class_weights`. Should be `(self, *args, class_weights=None, **kwargs)`.
- **Device mismatch risk**: class weights tensor is moved to `self.args.device` during `__init__`, but `self.args.device` may not be resolved yet at construction time. Safer to move during `compute_loss`.
- **No stratified val split**: `build_dataset.py` calls `train_test_split` without `stratify_by_column="labels"`, so rare cancer groups may be underrepresented in validation.
- **`--finetuned-model-path` and `--presence-classifier` are not mutually exclusive**: the pipeline does not guard against using both together. If both are set, the PresenceClassifier silently receives zero embeddings and produces wrong scores.
- **`evaluation_strategy`/`save_strategy` deprecated**: newer versions of `transformers` use `eval_strategy` instead.
- **No `local_files_only` in `build_dataset.py`**: `AutoTokenizer.from_pretrained` will attempt a network call unless `--local-only` is passed at the CLI level (which `build_dataset.py` does not currently support).

### Status

Not yet benchmarked. The WIP pipeline integration (`--finetuned-model-path` flag in `run_pipeline.py`) is functional but the known issues above should be resolved before a full training run.
