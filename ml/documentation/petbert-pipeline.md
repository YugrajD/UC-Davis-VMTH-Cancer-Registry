# PetBERT Pipeline

The production system that maps full veterinary pathology report text to standardized
Vet-ICD-O-canine-1 cancer labels (term, group, ICD code).

> **Role in the two-pipeline architecture:**
> This pipeline runs in production. It takes report text columns as input — the diagnosis
> field is not available at inference time. The **keyword pipeline** (`keyword-pipeline.md`)
> runs separately to generate ground-truth training labels from the diagnosis field.

---

## How It Works

Each report section is embedded independently through PetBERT and matched against the
taxonomy to produce up to 5 ranked predictions per case:

1. Each selected text column (HISTOPATHOLOGICAL SUMMARY, FINAL COMMENT, ANCILLARY TESTS)
   is embedded independently through PetBERT → 768-dim vector per column per case.
2. Every taxonomy label is embedded through the same PetBERT model.
3. *(Optional)* Label embeddings are enriched by blending 50/50 with the mean report
   embedding of keyword-confirmed cases.
4. An `(N, num_labels)` score matrix is produced by one of two strategies: cosine
   similarity (default) or a trained `PresenceClassifier` MLP.
5. Labels above the confidence threshold are returned as ranked predictions (up to 5).
6. Each prediction's label index is resolved to a term, group, and ICD code.

---

## Input Format

The pipeline reads `ml/data/report.csv` — one row per case:

| Column | Role |
|--------|------|
| `case_id` | Unique case identifier |
| `HISTOPATHOLOGICAL SUMMARY` | Microscopic pathology findings (primary input) |
| `FINAL COMMENT` | Pathologist's diagnostic conclusion / interpretation |
| `ANCILLARY TESTS` | IHC, special stains, PCR, and other ancillary results |
| `ADDENDUM` | Follow-up notes or clinical consultations |
| `CLINICAL ABSTRACT` | Referring clinician's history and differential diagnoses |
| `GROSS DESCRIPTION` | Macroscopic description of submitted tissue samples |

---

## Pipeline Flow

The entry point is `run_scan()` in `ml/petbert_pipeline/pipeline.py`.

### Step 1: Load Input Data

The CSV is read with `latin-1` encoding (handles extended characters in pathology reports).
UTF-8 BOM artifacts from Excel-exported files are stripped from column names. Each cell is
cleaned: whitespace stripped, `NaN` replaced with empty strings.

### Step 2: Embed Each Column Independently

```
col_texts  →  embed_columns_separate()  →  {column: (N, 768)}
```

Each selected column is passed through PetBERT separately. Each text is tokenized
(`padding=True`, `truncation=True`, `max_length=256`) then forward-passed through
PetBERT's base transformer (not the MLM head) and mean-pooled over non-padding tokens:

```python
mask = attention_mask.unsqueeze(-1).float()
mean_embedding = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
```

A cross-column mean embedding is also computed per case (`mean_embeddings`) and used for
PCA visualization, nearest-neighbor search, the embeddings NPZ, and label enrichment.
The presence classifier uses per-column concatenation, not this mean (see Step 4).

### Step 3: Build and Embed Taxonomy Labels

Each of the ~857 taxonomy labels is converted to a short text string:

```
"Hemangiosarcoma, NOS Blood vessel tumors"
```

These are embedded through the same PetBERT model producing a `(num_labels, 768)` matrix.

### Step 3.5: Optional Label Enrichment (`--enrich-labels-csv`)

When provided, each label embedding is blended 50/50 with the mean report embedding of
its keyword-confirmed cases:

```
For each label term T with at least one keyword-confirmed case:
  confirmed_embs = mean_embeddings[cases where matched_term == T]
  enriched[T] = (label_emb[T] + mean(confirmed_embs)) / 2
```

Enriched embeddings are stored in the embedding cache under `enriched_label_embeddings`
and reused on subsequent runs.

> **Note:** Label enrichment caused regressions in classifier training (Fix 6, Fix 9).
> It is off by default and not recommended for binary PresenceClassifier training.

### Step 4: Score Matrix

Produces an `(N, num_labels)` score matrix. Two mutually exclusive strategies:

**Default — Cosine similarity (per column):**

One `(N, num_labels)` similarity matrix per column. Element-wise maximum across columns
so the strongest column wins per (case, label) pair:

```
                     Label 0    Label 1    ...
Col: HIST SUMMARY  [  0.72       0.45      ... ]
Col: FINAL COMMENT [  0.68       0.55      ... ]
                     -----       -----
Max across cols    [  0.72       0.55      ... ]   ← used for ranking
```

**Alternative — PresenceClassifier MLP (`--presence-classifier`):**

When set, cosine similarity is skipped entirely. The classifier scores every (case, label)
pair using per-column embeddings concatenated with the label embedding:

```
Input:  concat([col1_emb (768), col2_emb (768), col3_emb (768), label_emb (768)])
        → 3072-dim vector
        Linear(3072 → hidden_dim) → ReLU → Dropout → Linear(hidden_dim → 1)
Output: raw logit → sigmoid → presence probability in [0, 1]
```

Empty columns are zeroed. The `n_cols` value is saved into the checkpoint for
backward compatibility. The score matrix is built efficiently by tiling embeddings
and flattening to a single batch rather than running N×M separate forward passes.

| | Cosine similarity | PresenceClassifier |
|--|--|--|
| Input | per-column embeddings, element-wise max | per-column embeddings concatenated (3072-dim + label) |
| Comparison | geometric angle | learned MLP |
| Training signal | none | supervised on keyword-confirmed (case, label) pairs |
| Per-column signal | preserved | preserved |

`column_scores.csv` always uses raw cosine regardless of which strategy is active, since
the classifier produces a single scalar per pair with no per-column breakdown.

### Step 5: Top-k Confidence Threshold

Labels sorted by score descending. Confidence threshold (default `0.6`) applied:

| Condition | Result | `method` |
|-----------|--------|----------|
| Score >= threshold | Include in ranked predictions (up to 5) | `embedding` |
| No label reaches threshold | Include top-1 only | `low_confidence` |
| All columns empty | No output row | `empty` |

### Step 6: Map Label Index to ICD Code

`labels/projection.py` resolves each index to term, group, and code.

### Step 7: Compute Per-Column Scores

Top-1 label and score recorded per column independently → `column_scores.csv`.

### Step 8: Write Outputs

Results written to `--out-dir` (see Output Files below).

---

## Worked Example

**Input row:**
```
case_id:                    "CASE-0003"
HISTOPATHOLOGICAL SUMMARY:  "...nodular coalescing cellular infiltrate...
                             histiocytes and small lymphocytes tightly surrounding vessels..."
FINAL COMMENT:              "The marked angiocentricity of the histiocytic and lymphocytic
                             infiltrate is consistent with reactive histiocytosis..."
ANCILLARY TESTS:            "1/15/2025: CD3: numerous small CD3+ T cells with strong expression."
```

**Top results after max across columns:**
1. "Histiocytic sarcoma" — 0.71 (FINAL COMMENT decisive)
2. "Reactive histiocytosis" — 0.67
3. "Cutaneous histiocytoma" — 0.61

All three pass the 0.6 threshold → 3 rows in `predictions.csv`.

---

## CLI Reference

| Option | Default | Description |
|--------|---------|-------------|
| `--csv` | `ml/data/report.csv` | Input CSV path |
| `--id-col` | `case_id` | Case ID column name |
| `--text-cols` | `HISTOPATHOLOGICAL SUMMARY,FINAL COMMENT,ANCILLARY TESTS` | Columns to embed independently |
| `--model` | `SAVSNET/PetBERT` | HuggingFace model name or local path |
| `--local-only` | off | Use only locally cached model files |
| `--out-dir` | `ml/output/report` | Output directory |
| `--max-rows` | all | Optional cap on rows to process |
| `--batch-size` | 16 | Texts per embedding batch |
| `--max-length` | 512 | Maximum token length (texts truncated beyond this) |
| `--embedding-min-sim` | 0.6 | Confidence threshold (use `0.05` when presence classifier is active) |
| `--device` | auto | `auto`, `cpu`, `cuda`, or `mps` |
| `--labels-csv` | `ml/labels/labels.csv` | Taxonomy CSV path |
| `--task` | `categorize` | `categorize`, `neighbors`, or `both` |
| `--neighbors-k` | 3 | Nearest neighbors per row |
| `--presence-classifier` | none | Path to trained `PresenceClassifier` checkpoint (`.pt`) |
| `--group-classifier` | none | Path to trained `GroupClassifier` checkpoint (`.pt`) |
| `--group-classifier-threshold` | 0.3 | Confidence threshold for group predictions |
| `--embedding-cache` | none | Path to embedding cache `.npz` |
| `--enrich-labels-csv` | none | Path to `keyword_predictions.csv` for label enrichment |
| `--finetuned-model-path` | none | Path to a fine-tuned PetBERT sequence classification checkpoint (WIP) |

---

## Example Commands

**Standard production run:**
```bash
ml/.venv/bin/python3 ml/scripts/run_pipeline.py --local-only
```

**With presence classifier and embedding cache:**
```bash
ml/.venv/bin/python3 ml/scripts/run_pipeline.py \
  --presence-classifier ml/model/checkpoints/presence_classifier_best.pt \
  --embedding-cache ml/data/embedding_cache.npz \
  --embedding-min-sim 0.05 \
  --local-only
```

**With group classifier:**
```bash
ml/.venv/bin/python3 ml/scripts/run_pipeline.py \
  --group-classifier ml/model/checkpoints/group_classifier_best.pt \
  --embedding-cache ml/data/embedding_cache.npz \
  --embedding-min-sim 0.05 \
  --local-only
```

**All report sections:**
```bash
ml/.venv/bin/python3 ml/scripts/run_pipeline.py \
  --text-cols "HISTOPATHOLOGICAL SUMMARY,FINAL COMMENT,ANCILLARY TESTS,ADDENDUM,CLINICAL ABSTRACT" \
  --local-only
```

**Quick test (first 50 rows):**
```bash
ml/.venv/bin/python3 ml/scripts/run_pipeline.py --max-rows 50 --local-only
```

---

## Output Files

| File | Rows | Purpose |
|------|------|---------|
| `petbert_predictions.csv` | One per (case, rank) | Up to 5 ranked predictions per case |
| `petbert_column_scores.csv` | One per (case × column) | Per-column similarity breakdown |
| `petbert_provenance.csv` | One per case | Traceability and debug info |
| `petbert_similarity_scores.csv` | One per case | Full score matrix (one column per taxonomy label) |
| `petbert_visualization.csv` | One per case | PCA 2-D coordinates for scatter plots |
| `petbert_embeddings.npz` | — | Raw 768-dim embedding vectors, ids, and texts |
| `petbert_summary.json` | — | Run metadata and aggregate prediction counts |

### `predictions.csv` columns

| Column | Description |
|--------|-------------|
| `case_id` | Case identifier |
| `diagnosis_index` | Rank (1 = best match, up to 5) |
| `predicted_term` | Taxonomy term |
| `predicted_group` | Tumor group |
| `predicted_code` | Vet-ICD-O-canine-1 code (blank when uncategorized) |
| `confidence` | Presence probability or cosine similarity |
| `method` | `embedding` or `low_confidence` |

### `column_scores.csv` columns

| Column | Description |
|--------|-------------|
| `case_id` | Case identifier |
| `column_name` | Report section (e.g. `FINAL COMMENT`) |
| `column_text` | Raw text content |
| `top_term` | Taxonomy label with highest cosine similarity from this column |
| `top_group` | Taxonomy group for `top_term` |
| `top_score` | Cosine similarity score |
| `was_decisive` | `True` if this column determined the final prediction |

### `provenance.csv` columns

| Column | Description |
|--------|-------------|
| `case_id` | Case identifier |
| `diagnosis_text` | Merged report text |
| `char_len` | Character length |
| `token_count` | Total BERT tokens across all embedded columns |
| `predicted_category` | Final label string (or "Uncategorized") |
| `embedding_category` | Raw top-1 label before confidence thresholding |
| `embedding_similarity` | Raw top-1 cosine similarity score |

---

## Dependencies

- `transformers` — HuggingFace model loading (PetBERT)
- `torch` — PyTorch forward pass
- `scikit-learn` — PCA for visualization
- `numpy` — array operations and cosine similarity
- `pandas` — CSV reading and DataFrame assembly

---

## Code Location

| File | Role |
|------|------|
| `ml/petbert_pipeline/pipeline.py` | Top-level orchestration (`run_scan`) |
| `ml/petbert_pipeline/embedding.py` | PetBERT loading, per-column embedding, fine-tuned classifier support |
| `ml/petbert_pipeline/embedding_cache.py` | Save/load the embedding cache |
| `ml/petbert_pipeline/categorization.py` | Top-k matching, confidence thresholding, group-based categorization |
| `ml/petbert_pipeline/types.py` | `ScanConfig` and `ScanOutputs` dataclasses |
| `ml/petbert_pipeline/utils.py` | Text cleaning, device selection |
| `ml/petbert_pipeline/io.py` | CSV/NPZ/JSON output writers |
| `ml/petbert_pipeline/cli.py` | CLI argument parsing |
| `ml/labels/taxonomy.py` | Vet-ICD-O taxonomy CSV parser |
| `ml/labels/catalog.py` | Label catalog builder |
| `ml/labels/projection.py` | Maps label index → term/group/code |
| `ml/labels/enrichment.py` | Optional label enrichment |
| `ml/model/presence_classifier.py` | `PresenceClassifier` MLP, `score_matrix()`, save/load |

---

## WIP: Fine-tuned PetBERT Mode

### Motivation

The binary `PresenceClassifier` and `GroupClassifier` both operate on frozen PetBERT
embeddings. Fine-tuning PetBERT end-to-end allows it to learn veterinary diagnostic
language from the task directly, without being limited by what frozen embeddings capture.

### Approach

```
Training:
    diagnosis text
        ↓
    keyword pipeline
        ↓
    group label
        ↓
    fine-tune PetBERT (AutoModelForSequenceClassification) on (report text, group label)
        ↓
    save HuggingFace checkpoint

Inference:
    report text
        ↓
    fine-tuned PetBERT
        ↓
    softmax over 45 groups
        ↓
    predicted group probabilities
        ↓
    cosine similarity within predicted group (using base PetBERT for label embeddings)
        ↓
    best term + ICD code
```

The keyword pipeline generates training labels from existing diagnosis text. At inference
the diagnosis field is not needed — the fine-tuned model predicts groups directly from
report text.

### Scripts

| Script | Role |
|--------|------|
| `ml/training/finetune/build_dataset.py` | Build HuggingFace `DatasetDict`; compute inverse-frequency class weights |
| `ml/training/finetune/train.py` | Fine-tune with `WeightedTrainer` (class-weighted CrossEntropyLoss) |

### Usage

```bash
# Step 1 — build dataset
ml/.venv/bin/python3 ml/training/finetune/build_dataset.py \
  --reports-csv database/data/output/report.csv \
  --predictions-csv ml/output/diagnoses/keyword_predictions.csv \
  --labels-csv ml/labels/labels.csv \
  --out-dir ml/data/finetune_dataset

# Step 2 — fine-tune
ml/.venv/bin/python3 ml/training/finetune/train.py \
  --dataset ml/data/finetune_dataset \
  --out-dir ml/model/checkpoints/petbert_finetuned \
  --epochs 5

# Step 3 — run pipeline with fine-tuned model
ml/.venv/bin/python3 ml/scripts/run_pipeline.py \
  --finetuned-model-path ml/model/checkpoints/petbert_finetuned \
  --local-only
```

### Known Issues

- **`WeightedTrainer` constructor order**: `__init__(self, class_weights=None, *args, **kwargs)` is fragile — if `model` is passed positionally it binds to `class_weights`. Should be `(self, *args, class_weights=None, **kwargs)`.
- **Device mismatch risk**: class weights moved to `self.args.device` during `__init__`, before device is resolved. Should be moved during `compute_loss`.
- **No stratified val split**: `train_test_split` called without `stratify_by_column="labels"` — rare groups may be underrepresented in validation.
- **`--finetuned-model-path` and `--presence-classifier` not mutually exclusive**: if both are set, the PresenceClassifier silently receives zero embeddings.
- **`evaluation_strategy`/`save_strategy` deprecated**: newer `transformers` versions use `eval_strategy`.
- **No `local_files_only` in `build_dataset.py`**: tokenizer will attempt a network call.

### Status

Not yet benchmarked. Pipeline integration is functional but the known issues above should
be resolved before a full training run.
