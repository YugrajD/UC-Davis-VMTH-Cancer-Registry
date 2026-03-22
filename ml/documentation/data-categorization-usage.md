# Data Categorization Pipeline — Usage

How to run the pipeline, what it expects as input, and what it produces as
output.  For a technical explanation of how the pipeline works see
[data-categorization.md](data-categorization.md).

---

## Inputs

- **Main input dataset:**
  - `ml/data/report.csv`
  - ID column: `case_id`
  - Text columns (specify via `--text-cols`): `HISTOPATHOLOGICAL SUMMARY`,
    `FINAL COMMENT`, `ANCILLARY TESTS`, `ADDENDUM`, `CLINICAL ABSTRACT`,
    `GROSS DESCRIPTION`, etc.
- **Taxonomy label source:**
  - `ml/labels/labels.csv` (Vet-ICD-O-canine-1, ~857 unique terms)
- **Optional — presence classifier checkpoint:**
  - `ml/model/checkpoints/presence_classifier_best.pt`
  - Replaces cosine similarity scores with learned presence probabilities.
  - Trained via `ml/scripts/run_training.py --mode binary`; see `classifier.md`.
- **Optional — keyword predictions (for label enrichment):**
  - `ml/output/diagnoses/keyword_predictions.csv`
  - Used with `--enrich-labels-csv` to enrich label embeddings.

---

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--csv` | `ml/data/report.csv` | Input CSV path |
| `--id-col` | `case_id` | Name of the case ID column |
| `--text-cols` | `HISTOPATHOLOGICAL SUMMARY,FINAL COMMENT,ANCILLARY TESTS` | Comma-separated column names to embed independently. Each column is embedded separately and the best similarity across any column wins per case. |
| `--col-weights` | `FINAL COMMENT:2.0,...` | Reserved — not used for scoring |
| `--model` | `SAVSNET/PetBERT` | HuggingFace model name or local path |
| `--local-only` | off | Use only locally cached model files |
| `--out-dir` | `ml/output/report` | Output directory |
| `--max-rows` | all | Optional cap on number of rows to process |
| `--batch-size` | 16 | Number of texts to embed at once |
| `--max-length` | 512 | Maximum token length (texts are truncated beyond this) |
| `--embedding-min-sim` | 0.6 | Confidence threshold for accepting a prediction (use 0.05 when the presence classifier is active — scores are then probabilities, not raw cosine similarities) |
| `--device` | auto | Compute device: `auto`, `cpu`, `cuda`, or `mps` |
| `--labels-csv` | `ml/labels/labels.csv` | Path to the taxonomy CSV |
| `--task` | `categorize` | `categorize`, `neighbors`, or `both` |
| `--neighbors-k` | 3 | Number of nearest neighbors per row (when task includes neighbors) |
| `--presence-classifier` | none | Path to a trained `PresenceClassifier` checkpoint (`.pt`). When set, presence probabilities replace raw cosine similarities for label scoring. |
| `--embedding-cache` | none | Path to an embedding cache `.npz`. If the cache is valid, PetBERT is skipped entirely. On a cache miss the embeddings are computed and saved here for future runs. |
| `--enrich-labels-csv` | none | Path to `keyword_predictions.csv`. Each label embedding is averaged 50/50 with the mean cached report embedding of its keyword-confirmed cases, pulling label representations toward real clinical language. Requires `--embedding-cache`. |

---

## Example Commands

**Basic run** — uses all defaults (`ml/data/report.csv`, top 3 sections, PetBERT, threshold 0.6):
```bash
ml/.venv/bin/python3 -m petbert_pipeline --local-only
```

**With presence classifier and embedding cache** (standard mode for trained pipeline):
```bash
ml/.venv/bin/python3 -m petbert_pipeline \
  --presence-classifier ml/model/checkpoints/presence_classifier_best.pt \
  --embedding-cache ml/data/embedding_cache.npz \
  --embedding-min-sim 0.05 \
  --enrich-labels-csv ml/output/diagnoses/keyword_predictions.csv \
  --local-only
```

**All available sections** — include addendum and clinical abstract in addition to the defaults:
```bash
ml/.venv/bin/python3 -m petbert_pipeline \
  --text-cols "HISTOPATHOLOGICAL SUMMARY,FINAL COMMENT,ANCILLARY TESTS,ADDENDUM,CLINICAL ABSTRACT" \
  --local-only
```

**Stricter confidence threshold** (cosine-only, no classifier):
```bash
ml/.venv/bin/python3 -m petbert_pipeline \
  --embedding-min-sim 0.7 \
  --local-only
```

**Quick test run** — process only the first 50 rows:
```bash
ml/.venv/bin/python3 -m petbert_pipeline \
  --max-rows 50 \
  --local-only
```

**Include nearest-neighbor output** alongside categorization:
```bash
ml/.venv/bin/python3 -m petbert_pipeline \
  --task both \
  --neighbors-k 5 \
  --local-only
```

---

## Output Files

Default outputs are written to the configured `--out-dir`.  Each file serves a
specific purpose so the data is easy to read and work with:

| File | Rows | Purpose |
|------|------|---------|
| `petbert_predictions.csv` | One per (case, prediction rank) | Presentation-ready results — up to 5 ranked predictions per case |
| `petbert_column_scores.csv` | One per (case × column) | Per-column similarity breakdown — shows which section drove each prediction |
| `petbert_provenance.csv` | One per case | Traceability and debug info: text stats and raw ML scores |
| `petbert_similarity_scores.csv` | One per case | Full score matrix (one column per taxonomy label); values are presence probabilities when classifier is active, cosine similarities otherwise |
| `petbert_visualization.csv` | One per case | PCA coordinates for 2-D plotting |
| `petbert_embeddings.npz` | N/A | Compressed NumPy archive with the raw 768-dim embedding vectors, ids, and texts |
| `petbert_summary.json` | N/A | Run metadata and aggregate counts (term/group/code distributions, method counts) |

### `predictions.csv` columns

| Column | Description |
|--------|-------------|
| `case_id` | Case identifier |
| `diagnosis_index` | Rank of this prediction for the case (1 = best match, up to 5) |
| `predicted_term` | Taxonomy term |
| `predicted_group` | Tumor group |
| `predicted_code` | Vet-ICD-O-canine-1 code, blank when uncategorized |
| `confidence` | Score (presence probability when classifier is active; cosine similarity otherwise) |
| `method` | Classification method: `embedding` or `low_confidence` |

Cases with no text in any of the selected columns produce no rows.

### `column_scores.csv` columns

| Column | Description |
|--------|-------------|
| `row_index` | Index of the original CSV row (0-based) |
| `case_id` | Case identifier |
| `column_name` | Which report section (e.g. `FINAL COMMENT`) |
| `column_text` | Raw text content of this section for this case |
| `top_term` | Taxonomy label with the highest cosine similarity from this column's embedding alone |
| `top_group` | Taxonomy group for `top_term` |
| `top_code` | ICD-O code for `top_term` |
| `top_score` | Cosine similarity score (this column vs. `top_term`) |
| `was_decisive` | `True` if this column had the highest score across all columns and determined the final prediction |

### `provenance.csv` columns

| Column | Description |
|--------|-------------|
| `row_index` | Index of the original CSV row (0-based) |
| `case_id` | Case identifier |
| `diagnosis_index` | Always 1 (one record per original case) |
| `diagnosis_text` | The merged report text for this case |
| `char_len` | Character length of the merged text |
| `token_count` | Total BERT tokens across all embedded columns |
| `predicted_category` | Final label string (or "Uncategorized" / "") |
| `predicted_label_index` | Integer index into the taxonomy list |
| `embedding_category` | Raw top-1 label before confidence thresholding |
| `embedding_similarity` | Raw top-1 cosine similarity score |

### `similarity_scores.csv` columns

One row per case, keyed by `row_index` + `diagnosis_index`.  Contains one
`score_*` column per taxonomy label (~857 columns) with the max cosine
similarity across all text columns for that label (or presence probabilities
when the classifier is active).

### `visualization.csv` columns

| Column | Description |
|--------|-------------|
| `row_index` | Index of the original CSV row (0-based) |
| `diagnosis_index` | Always 1 (one record per original case) |
| `case_id` | Case identifier |
| `predicted_group` | Tumor group (useful for coloring points) |
| `pca1` | First PCA component |
| `pca2` | Second PCA component |

---

## Dependencies

From `ml/requirements.txt`:

- `transformers` -- HuggingFace model loading (PetBERT)
- `torch` -- PyTorch (neural network forward pass)
- `scikit-learn` -- PCA for 2-D visualization
- `numpy` -- Array operations and cosine similarity math
- `pandas` -- CSV reading and DataFrame assembly
