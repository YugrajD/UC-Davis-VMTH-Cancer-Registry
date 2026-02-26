# Data Categorization Pipeline — Usage

How to run the pipeline, what it expects as input, and what it produces as
output.  For a technical explanation of how the pipeline works see
[data-categorization.md](data-categorization.md).

---

## Inputs

- **Main input dataset:**
  - `ml/data/reportText.csv`
  - ID column: `case_id`
  - Text columns (specify via `--text-cols`): `HISTOPATHOLOGICAL SUMMARY`,
    `FINAL COMMENT`, `ANCILLARY TESTS`, `ADDENDUM`, `CLINICAL ABSTRACT`,
    `GROSS DESCRIPTION`, etc.
- **Taxonomy label source:**
  - `ml/labels/labels.csv` (Vet-ICD-O-canine-1, ~857 unique terms)

---

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--csv` | `ml/data/reportText.csv` | Input CSV path |
| `--id-col` | `case_id` | Name of the case ID column |
| `--text-cols` | `HISTOPATHOLOGICAL SUMMARY,FINAL COMMENT,ANCILLARY TESTS` | Comma-separated column names to embed independently and weighted-average. Each non-empty column is prefixed with its name so the model sees section labels. |
| `--col-weights` | `FINAL COMMENT:2.0,HISTOPATHOLOGICAL SUMMARY:1.5,ANCILLARY TESTS:0.5` | Per-column embedding weights as `COL:weight,...` pairs. Columns absent from this list default to 1.0. |
| `--model` | `SAVSNET/PetBERT` | HuggingFace model name or local path |
| `--local-only` | off | Use only locally cached model files |
| `--out-dir` | `ml/output/reportText` | Output directory |
| `--max-rows` | all | Optional cap on number of rows to process |
| `--batch-size` | 16 | Number of texts to embed at once |
| `--max-length` | 256 | Maximum token length (texts are truncated beyond this) |
| `--embedding-min-sim` | 0.6 | Confidence threshold for accepting a prediction |
| `--device` | auto | Compute device: `auto`, `cpu`, `cuda`, or `mps` |
| `--labels-csv` | `ml/labels/labels.csv` | Path to the taxonomy CSV |
| `--task` | `categorize` | `categorize`, `neighbors`, or `both` |
| `--neighbors-k` | 3 | Number of nearest neighbors per row (when task includes neighbors) |

---

## Example Commands

**Basic run** -- uses all defaults (`ml/data/reportText.csv`, top 3 sections, PetBERT, threshold 0.6):
```bash
ml/.venv11/bin/python ml/scripts/petbert_scan.py --local-only
```

**All available sections** -- include addendum and clinical abstract in addition to the defaults:
```bash
ml/.venv11/bin/python ml/scripts/petbert_scan.py \
  --text-cols "HISTOPATHOLOGICAL SUMMARY,FINAL COMMENT,ANCILLARY TESTS,ADDENDUM,CLINICAL ABSTRACT" \
  --local-only
```

**Stricter confidence threshold:**
```bash
ml/.venv11/bin/python ml/scripts/petbert_scan.py \
  --embedding-min-sim 0.7 \
  --local-only
```

**Quick test run** -- process only the first 50 rows:
```bash
ml/.venv11/bin/python ml/scripts/petbert_scan.py \
  --max-rows 50 \
  --local-only
```

**Include nearest-neighbor output** alongside categorization:
```bash
ml/.venv11/bin/python ml/scripts/petbert_scan.py \
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
| `petbert_scan_predictions.csv` | One per original case | Presentation-ready results |
| `petbert_scan_provenance.csv` | One per sub-diagnosis | Traceability and debug info: text stats and raw ML scores |
| `petbert_scan_similarity_scores.csv` | One per sub-diagnosis | Full cosine similarity matrix (one column per taxonomy label) |
| `petbert_scan_visualization.csv` | One per sub-diagnosis | PCA coordinates for 2-D plotting |
| `petbert_scan_embeddings.npz` | N/A | Compressed NumPy archive with the raw 768-dim embedding vectors, ids, and texts |
| `petbert_scan_summary.json` | N/A | Run metadata and aggregate counts (term/group/code distributions, method counts) |

### `predictions.csv` columns

| Column | Description |
|--------|-------------|
| `case_id` | Case identifier |
| `original_text` | The FINAL COMMENT text for this case |
| `predicted_term` | Taxonomy term |
| `predicted_group` | Tumor group |
| `predicted_code` | Vet-ICD-O-canine-1 code, blank when uncategorized |
| `confidence` | Cosine similarity score |
| `method` | Classification method: `embedding`, `low_confidence`, or `empty` |

### `provenance.csv` columns

| Column | Description |
|--------|-------------|
| `row_index` | Index of the original CSV row (0-based) |
| `case_id` | Case identifier |
| `diagnosis_index` | Sub-diagnosis position within the case (always 1 for report format) |
| `diagnosis_text` | The merged report text that was embedded |
| `char_len` | Character length of the merged text |
| `token_count` | Number of BERT tokens after tokenization |
| `predicted_category` | Final label string (or "Uncategorized" / "") |
| `predicted_label_index` | Integer index into the taxonomy list |
| `embedding_category` | Raw top-1 label before confidence thresholding |
| `embedding_similarity` | Raw top-1 cosine similarity score |

### `similarity_scores.csv` columns

One row per sub-diagnosis, keyed by `row_index` + `diagnosis_index`.  Contains
one `score_*` column per taxonomy label (~857 columns) with the cosine
similarity between the case embedding and that label's embedding.

### `visualization.csv` columns

| Column | Description |
|--------|-------------|
| `row_index` | Index of the original CSV row (0-based) |
| `diagnosis_index` | Sub-diagnosis position within the case |
| `case_id` | Case identifier |
| `predicted_group` | Tumor group (useful for coloring points) |
| `pca1` | First PCA component |
| `pca2` | Second PCA component |

---

## Dependencies

From `ml/requirements.txt`:

- `transformers==4.38.1` -- HuggingFace model loading (PetBERT)
- `torch==2.2.0` -- PyTorch (neural network forward pass)
- `scikit-learn==1.4.0` -- PCA for 2-D visualization
- `numpy==1.26.4` -- Array operations and cosine similarity math
- `pandas==2.2.0` -- CSV reading and DataFrame assembly

---

## Testing Predictions

The `petbert_test.py` script (`ml/scripts/petbert_test.py`) evaluates how well
the pipeline's predictions match a target keyword.  It reads the provenance
output and reports what fraction of entries had at least one sub-diagnosis whose
`predicted_category` contains the keyword.

### CLI options

| Option | Default | Description |
|--------|---------|-------------|
| `--csv` | `ml/output/reportText/petbert_scan_provenance.csv` | Path to the provenance CSV to evaluate |
| `--keyword` | `sarcoma` | Target keyword to search for in `predicted_category` |
| `--group-by` | `visit` | Grouping mode: `visit` (by row_index) or `patient` (by case_id) |
| `--id-col` | `case_id` | Column name for case ID (only used in patient mode) |

### Examples

**Evaluate sarcoma predictions:**
```bash
ml/.venv11/bin/python ml/scripts/petbert_test.py \
  --csv ml/output/reportText/petbert_scan_provenance.csv \
  --id-col case_id
```

**Evaluate with a different keyword:**
```bash
ml/.venv11/bin/python ml/scripts/petbert_test.py \
  --csv ml/output/reportText/petbert_scan_provenance.csv \
  --keyword lymphoma \
  --id-col case_id
```
