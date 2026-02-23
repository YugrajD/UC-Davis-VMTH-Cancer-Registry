# Data Categorization Pipeline — Usage

How to run the pipeline, what it expects as input, and what it produces as
output.  For a technical explanation of how the pipeline works see
[data-categorization.md](data-categorization.md).

---

## Inputs

- **Main input dataset:**
  - `ml/data/data.csv`
  - Required columns: `anon_id`, `Clinical Diagnoses`
- **Taxonomy label source:**
  - `ml/labels/labels.csv` (Vet-ICD-O-canine-1, ~857 unique terms)
- **Auxiliary supervision datasets** (optional):
  - `ml/data/dataCarcinoma.csv` -- patient IDs known to have carcinoma
  - `ml/data/dataSarcoma.csv` -- patient IDs known to have sarcoma

---

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--csv` | `ml/data/data.csv` | Input CSV path |
| `--id-col` | `anon_id` | Name of the patient ID column |
| `--text-col` | `Clinical Diagnoses` | Name of the diagnosis text column |
| `--model` | `SAVSNET/PetBERT` | HuggingFace model name or local path |
| `--local-only` | off | Use only locally cached model files |
| `--out-dir` | `ml/output` | Output directory |
| `--max-rows` | all | Optional cap on number of rows to process |
| `--batch-size` | 16 | Number of texts to embed at once |
| `--max-length` | 256 | Maximum token length (texts are truncated beyond this) |
| `--embedding-min-sim` | 0.6 | Confidence threshold for accepting a prediction |
| `--device` | auto | Compute device: `auto`, `cpu`, `cuda`, or `mps` |
| `--labels-csv` | `ml/labels/labels.csv` | Path to the taxonomy CSV |
| `--task` | `categorize` | `categorize`, `neighbors`, or `both` |
| `--neighbors-k` | 3 | Number of nearest neighbors per row (when task includes neighbors) |
| `--use-auxiliary-labels` | off | Enable carcinoma/sarcoma auxiliary override |
| `--carcinoma-csv` | `ml/data/dataCarcinoma.csv` | CSV of carcinoma patient IDs |
| `--sarcoma-csv` | `ml/data/dataSarcoma.csv` | CSV of sarcoma patient IDs |

---

## Example Commands

**Basic run** -- uses all defaults (`ml/data/data.csv`, PetBERT, threshold 0.6):
```bash
ml/.venv11/bin/python ml/scripts/petbert_scan.py --local-only
```

**Custom output directory:**
```bash
ml/.venv11/bin/python ml/scripts/petbert_scan.py \
  --out-dir ml/output/data \
  --local-only
```

**With auxiliary label overrides** -- constrains known carcinoma/sarcoma patients
to matching taxonomy labels:
```bash
ml/.venv11/bin/python ml/scripts/petbert_scan.py \
  --use-auxiliary-labels \
  --out-dir ml/output/data \
  --local-only
```

**Different dataset** -- run on the full sarcoma dataset with a stricter
confidence threshold:
```bash
ml/.venv11/bin/python ml/scripts/petbert_scan.py \
  --csv ml/data/dataSarcomaComplete.csv \
  --embedding-min-sim 0.7 \
  --out-dir ml/output/dataSarcoma \
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
| `petbert_scan_predictions.csv` | One per original patient | Presentation-ready results with concatenated multi-diagnosis predictions |
| `petbert_scan_provenance.csv` | One per sub-diagnosis | Traceability and debug info: text stats, auxiliary labels, raw ML scores |
| `petbert_scan_similarity_scores.csv` | One per sub-diagnosis | Full cosine similarity matrix (one column per taxonomy label) |
| `petbert_scan_visualization.csv` | One per sub-diagnosis | PCA coordinates for 2-D plotting |
| `petbert_scan_embeddings.npz` | N/A | Compressed NumPy archive with the raw 768-dim embedding vectors, ids, and texts |
| `petbert_scan_summary.json` | N/A | Run metadata and aggregate counts (term/group/code distributions, method counts) |

### `predictions.csv` columns

For multi-diagnosis patients, values are concatenated with `1)`, `2)` prefixes.
Single-diagnosis patients show plain values with no prefix.  `predicted_code`
is blank for uncategorized predictions.

| Column | Description |
|--------|-------------|
| `anon_id` | Patient identifier |
| `original_text` | The full unsplit clinical text |
| `predicted_term` | Taxonomy term(s) |
| `predicted_group` | Tumor group(s) |
| `predicted_code` | Vet-ICD-O-canine-1 code(s), blank when uncategorized |
| `confidence` | Cosine similarity score(s) |
| `method` | Classification method(s): `embedding`, `low_confidence`, or `empty` |

### `provenance.csv` columns

One row per sub-diagnosis.  Keyed by `row_index` + `diagnosis_index` so it can
be joined with the other per-sub-diagnosis files.

| Column | Description |
|--------|-------------|
| `row_index` | Index of the original CSV row (0-based) |
| `anon_id` | Patient identifier (matches the predictions CSV) |
| `diagnosis_index` | Sub-diagnosis position within the original entry (1-based) |
| `diagnosis_text` | The individual sub-diagnosis text that was embedded |
| `char_len` | Character length of the sub-diagnosis text |
| `token_count` | Number of BERT tokens after tokenization |
| `predicted_category` | Final label string (or "Uncategorized" / "") |
| `auxiliary_label` | "carcinoma", "sarcoma", "conflict", or "" |
| `predicted_label_index` | Integer index into the taxonomy list |
| `keyword_category` | Reserved for future keyword classifier (empty) |
| `keyword_confidence` | Reserved for future keyword scores (0.0) |
| `embedding_category` | Raw top-1 label before confidence thresholding |
| `embedding_similarity` | Raw top-1 cosine similarity score |

### `similarity_scores.csv` columns

One row per sub-diagnosis, keyed by `row_index` + `diagnosis_index`.  Contains
one `score_*` column per taxonomy label (~857 columns) with the cosine
similarity between the sub-diagnosis embedding and that label's embedding.

### `visualization.csv` columns

| Column | Description |
|--------|-------------|
| `row_index` | Index of the original CSV row (0-based) |
| `diagnosis_index` | Sub-diagnosis position (1-based) |
| `anon_id` | Patient identifier |
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

### Grouping modes (`--group-by`)

| Mode | Groups by | A match means... |
|------|-----------|------------------|
| `visit` (default) | `row_index` | Any sub-diagnosis of a single visit contains the keyword |
| `patient` | `anon_id` | Any prediction across **all** visits for the same patient contains the keyword |

The `patient` mode is useful when a single patient may have multiple visits in
the input CSV.  If *any* of that patient's predictions contain the keyword, the
patient counts as a match.

### How it works

1. Load the provenance CSV.
2. Drop rows where `diagnosis_text` is empty.
3. For each row, check whether `predicted_category` contains the keyword
   (case-insensitive).
4. Group by `row_index` (visit mode) or `anon_id` (patient mode) so that
   multi-diagnosis entries count as **one** match if *any* sub-diagnosis hits.
5. Report the number of valid visits/patients, matches, and success rate.

### CLI options

| Option | Default | Description |
|--------|---------|-------------|
| `--csv` | `ml/output/data/petbert_scan_provenance.csv` | Path to the provenance CSV to evaluate |
| `--keyword` | `sarcoma` | Target keyword to search for in `predicted_category` |
| `--group-by` | `visit` | Grouping mode: `visit` (by row_index) or `patient` (by anon_id) |
| `--id-col` | `anon_id` | Column name for patient ID (only used in patient mode) |

### Examples

**Evaluate sarcoma predictions** (default -- group by visit):
```bash
ml/.venv11/bin/python ml/scripts/petbert_test.py
```

**Evaluate grouped by patient:**
```bash
ml/.venv11/bin/python ml/scripts/petbert_test.py --group-by patient
```

**Evaluate with a different keyword:**
```bash
ml/.venv11/bin/python ml/scripts/petbert_test.py \
  --keyword lymphoma
```

### Sample output

Visit mode:
```
Valid visits: 92838
Matches:      78412
Success rate: 84.46%
```

Patient mode:
```
Total patients: 45219
Matches:        39102
Success rate:   86.47%
```
