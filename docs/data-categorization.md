# Data Categorization Pipeline

This document describes how the pipeline maps free-text clinical diagnosis
strings to standardized Vet-ICD-O-canine-1 codes using the PetBERT language
model.  Each input row produces:

- **Term** -- the specific diagnosis name (e.g. "Hemangiosarcoma, NOS")
- **Group** -- the tumor category (e.g. "Blood vessel tumors")
- **Code** -- the ICD-O morphology code (e.g. "9120/3")

---

## How It Works (High-Level)

The pipeline does **not** fine-tune or train any model.  Instead it uses
PetBERT purely as a **feature extractor** and compares embeddings via cosine
similarity:

1. Every clinical diagnosis string is passed through PetBERT to get a
   768-dimensional embedding vector.
2. Every taxonomy label is also passed through PetBERT to get a
   768-dimensional embedding vector.
3. For each diagnosis, the taxonomy label whose embedding is most similar
   (by cosine similarity) is selected as the prediction.
4. That label's term, group, and code become the output.

There is no classification head, no softmax, and no training loop.  The
"evaluation" is a nearest-neighbor lookup in embedding space.

---

## Pipeline Flow (Step by Step)

The entry point is `run_scan()` in `ml/petbert_scan/pipeline.py`.

### Step 1: Load and Clean Input Data

```
Input CSV  -->  pandas.read_csv(encoding='latin-1')
```

- The CSV (default `ml/data/data.csv`) must have an `anon_id` column and a
  `Clinical Diagnoses` column.
- Each cell is cleaned: whitespace is stripped, `NaN` values are replaced with
  empty strings (`clean_text()` in `utils.py`).
- The result is two parallel lists: `ids` and `texts`.

### Step 2: Embed Diagnosis Texts with PetBERT

```
"Multicentric lymphoma, ..." --[tokenize]--> [101, 2093, ...] --[PetBERT]--> [0.12, -0.34, ..., 0.07]
                                                                               ^^^^^^^^^^^^^^^^^^^^^^^^
                                                                               768-dimensional vector
```

**Model:** `SAVSNET/PetBERT` -- a BERT-style masked language model pre-trained
on veterinary clinical text from the SAVSNET project.  It is loaded via
HuggingFace `transformers` (`AutoModelForMaskedLM`).

**How data is passed into the model** (`embedding.py: embed_texts`):

1. **Tokenization** -- The raw text string is fed to PetBERT's tokenizer with
   `padding=True`, `truncation=True`, `max_length=256`.  This converts the
   text into integer token IDs plus an attention mask that marks real tokens
   vs. padding.  Texts are processed in batches (default 16).

2. **Forward pass** -- The token IDs are passed through PetBERT's **base
   transformer** (not the masked-LM prediction head):
   ```python
   outputs = model.base_model(input_ids=input_ids, attention_mask=attention_mask)
   ```
   The masked-LM head is skipped because we don't need word predictions --
   we only want the hidden-state representation.

3. **[CLS] token extraction** -- The output `last_hidden_state` has shape
   `(batch, seq_len, 768)`.  We take position 0 (the **[CLS]** special token)
   as the fixed-size representation of the entire input:
   ```python
   cls_embedding = outputs.last_hidden_state[:, 0, :]  # shape: (batch, 768)
   ```
   This is standard practice for BERT-family models: the [CLS] token is
   designed to aggregate information from the full sequence.

**Output:** A NumPy array of shape `(num_rows, 768)` -- one embedding per
diagnosis row.

### Step 3: Build and Embed Taxonomy Labels

The taxonomy file `ml/labels/labels.csv` contains ~857 unique entries from the
Vet-ICD-O-canine-1 coding system.  Each entry has a `code`, `group`, and
`term` (parsed by `labels/taxonomy.py`).

To make labels comparable to diagnosis text, each label is converted into a
natural-language sentence:

```
"Veterinary diagnosis term: Hemangiosarcoma, NOS. Group: Blood vessel tumors. Code: 9120/3."
```

These sentences are then embedded through the **exact same** PetBERT model
and tokenizer, producing a `(num_labels, 768)` matrix.  This ensures
diagnosis embeddings and label embeddings live in the same vector space.

### Step 4: Cosine Similarity Matching

```
                     Label 0    Label 1    ...    Label 856
Diagnosis 0        [  0.72       0.45      ...      0.31   ]  <-- pick argmax
Diagnosis 1        [  0.38       0.81      ...      0.29   ]
...
Diagnosis N        [  0.55       0.41      ...      0.67   ]
```

For each diagnosis embedding, we compute its **cosine similarity** against
every label embedding (`embedding.py: cosine_similarity_matrix`):

```
cosine_sim(a, b) = dot(a, b) / (||a|| * ||b||)
```

This produces an `(N, M)` similarity matrix where `N` = number of diagnosis
rows and `M` = number of taxonomy labels.  For each row, the label with the
**highest** cosine similarity is selected:

```python
pred_idx = np.argmax(sims, axis=1)       # index of best label
pred_score = sims[row, pred_idx]          # similarity score (0-1)
```

### Step 5: Confidence Threshold

Not all matches are good.  A confidence threshold (default `0.6`) is applied:

| Condition | Result | `category_method` |
|-----------|--------|-------------------|
| Text is empty | No prediction | `empty` |
| Best score >= 0.6 | Use the top label | `embedding` |
| Best score < 0.6 | Mark as "Uncategorized" | `low_confidence` |

Even for `low_confidence` rows, the closest label index is still recorded so
it can be reviewed manually.

### Step 6: Auxiliary Label Override (Optional)

If `--use-auxiliary-labels` is enabled, patients known to have carcinoma or
sarcoma (from external CSV lists of patient IDs) get their prediction
**constrained**:

1. Load patient ID sets from `dataCarcinoma.csv` and `dataSarcoma.csv`.
2. Find all taxonomy labels whose term contains "carcinoma" or "sarcoma".
3. For a patient in the carcinoma set: ignore all non-carcinoma labels and
   pick the highest-scoring carcinoma label instead.
4. Same logic for sarcoma patients.
5. If a patient appears in **both** lists: mark as "conflict" and keep the
   unconstrained prediction.

This is a form of semi-supervised correction -- it uses external knowledge to
steer predictions toward the right tumor family.

### Step 7: Map Label Index to ICD Code

The categorization steps produce an integer index into the taxonomy list.
`labels/projection.py` resolves each index to the final output fields:

```python
taxonomy_labels[idx].term   -->  predicted_term   (e.g. "Hemangiosarcoma, NOS")
taxonomy_labels[idx].group  -->  predicted_group   (e.g. "Blood vessel tumors")
taxonomy_labels[idx].code   -->  predicted_code    (e.g. "9120/3")
```

### Step 8: Write Outputs

Results are written to the `--out-dir` directory (see Output Files below).

---

## Worked Example

**Input row:**
```
anon_id: "ID_42"
Clinical Diagnoses: "Multicentric lymphoma stage IVa"
```

**Processing:**
1. Text cleaned -> `"Multicentric lymphoma stage IVa"`
2. Tokenized (22 tokens) and embedded via PetBERT -> 768-dim vector
3. Cosine similarity computed against all 857 label embeddings
4. Closest label: index 312, term "Lymphoma, NOS", score 0.78
5. Score 0.78 >= 0.6 threshold -> accepted (method: `embedding`)
6. No auxiliary override (ID_42 not in carcinoma/sarcoma lists)
7. Taxonomy lookup: term="Lymphoma, NOS", group="Malignant lymphomas", code="9590/3"

**Output:**
| Column | Value |
|--------|-------|
| `predicted_term` | Lymphoma, NOS |
| `predicted_group` | Malignant lymphomas |
| `predicted_code` | 9590/3 |
| `category_confidence` | 0.78 |
| `category_method` | embedding |

---

## Code Location

| File | Role |
|------|------|
| `ml/petbert_scan/pipeline.py` | Top-level orchestration (`run_scan`) |
| `ml/petbert_scan/embedding.py` | PetBERT loading, text embedding, cosine similarity |
| `ml/petbert_scan/categorization.py` | Similarity matching and confidence thresholding |
| `ml/petbert_scan/auxiliary_policy.py` | Carcinoma/sarcoma auxiliary override logic |
| `ml/petbert_scan/types.py` | `ScanConfig` and `ScanOutputs` dataclasses |
| `ml/petbert_scan/utils.py` | Text cleaning, device selection |
| `ml/petbert_scan/io.py` | CSV/NPZ/JSON output writers |
| `ml/petbert_scan/cli.py` | Command-line argument parsing |
| `ml/labels/taxonomy.py` | Vet-ICD-O taxonomy CSV parser |
| `ml/labels/catalog.py` | Label catalog builder (label text generation) |
| `ml/labels/projection.py` | Maps label index -> term/group/code |
| `ml/labels/auxiliary.py` | Helpers for auxiliary label constraints |

## Inputs

- **Main input dataset:**
  - `ml/data/data.csv`
  - Required columns: `anon_id`, `Clinical Diagnoses`
- **Taxonomy label source:**
  - `ml/labels/labels.csv` (Vet-ICD-O-canine-1, ~857 unique terms)
- **Auxiliary supervision datasets** (optional):
  - `ml/data/dataCarcinoma.csv` -- patient IDs known to have carcinoma
  - `ml/data/dataSarcoma.csv` -- patient IDs known to have sarcoma

## Output Files

Default outputs are written to the configured `--out-dir`:

| File | Description |
|------|-------------|
| `petbert_scan_rows.csv` | Full row-level output: input columns, predictions, confidence, method, PCA coordinates |
| `petbert_scan_categories.csv` | Classification-focused output with taxonomy mapping and per-label similarity scores |
| `petbert_scan_embeddings.npz` | Compressed NumPy archive with the raw 768-dim embedding vectors, ids, and texts |
| `petbert_scan_summary.json` | Run metadata and aggregate counts (term/group/code distributions, method counts) |

**Key output columns:**

| Column | Description |
|--------|-------------|
| `predicted_term` | Taxonomy term name (e.g. "Hemangiosarcoma, NOS") |
| `predicted_group` | Tumor category (e.g. "Blood vessel tumors") |
| `predicted_code` | Vet-ICD-O-canine-1 code (e.g. "9120/3") |
| `category_confidence` | Cosine similarity score (0.0 - 1.0) |
| `category_method` | How the prediction was made (see table in Step 5 above) |
| `auxiliary_label` | "carcinoma", "sarcoma", "conflict", or "" |
| `predicted_label_index` | Integer index into the taxonomy list |

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

## Dependencies

From `ml/requirements.txt`:

- `transformers==4.38.1` -- HuggingFace model loading (PetBERT)
- `torch==2.2.0` -- PyTorch (neural network forward pass)
- `scikit-learn==1.4.0` -- PCA for 2-D visualization
- `numpy==1.26.4` -- Array operations and cosine similarity math
- `pandas==2.2.0` -- CSV reading and DataFrame assembly

## Example Commands

```bash
ml/.venv11/bin/python ml/scripts/petbert_scan.py \
  --csv ml/data/data.csv \
  --id-col anon_id \
  --text-col "Clinical Diagnoses" \
  --labels-csv ml/labels/labels.csv \
  --use-auxiliary-labels \
  --carcinoma-csv ml/data/dataCarcinoma.csv \
  --sarcoma-csv ml/data/dataSarcoma.csv \
  --task categorize \
  --out-dir ml/output/data \
  --local-only
```

```bash
ml/.venv11/bin/python ml/scripts/petbert_scan.py \
  --csv ml/data/dataSarcomaComplete.csv \
  --id-col anon_id \
  --text-col "Clinical Diagnoses" \
  --labels-csv ml/labels/labels.csv \
  --task categorize \
  --out-dir ml/output/dataSarcoma\
  --local-only
```

```bash
ml/.venv11/bin/python ml/scripts/petbert_test.py \
  --csv ml/output/dataSarcoma/petbert_scan_categories.csv \
  --keyword sarcoma
```
