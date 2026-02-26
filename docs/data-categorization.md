# Data Categorization Pipeline — How It Works

This document describes how the pipeline maps full veterinary pathology reports
to standardized Vet-ICD-O-canine-1 codes using the PetBERT language model.
Each selected report section is embedded **independently** and the resulting
vectors are combined into a single weighted-average embedding, which is then
matched against the taxonomy to produce:

- **Term** -- the specific diagnosis name (e.g. "Hemangiosarcoma, NOS")
- **Group** -- the tumor category (e.g. "Blood vessel tumors")
- **Code** -- the ICD-O morphology code (e.g. "9120/3")

For CLI options, output file formats, and example commands see
[data-categorization-usage.md](data-categorization-usage.md) (in `ml/scripts/`).

---

## How It Works (High-Level)

The pipeline does **not** fine-tune or train any model.  Instead it uses
PetBERT purely as a **feature extractor** and compares embeddings via cosine
similarity:

1. Each selected column from `reportText.csv` (e.g. `HISTOPATHOLOGICAL
   SUMMARY`, `FINAL COMMENT`, `ANCILLARY TESTS`) is embedded **separately**
   through PetBERT, giving each section its own full token budget.
2. The per-column embedding vectors are combined into a single 768-dimensional
   vector via a **weighted average** (higher-weight columns contribute more to
   the final representation; empty cells are excluded automatically).
3. Every taxonomy label is also passed through PetBERT to get a 768-dimensional
   embedding vector.
4. The taxonomy label whose embedding is most similar (by cosine similarity) to
   the case embedding is selected as the prediction.
5. That label's term, group, and code become the output.

There is no classification head, no softmax, and no training loop.  The
"evaluation" is a nearest-neighbor lookup in embedding space.

---

## Input Format

The pipeline reads `ml/data/reportText.csv`, which contains one row per case
and a `case_id` column followed by named text-section columns:

| Column | Role |
|--------|------|
| `case_id` | Unique case identifier |
| `HISTOPATHOLOGICAL SUMMARY` | Microscopic pathology findings (primary input) |
| `FINAL COMMENT` | Pathologist's diagnostic conclusion / interpretation |
| `ANCILLARY TESTS` | IHC, special stains, PCR, and other ancillary results |
| `ADDENDUM` | Follow-up notes or clinical consultations |
| `CLINICAL ABSTRACT` | Referring clinician's history and differential diagnoses |
| `GROSS DESCRIPTION` | Macroscopic description of submitted tissue samples |
| _(others)_ | Case-specific columns (e.g. `IMMUNOHISTOCHEMISTRY`, `COPLOW DIAGNOSES`) |

By default the pipeline embeds `HISTOPATHOLOGICAL SUMMARY`, `FINAL COMMENT`,
and `ANCILLARY TESTS`.  Any subset of columns can be specified via `--text-cols`.

---

## Per-Column Embedding and Weighted Average

Each selected column is embedded separately so that every section gets its
own full `max_length` (default 256) token budget.  This avoids truncation
that would occur if all columns were concatenated into one long string before
tokenization.

The per-column embeddings are combined into a single 768-dimensional vector
using a weighted average.  Default weights:

| Column | Default weight |
|--------|---------------|
| `FINAL COMMENT` | 2.0 |
| `HISTOPATHOLOGICAL SUMMARY` | 1.5 |
| `ANCILLARY TESTS` | 0.5 |

Weights can be overridden at runtime with `--col-weights`:

```
--col-weights "FINAL COMMENT:3.0,HISTOPATHOLOGICAL SUMMARY:2.0,ANCILLARY TESTS:0.5,ADDENDUM:0.5"
```

Two important behaviors:
- **Empty cells are excluded** — if a column has no text for a given row, its
  weight is set to zero for that row and the remaining columns are
  renormalized, so a missing section never dilutes the average.
- **Columns not listed in `--col-weights` default to weight 1.0**, so adding
  extra columns via `--text-cols` works without needing to update weights.

The weighted average is computed in `embed_columns_weighted()` in `embedding.py`.

---

## Pipeline Flow (Step by Step)

The entry point is `run_scan()` in `ml/petbert_scan/pipeline.py`.

### Step 1: Load Input Data

```
reportText.csv  -->  pandas.read_csv(encoding='latin-1')
                -->  clean_text() per cell
                -->  col_texts dict: {col_name: [str, ...]} per selected column
```

- The CSV is read with `latin-1` encoding (handles extended characters in
  pathology reports).
- UTF-8 BOM artifacts from Excel-exported files are stripped from column names.
- Each cell is cleaned: whitespace is stripped, `NaN` values are replaced with
  empty strings (`clean_text()` in `utils.py`).
- Each selected column produces its own list of N strings (`col_texts` dict).
- A merged string per row is also built via `merge_report_columns()` — this is
  used for display and provenance output only, not for embedding.

### Step 1.5: Sub-Diagnosis Expansion (Pass-Through for Report Format)

```
N input rows  -->  split_numbered_diagnoses()  -->  N rows (1-to-1 for reports)
```

The pipeline contains a splitting step for the legacy format where a single
field held multiple diagnoses numbered `"1) ... 2) ..."`.  Since merged report
text does not follow that pattern, each case passes through as a single unit.
The expansion lists (`expanded_ids`, `original_row_indices`, `diagnosis_indices`)
are still populated for consistency with the output schema — `diagnosis_index`
will always be `1`.

### Step 2: Embed Report Columns with PetBERT

```
HISTOPATHOLOGICAL SUMMARY texts  --[PetBERT]-->  (N, 768)  \
FINAL COMMENT texts               --[PetBERT]-->  (N, 768)  --[weighted avg]--> (N, 768)
ANCILLARY TESTS texts             --[PetBERT]-->  (N, 768)  /
```

**Model:** `SAVSNET/PetBERT` -- a BERT-style masked language model pre-trained
on veterinary clinical text from the SAVSNET project.  It is loaded via
HuggingFace `transformers` (`AutoModelForMaskedLM`).

**Per-column embedding** (`embedding.py: embed_columns_weighted`):

Each column is processed independently through the same model:

1. **Tokenization** -- Each column's texts are fed to PetBERT's tokenizer with
   `padding=True`, `truncation=True`, `max_length=256`.  Every column gets
   its own full 256-token budget, so no section is crowded out by another.
   Texts are processed in batches (default 16).

2. **Forward pass** -- Token IDs are passed through PetBERT's **base
   transformer** (not the masked-LM prediction head):
   ```python
   outputs = model.base_model(input_ids=input_ids, attention_mask=attention_mask)
   ```

3. **[CLS] token extraction** -- The output `last_hidden_state` has shape
   `(batch, seq_len, 768)`.  Position 0 (the **[CLS]** token) is taken as
   the fixed-size representation of the entire column text:
   ```python
   cls_embedding = outputs.last_hidden_state[:, 0, :]  # shape: (batch, 768)
   ```

4. **Weighted average** -- The per-column `(N, 768)` matrices are combined:
   - Each column's raw weight is multiplied by a content mask (0 for empty
     cells, 1 for non-empty), so empty cells don't affect the row's average.
   - Effective weights are normalized per-row so they sum to 1.
   - The normalized weighted sum produces a single `(N, 768)` matrix.

5. **Expansion** -- Because the multi-diagnosis splitting step (Step 1.5) may
   produce more than one output row per input row, the row-level embeddings
   are replicated using `original_row_indices` to produce the final `(M, 768)`
   embedding matrix.

**Output:** A NumPy array of shape `(M, 768)` -- one embedding per (expanded) case.

### Step 3: Build and Embed Taxonomy Labels

The taxonomy file `ml/labels/labels.csv` contains ~857 unique entries from the
Vet-ICD-O-canine-1 coding system.  Each entry has a `code`, `group`, and
`term` (parsed by `labels/taxonomy.py`).

To make labels comparable to report text, each label is converted into a
natural-language sentence:

```
"Veterinary diagnosis term: Hemangiosarcoma, NOS. Group: Blood vessel tumors. Code: 9120/3."
```

These sentences are then embedded through the **exact same** PetBERT model
and tokenizer, producing a `(num_labels, 768)` matrix.  This ensures
report embeddings and label embeddings live in the same vector space.

### Step 4: Cosine Similarity Matching

```
                     Label 0    Label 1    ...    Label 856
Case 0             [  0.72       0.45      ...      0.31   ]  <-- pick argmax
Case 1             [  0.38       0.81      ...      0.29   ]
...
Case N             [  0.55       0.41      ...      0.67   ]
```

For each case embedding, we compute its **cosine similarity** against every
label embedding (`embedding.py: cosine_similarity_matrix`):

```
cosine_sim(a, b) = dot(a, b) / (||a|| * ||b||)
```

This produces an `(N, num_labels)` similarity matrix.  For each row, the label
with the **highest** cosine similarity is selected:

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

### Step 6: Map Label Index to ICD Code

The categorization step produces an integer index into the taxonomy list.
`labels/projection.py` resolves each index to the final output fields:

```python
taxonomy_labels[idx].term   -->  predicted_term   (e.g. "Hemangiosarcoma, NOS")
taxonomy_labels[idx].group  -->  predicted_group   (e.g. "Blood vessel tumors")
taxonomy_labels[idx].code   -->  predicted_code    (e.g. "9120/3")
```

### Step 7: Write Outputs

Results are written to the `--out-dir` directory (see
[data-categorization-usage.md](data-categorization-usage.md) for output file
formats).

---

## Worked Example

**Input row (from `reportText.csv`):**
```
case_id:                    "CASE-0003"
HISTOPATHOLOGICAL SUMMARY:  "T1: Examined are 2 sections of haired skin with a
                             nodular coalescing cellular infiltrate... histiocytes
                             and small lymphocytes often tightly surrounding small
                             vessels..."
FINAL COMMENT:              "The marked angiocentricity of the histiocytic and
                             lymphocytic infiltrate is consistent with reactive
                             histiocytosis..."
ANCILLARY TESTS:            "1/15/2025: CD3: numerous small CD3+ T cells with
                             strong expression."
```

**Per-column embedding:**
```
HISTOPATHOLOGICAL SUMMARY → tokenize (≤256 tokens) → PetBERT → [0.08, -0.21, ..., 0.14]  × weight 1.5
FINAL COMMENT             → tokenize (≤256 tokens) → PetBERT → [0.12, -0.34, ..., 0.07]  × weight 2.0
ANCILLARY TESTS           → tokenize (≤256 tokens) → PetBERT → [0.05,  0.11, ..., 0.22]  × weight 0.5
                                                               ↓ normalize weights (sum=1), weighted sum
                                                               → [0.10, -0.23, ..., 0.11]  (768-dim)
```

**Processing:**
1. Each column embedded separately via PetBERT → three 768-dim vectors
2. Weighted average (weights 1.5 / 2.0 / 0.5, normalized) → single 768-dim vector
3. Cosine similarity computed against all 857 label embeddings
3. Closest label: "Histiocytic sarcoma", score 0.71
4. Score 0.71 >= 0.6 threshold → accepted (method: `embedding`)
5. Taxonomy lookup: term="Histiocytic sarcoma", group="Histiocytic tumors", code="9755/3"

**Output:**
| Column | Value |
|--------|-------|
| `case_id` | CASE-0003 |
| `predicted_term` | Histiocytic sarcoma |
| `predicted_group` | Histiocytic tumors |
| `predicted_code` | 9755/3 |
| `confidence` | 0.71 |
| `method` | embedding |

---

## Code Location

| File | Role |
|------|------|
| `ml/petbert_scan/pipeline.py` | Top-level orchestration (`run_scan`) |
| `ml/petbert_scan/embedding.py` | PetBERT loading, per-column weighted embedding, cosine similarity |
| `ml/petbert_scan/categorization.py` | Similarity matching and confidence thresholding |
| `ml/petbert_scan/types.py` | `ScanConfig` (incl. `col_weights`) and `ScanOutputs` dataclasses |
| `ml/petbert_scan/utils.py` | Text cleaning, section merging (display only), diagnosis splitting, device selection |
| `ml/petbert_scan/io.py` | CSV/NPZ/JSON output writers |
| `ml/petbert_scan/cli.py` | Command-line argument parsing |
| `ml/labels/taxonomy.py` | Vet-ICD-O taxonomy CSV parser |
| `ml/labels/catalog.py` | Label catalog builder (label text generation) |
| `ml/labels/projection.py` | Maps label index → term/group/code |
