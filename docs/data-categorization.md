# Data Categorization Pipeline — How It Works

This document describes how the pipeline maps full veterinary pathology reports
to standardized Vet-ICD-O-canine-1 codes using the PetBERT language model.
Selected sections of each report are merged into one input string, which is
then embedded and matched against the taxonomy to produce:

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

1. The specified columns from `reportText.csv` (e.g. `HISTOPATHOLOGICAL
   SUMMARY`, `FINAL COMMENT`, `ANCILLARY TESTS`) are **merged** into one
   labelled string per case.
2. The merged report string is passed through PetBERT to get a 768-dimensional
   embedding vector.
3. Every taxonomy label is also passed through PetBERT to get a 768-dimensional
   embedding vector.
4. The taxonomy label whose embedding is most similar (by cosine similarity) to
   the report embedding is selected as the prediction.
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

By default the pipeline merges `HISTOPATHOLOGICAL SUMMARY`, `FINAL COMMENT`,
and `ANCILLARY TESTS`.  Any subset of columns can be specified via `--text-cols`.

---

## Section Merging

Before embedding, the selected columns are concatenated into one string with
section-label prefixes so PetBERT can distinguish between them:

```
[HISTOPATHOLOGICAL SUMMARY] T1: Examined are 2 sections of haired skin with
a nodular coalescing cellular infiltrate... [FINAL COMMENT] The marked
angiocentricity of the histiocytic infiltrate is consistent with reactive
histiocytosis... [ANCILLARY TESTS] 1/15/2025: CD3: numerous small CD3+ T cells.
```

Empty or NaN columns are silently skipped so sparse sections don't contribute
noise.  The merge is performed by `merge_report_columns()` in `utils.py`.

---

## Pipeline Flow (Step by Step)

The entry point is `run_scan()` in `ml/petbert_scan/pipeline.py`.

### Step 1: Load and Merge Input Data

```
reportText.csv  -->  pandas.read_csv(encoding='latin-1')
                -->  merge_report_columns(row, text_cols)
                -->  one labelled string per case
```

- The CSV is read with `latin-1` encoding (handles extended characters in
  pathology reports).
- UTF-8 BOM artifacts from Excel-exported files are stripped from column names.
- Each cell is cleaned: whitespace is stripped, `NaN` values are replaced with
  empty strings (`clean_text()` in `utils.py`).
- The specified text columns are merged row-wise into one string per case.
- The result is two parallel lists: `ids` (case IDs) and `texts` (merged strings).

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

### Step 2: Embed Report Text with PetBERT

```
"[HISTOPATHOLOGICAL SUMMARY] T1: ..." --[tokenize]--> [101, 2093, ...] --[PetBERT]--> [0.12, -0.34, ..., 0.07]
                                                                                        ^^^^^^^^^^^^^^^^^^^^^^^^
                                                                                        768-dimensional vector
```

**Model:** `SAVSNET/PetBERT` -- a BERT-style masked language model pre-trained
on veterinary clinical text from the SAVSNET project.  It is loaded via
HuggingFace `transformers` (`AutoModelForMaskedLM`).

**How data is passed into the model** (`embedding.py: embed_texts`):

1. **Tokenization** -- The merged string is fed to PetBERT's tokenizer with
   `padding=True`, `truncation=True`, `max_length=256`.  This converts the
   text into integer token IDs plus an attention mask.  Texts are processed in
   batches (default 16).  Long reports are truncated to 256 tokens.

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
   This is standard practice for BERT-family models: the [CLS] token aggregates
   information from the full sequence.

**Output:** A NumPy array of shape `(N, 768)` -- one embedding per case.

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

### Step 6: Auxiliary Label Override (Optional)

If `--use-auxiliary-labels` is enabled, cases known to have carcinoma or
sarcoma (from external CSV lists of case IDs) get their prediction
**constrained**:

1. Load case ID sets from `dataCarcinoma.csv` and `dataSarcoma.csv`.
2. Find all taxonomy labels whose term contains "carcinoma" or "sarcoma".
3. For a case in the carcinoma set: ignore all non-carcinoma labels and
   pick the highest-scoring carcinoma label instead.
4. Same logic for sarcoma cases.
5. If a case appears in **both** lists: mark as "conflict" and keep the
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

**After merging:**
```
[HISTOPATHOLOGICAL SUMMARY] T1: Examined are 2 sections of haired skin with a
nodular coalescing cellular infiltrate... [FINAL COMMENT] The marked
angiocentricity of the histiocytic and lymphocytic infiltrate is consistent
with reactive histiocytosis... [ANCILLARY TESTS] 1/15/2025: CD3: numerous
small CD3+ T cells with strong expression.
```

**Processing:**
1. Merged string tokenized (~256 tokens, truncated if longer) and embedded via PetBERT → 768-dim vector
2. Cosine similarity computed against all 857 label embeddings
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
| `ml/petbert_scan/embedding.py` | PetBERT loading, text embedding, cosine similarity |
| `ml/petbert_scan/categorization.py` | Similarity matching and confidence thresholding |
| `ml/petbert_scan/auxiliary_policy.py` | Carcinoma/sarcoma auxiliary override logic |
| `ml/petbert_scan/types.py` | `ScanConfig` and `ScanOutputs` dataclasses |
| `ml/petbert_scan/utils.py` | Text cleaning, section merging, diagnosis splitting, device selection |
| `ml/petbert_scan/io.py` | CSV/NPZ/JSON output writers |
| `ml/petbert_scan/cli.py` | Command-line argument parsing |
| `ml/labels/taxonomy.py` | Vet-ICD-O taxonomy CSV parser |
| `ml/labels/catalog.py` | Label catalog builder (label text generation) |
| `ml/labels/projection.py` | Maps label index → term/group/code |
| `ml/labels/auxiliary.py` | Helpers for auxiliary label constraints |
