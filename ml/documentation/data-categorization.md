# Data Categorization Pipeline — How It Works

This document describes how the pipeline maps veterinary pathology reports
to standardized Vet-ICD-O-canine-1 codes using the PetBERT language model.
Each report section is embedded independently and matched against the taxonomy
via cosine similarity to produce up to 5 ranked predictions per case:

- **Term** -- the specific diagnosis name (e.g. "Hemangiosarcoma, NOS")
- **Group** -- the tumor category (e.g. "Blood vessel tumors")
- **Code** -- the ICD-O morphology code (e.g. "9120/3")

For CLI options, output file formats, and example commands see
[data-categorization-usage.md](../ml/scripts/data-categorization-usage.md).

---

## How It Works (High-Level)

The pipeline does **not** fine-tune or train any model.  Instead it uses
PetBERT purely as a **feature extractor** and compares embeddings via cosine
similarity:

1. Each selected text column (e.g. `HISTOPATHOLOGICAL SUMMARY`, `FINAL COMMENT`,
   `ANCILLARY TESTS`) is embedded independently through PetBERT, producing a
   768-dimensional vector per column per case via **mean pooling**.
2. Every taxonomy label is also embedded through the same PetBERT model.
3. For each case, the label with the highest cosine similarity across **any**
   column is selected as the top candidate.
4. All labels above the confidence threshold are returned as ranked predictions
   (up to 5 per case).  If no label passes the threshold the top-1 is returned
   as `low_confidence`.
5. Each prediction's label index is resolved to a term, group, and code.

There is no classification head, no softmax, and no training loop.  The
"evaluation" is a nearest-neighbor lookup in embedding space.

---

## Input Format

The pipeline reads `ml/data/report.csv`, which contains one row per case
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

---

## Pipeline Flow (Step by Step)

The entry point is `run_scan()` in `ml/petbert_scan/pipeline.py`.

### Step 1: Load Input Data

```
report.csv  -->  pandas.read_csv(encoding='latin-1')
            -->  clean_text() per cell
            -->  col_texts: {column_name: [text per row]}
            -->  merged display string per row (for provenance / neighbors only)
```

- The CSV is read with `latin-1` encoding (handles extended characters in
  pathology reports).
- UTF-8 BOM artifacts from Excel-exported files are stripped from column names.
- Each cell is cleaned: whitespace is stripped, `NaN` values are replaced with
  empty strings (`clean_text()` in `utils.py`).
- Selected columns are also merged into a single labeled display string per row
  via `merge_report_columns()` — used for provenance output and neighbors only.

### Step 2: Embed Each Column Independently with PetBERT

```
col_texts  --[embed_columns_separate()]-->  {column: (N, 768)}
```

**Model:** `SAVSNET/PetBERT` -- a BERT-style masked language model pre-trained
on veterinary clinical text from the SAVSNET project.  Loaded via HuggingFace
`transformers` (`AutoModelForMaskedLM`).

Each selected column is passed through PetBERT separately via
`embed_columns_separate()` in `embedding.py`, which calls `embed_texts()` per
column.  Each text is:

1. **Tokenized** with `padding=True`, `truncation=True`, `max_length=256`.

2. **Forward-passed** through PetBERT's base transformer (not the MLM head):
   ```python
   outputs = model.base_model(input_ids=input_ids, attention_mask=attention_mask)
   ```

3. **Mean-pooled** over non-padding tokens:
   ```python
   mask = attention_mask.unsqueeze(-1).float()
   mean_embedding = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
   ```

**Output:** A dict `{col: (N, 768)}` — one embedding per case per column.
Empty cells in a column produce a zero-masked embedding so they cannot
influence similarity scores.

A mean embedding across non-empty columns is also computed per case and used
for PCA visualization, nearest-neighbor search, and the embeddings NPZ.

### Step 3: Build and Embed Taxonomy Labels

The taxonomy file `ml/labels/labels.csv` contains ~857 unique entries from the
Vet-ICD-O-canine-1 coding system.  Each entry has a `code`, `group`, and
`term` (parsed by `labels/taxonomy.py`).

To make labels comparable to report text, each label is converted into a
short text string combining the term and group:

```
"Hemangiosarcoma, NOS Blood vessel tumors"
```

These sentences are embedded through the **same** PetBERT model and tokenizer,
producing a `(num_labels, 768)` matrix.

### Step 4: Cosine Similarity Matching (Per Column)

For each column, cosine similarity is computed against every label embedding:

```
cosine_sim(a, b) = dot(a, b) / (||a|| * ||b||)
```

This produces one `(N, num_labels)` similarity matrix per column.  The
element-wise **maximum** across columns is then taken so the strongest column
wins per (case, label) pair:

```
                     Label 0    Label 1    ...    Label 856
Col: HIST SUMMARY  [  0.72       0.45      ...      0.31   ]
Col: FINAL COMMENT [  0.68       0.55      ...      0.40   ]
                     -----       -----               -----
Max across cols    [  0.72       0.55      ...      0.40   ]  <-- used for ranking
```

### Step 5: Top-k Confidence Threshold

Labels are sorted by score descending.  A confidence threshold (default `0.6`)
is applied to select up to 5 predictions per case:

| Condition | Result | `method` |
|-----------|--------|----------|
| Text is empty (all columns) | No output row | `empty` |
| Score >= 0.6 | Include in ranked predictions (up to 5) | `embedding` |
| No label reaches 0.6 | Include top-1 only | `low_confidence` |

The `diagnosis_index` in the predictions CSV is the rank (1 = best match).

### Step 6: Map Label Index to ICD Code

`labels/projection.py` resolves each label index to the final output fields:

```python
taxonomy_labels[idx].term   -->  predicted_term   (e.g. "Hemangiosarcoma, NOS")
taxonomy_labels[idx].group  -->  predicted_group   (e.g. "Blood vessel tumors")
taxonomy_labels[idx].code   -->  predicted_code    (e.g. "9120/3")
```

### Step 7: Compute Per-Column Scores

For each column, the top-1 label and score are recorded independently.  This
produces the `column_scores.csv` file that shows which report section was most
informative and which column was `was_decisive` (had the highest score that
determined the final prediction).

### Step 8: Write Outputs

Results are written to the `--out-dir` directory (see
[data-categorization-usage.md](../ml/scripts/data-categorization-usage.md)
for output file formats).

---

## Worked Example

**Input row (from `report.csv`):**
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
HISTOPATHOLOGICAL SUMMARY → tokenize → PetBERT → mean pool → (768-dim)
FINAL COMMENT             → tokenize → PetBERT → mean pool → (768-dim)
ANCILLARY TESTS           → tokenize → PetBERT → mean pool → (768-dim)
```

**Similarity matching (top results after max across columns):**
1. "Histiocytic sarcoma" — 0.71 (FINAL COMMENT was decisive)
2. "Reactive histiocytosis" — 0.67
3. "Cutaneous histiocytoma" — 0.61

All three pass the 0.6 threshold → 3 rows in `predictions.csv`.

**Column scores output (`column_scores.csv`):**
| column_name | top_term | top_score | was_decisive |
|---|---|---|---|
| HISTOPATHOLOGICAL SUMMARY | Histiocytic sarcoma | 0.68 | False |
| FINAL COMMENT | Histiocytic sarcoma | 0.71 | True |
| ANCILLARY TESTS | T-cell lymphoma | 0.54 | False |

---

## Code Location

| File | Role |
|------|------|
| `ml/petbert_scan/pipeline.py` | Top-level orchestration (`run_scan`) |
| `ml/petbert_scan/embedding.py` | PetBERT loading, per-column mean-pooled embedding, cosine similarity |
| `ml/petbert_scan/categorization.py` | Top-k similarity matching and confidence thresholding |
| `ml/petbert_scan/types.py` | `ScanConfig` and `ScanOutputs` dataclasses |
| `ml/petbert_scan/utils.py` | Text cleaning, section merging (display only), device selection |
| `ml/petbert_scan/io.py` | CSV/NPZ/JSON output writers |
| `ml/petbert_scan/cli.py` | Command-line argument parsing |
| `ml/labels/taxonomy.py` | Vet-ICD-O taxonomy CSV parser |
| `ml/labels/catalog.py` | Label catalog builder (label text generation) |
| `ml/labels/projection.py` | Maps label index → term/group/code |
