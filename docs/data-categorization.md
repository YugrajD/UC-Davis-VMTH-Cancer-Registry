# Data Categorization Pipeline — How It Works

This document describes how the pipeline maps veterinary pathology reports
to standardized Vet-ICD-O-canine-1 codes using the PetBERT language model.
Each sub-diagnosis string is embedded independently and matched against the
taxonomy via cosine similarity to produce:

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

1. Multi-diagnosis entries (e.g. `"1) Osteosarcoma 2) Cystitis"`) are split
   into individual sub-diagnosis strings so each is categorized independently.
2. Each sub-diagnosis string is embedded through PetBERT, producing a
   single 768-dimensional vector via **mean pooling** of the attended token
   hidden states.
3. Every taxonomy label is also embedded through the same PetBERT model.
4. The taxonomy label whose embedding is most similar (by cosine similarity)
   to the sub-diagnosis embedding is selected as the prediction.
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

The selected columns (default: `HISTOPATHOLOGICAL SUMMARY`, `FINAL COMMENT`,
`ANCILLARY TESTS`) are merged into a single display string per row for
provenance output.  The **embedding** operates on the sub-diagnosis strings
extracted from that merged text, not on the raw column texts.

---

## Pipeline Flow (Step by Step)

The entry point is `run_scan()` in `ml/petbert_scan/pipeline.py`.

### Step 1: Load Input Data

```
reportText.csv  -->  pandas.read_csv(encoding='latin-1')
                -->  clean_text() per cell
                -->  merged display string per row (for provenance only)
```

- The CSV is read with `latin-1` encoding (handles extended characters in
  pathology reports).
- UTF-8 BOM artifacts from Excel-exported files are stripped from column names.
- Each cell is cleaned: whitespace is stripped, `NaN` values are replaced with
  empty strings (`clean_text()` in `utils.py`).
- Selected columns are merged into a labeled string per row via
  `merge_report_columns()` — used for display and provenance only, not for
  embedding.

### Step 1.5: Sub-Diagnosis Splitting

```
N input rows  -->  split_numbered_diagnoses()  -->  M sub-diagnosis strings (M >= N)
```

Clinical entries formatted as `"1) Osteosarcoma 2) Cystitis"` are split into
individual sub-diagnosis strings.  Each sub-diagnosis is then embedded and
categorized independently, and results are collapsed back to one prediction
row per original case in the output.

For cases with a single un-numbered diagnosis, the text passes through as-is
(`M == N`).

### Step 2: Embed Sub-Diagnoses with PetBERT

```
sub-diagnosis strings  --[PetBERT, mean pool]-->  (M, 768)
```

**Model:** `SAVSNET/PetBERT` -- a BERT-style masked language model pre-trained
on veterinary clinical text from the SAVSNET project.  Loaded via HuggingFace
`transformers` (`AutoModelForMaskedLM`).

**Embedding** (`embedding.py: embed_texts`):

Each sub-diagnosis string is passed through the same model:

1. **Tokenization** -- Texts are tokenized with `padding=True`,
   `truncation=True`, `max_length=256`.  Processed in batches (default 16).

2. **Forward pass** -- Token IDs are passed through PetBERT's **base
   transformer** (not the masked-LM prediction head):
   ```python
   outputs = model.base_model(input_ids=input_ids, attention_mask=attention_mask)
   ```

3. **Mean pooling** -- The output `last_hidden_state` has shape
   `(batch, seq_len, 768)`.  The hidden states of all non-padding tokens
   are averaged to produce a single 768-dim vector per text:
   ```python
   mask = attention_mask.unsqueeze(-1).float()      # (B, T, 1)
   mean_embedding = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
   ```
   Mean pooling outperforms [CLS] for cosine-based retrieval when the model
   has not been fine-tuned for sentence similarity.

**Output:** A NumPy array of shape `(M, 768)` -- one embedding per sub-diagnosis.

### Step 3: Build and Embed Taxonomy Labels

The taxonomy file `ml/labels/labels.csv` contains ~857 unique entries from the
Vet-ICD-O-canine-1 coding system.  Each entry has a `code`, `group`, and
`term` (parsed by `labels/taxonomy.py`).

To make labels comparable to sub-diagnosis text, each label is converted into a
natural-language sentence:

```
"Veterinary diagnosis term: Hemangiosarcoma, NOS. Group: Blood vessel tumors. Code: 9120/3."
```

These sentences are embedded through the **same** PetBERT model and tokenizer,
producing a `(num_labels, 768)` matrix.  This ensures diagnosis embeddings and
label embeddings live in the same vector space.

### Step 4: Cosine Similarity Matching

```
                     Label 0    Label 1    ...    Label 856
Sub-diag 0         [  0.72       0.45      ...      0.31   ]  <-- pick argmax
Sub-diag 1         [  0.38       0.81      ...      0.29   ]
...
Sub-diag M         [  0.55       0.41      ...      0.67   ]
```

For each sub-diagnosis embedding, cosine similarity is computed against every
label embedding (`embedding.py: cosine_similarity_matrix`):

```
cosine_sim(a, b) = dot(a, b) / (||a|| * ||b||)
```

This produces an `(M, num_labels)` similarity matrix.  For each row, the label
with the **highest** cosine similarity is selected.

### Step 5: Confidence Threshold

A confidence threshold (default `0.6`) is applied:

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

**Sub-diagnosis splitting:** No numbered prefix → one sub-diagnosis string
(merged section text).

**Embedding:**
```
merged text → tokenize (≤256 tokens) → PetBERT → mean pool → [0.10, -0.23, ..., 0.11]  (768-dim)
```

**Processing:**
1. Sub-diagnosis embedded via PetBERT mean pooling → 768-dim vector
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
| `ml/petbert_scan/embedding.py` | PetBERT loading, mean-pooled embedding, cosine similarity |
| `ml/petbert_scan/categorization.py` | Similarity matching and confidence thresholding |
| `ml/petbert_scan/types.py` | `ScanConfig` and `ScanOutputs` dataclasses |
| `ml/petbert_scan/utils.py` | Text cleaning, section merging (display only), diagnosis splitting, device selection |
| `ml/petbert_scan/io.py` | CSV/NPZ/JSON output writers |
| `ml/petbert_scan/cli.py` | Command-line argument parsing |
| `ml/labels/taxonomy.py` | Vet-ICD-O taxonomy CSV parser |
| `ml/labels/catalog.py` | Label catalog builder (label text generation) |
| `ml/labels/projection.py` | Maps label index → term/group/code |
