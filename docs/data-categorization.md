# Data Categorization Pipeline — How It Works

This document describes how the pipeline maps free-text clinical diagnosis
strings to standardized Vet-ICD-O-canine-1 codes using the PetBERT language
model.  Each input row may contain multiple numbered diagnoses, and the
pipeline produces one prediction per sub-diagnosis:

- **Term** -- the specific diagnosis name (e.g. "Hemangiosarcoma, NOS")
- **Group** -- the tumor category (e.g. "Blood vessel tumors")
- **Code** -- the ICD-O morphology code (e.g. "9120/3")

For CLI options, output file formats, and example commands see
[data-categorization-usage.md](data-categorization-usage.md).

---

## How It Works (High-Level)

The pipeline does **not** fine-tune or train any model.  Instead it uses
PetBERT purely as a **feature extractor** and compares embeddings via cosine
similarity:

1. Clinical entries with multiple numbered diagnoses (e.g. `"1) Osteosarcoma
   2) Chronic cystitis"`) are **split** into individual sub-diagnoses.
2. Every sub-diagnosis string is passed through PetBERT to get a
   768-dimensional embedding vector.
3. Every taxonomy label is also passed through PetBERT to get a
   768-dimensional embedding vector.
4. For each sub-diagnosis, the taxonomy label whose embedding is most similar
   (by cosine similarity) is selected as the prediction.
5. That label's term, group, and code become the output.

There is no classification head, no softmax, and no training loop.  The
"evaluation" is a nearest-neighbor lookup in embedding space.

---

## Multi-Diagnosis Splitting

Veterinary clinical entries frequently contain multiple diagnoses in a single
field, numbered as:

```
1) Osteosarcoma: primary lesion at right proximal femur 2) Chronic cystitis
```

The pipeline automatically detects this `1) ... 2) ... 3) ...` pattern and
splits the text into individual sub-diagnoses **before** embedding.  Each
sub-diagnosis is then embedded and categorized independently, producing one
prediction per condition rather than one blended prediction for the entire
entry.

### How splitting works

- The splitter (`split_numbered_diagnoses()` in `utils.py`) checks whether the
  text starts with `1)`.  Only then does it split on all `N)` markers.
- Texts without numbered formatting are left as-is (single sub-diagnosis).
- Empty texts produce a single empty-string sub-diagnosis.

### Why split instead of top-K?

Embedding the full multi-diagnosis string produces a single vector that
blends all conditions together.  A top-K approach on that blended embedding
would likely return K variations of the *dominant* diagnosis rather than
capturing each distinct condition.  By splitting first, each sub-diagnosis
gets its own embedding that accurately represents that specific condition.

### Output provenance columns

Two new columns track which sub-diagnosis each output row came from:

| Column | Description |
|--------|-------------|
| `row_index` | Index of the original CSV row (0-based) |
| `diagnosis_index` | Position of this sub-diagnosis within its original entry (1-based, matching the `1)`, `2)` numbering) |
| `original_text` | The full unsplit clinical text for reference |

The `Clinical Diagnoses` column (or whatever `--text-col` is set to) contains
the individual **sub-diagnosis text** that was actually embedded and matched.

### Worked example (multi-diagnosis)

**Input row:**
```
anon_id: "ID_1"
Clinical Diagnoses: "1) Osteosarcoma: right proximal femur 2) Chronic cystitis"
```

**After splitting:**
| Sub-diagnosis | Text |
|---------------|------|
| 1 | `Osteosarcoma: right proximal femur` |
| 2 | `Chronic cystitis` |

**Output (2 rows):**

| row_index | diagnosis_index | Clinical Diagnoses | predicted_term | confidence | method |
|-----------|-----------------|--------------------|----------------|-----------|--------|
| 0 | 1 | Osteosarcoma: right proximal femur | Osteoblastic osteosarcoma | 0.73 | embedding |
| 0 | 2 | Chronic cystitis | Uncategorized | 0.32 | low_confidence |

Non-cancer sub-diagnoses like "Chronic cystitis" naturally receive low
similarity scores against the cancer-specific taxonomy and are marked as
`low_confidence` / `Uncategorized`.

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

### Step 1.5: Split Multi-Diagnosis Entries

```
N input rows  -->  split_numbered_diagnoses()  -->  M sub-diagnosis rows (M >= N)
```

- Each text is checked for the `1) ... 2) ...` pattern.
- Multi-diagnosis entries are split into individual sub-diagnoses.
- Parallel tracking lists are created:
  - `expanded_ids` -- patient ID repeated for each sub-diagnosis
  - `expanded_texts` -- individual sub-diagnosis strings
  - `original_row_indices` -- maps each sub-diagnosis to its original CSV row
  - `diagnosis_indices` -- 1-based position within the original entry
  - `original_texts` -- the full unsplit text for each sub-diagnosis

All downstream steps operate on the expanded M-length lists.

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

**Output:** A NumPy array of shape `(M, 768)` -- one embedding per
sub-diagnosis.

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
Diagnosis M        [  0.55       0.41      ...      0.67   ]
```

For each sub-diagnosis embedding, we compute its **cosine similarity** against
every label embedding (`embedding.py: cosine_similarity_matrix`):

```
cosine_sim(a, b) = dot(a, b) / (||a|| * ||b||)
```

This produces an `(M, num_labels)` similarity matrix where `M` = number of
sub-diagnoses and `num_labels` = number of taxonomy labels.  For each row,
the label with the **highest** cosine similarity is selected:

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
steer predictions toward the right tumor family.  When multi-diagnosis
splitting is active, the override applies to every sub-diagnosis belonging
to that patient (using the expanded patient IDs).

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

## Worked Example (Single Diagnosis)

**Input row:**
```
anon_id: "ID_42"
Clinical Diagnoses: "Multicentric lymphoma stage IVa"
```

**Processing:**
1. Text cleaned -> `"Multicentric lymphoma stage IVa"`
2. No numbered pattern detected -> 1 sub-diagnosis (the full text)
3. Tokenized (22 tokens) and embedded via PetBERT -> 768-dim vector
4. Cosine similarity computed against all 857 label embeddings
5. Closest label: index 312, term "Lymphoma, NOS", score 0.78
6. Score 0.78 >= 0.6 threshold -> accepted (method: `embedding`)
7. No auxiliary override (ID_42 not in carcinoma/sarcoma lists)
8. Taxonomy lookup: term="Lymphoma, NOS", group="Malignant lymphomas", code="9590/3"

**Output:**
| Column | Value |
|--------|-------|
| `row_index` | 0 |
| `diagnosis_index` | 1 |
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
| `ml/petbert_scan/utils.py` | Text cleaning, diagnosis splitting, device selection |
| `ml/petbert_scan/io.py` | CSV/NPZ/JSON output writers |
| `ml/petbert_scan/cli.py` | Command-line argument parsing |
| `ml/labels/taxonomy.py` | Vet-ICD-O taxonomy CSV parser |
| `ml/labels/catalog.py` | Label catalog builder (label text generation) |
| `ml/labels/projection.py` | Maps label index -> term/group/code |
| `ml/labels/auxiliary.py` | Helpers for auxiliary label constraints |
