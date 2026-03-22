# Keyword Pipeline — How It Works

This document describes the **keyword pipeline** — the system that maps individual
diagnosis field text to standardized Vet-ICD-O-canine-1 labels using rule-based
keyword matching.

> **Role in the two-pipeline architecture:**
> - **Keyword pipeline** (this document) — training only: `diagnosis text → cancer label`
>   Produces ground-truth labels used to train the PetBERT classifier.
>   Does not run in production. Diagnosis rows with no keyword match are treated as
>   non-cancer (Uncategorized).
> - **PetBERT pipeline** (`petbert_pipeline/`) — production system: `report text → cancer label`
>   See [petbert-pipeline.md](petbert-pipeline.md).

The keyword pipeline requires no ML model. Each diagnosis string is scanned for
known taxonomy term keywords using word-boundary regex patterns. The longest
matching keyword wins, with a fallback for `-oma`/`-emia` suffix words.

---

## Why a Separate Keyword Pipeline?

The PetBERT pipeline (production) classifies full multi-section pathology reports.
Training it requires ground-truth cancer labels, but those labels don't exist in
the raw database — only free-text diagnosis strings do.

The keyword pipeline bridges that gap: it scans the short, structured `diagnosis`
field (e.g. `"Hemangiosarcoma, NOS"`, `"Mast cell tumor, grade II"`) that
pathologists write directly against the Vet-ICD-O taxonomy. Because this field is
concise and written in taxonomy-adjacent language, keyword matching achieves
reasonable coverage (~19%) without needing a trained model.

Cases not matched are assumed non-cancer and used as negative training examples.
This assumption holds well for a general clinic, but keyword coverage is the
primary ceiling on classifier quality — more matched cancer cases mean a stronger
training signal.

---

## How It Works (High-Level)

1. Load the `diagnoses.csv` input (one diagnosis row per line; one report can have
   multiple diagnosis rows).
2. Load the Vet-ICD-O taxonomy from `labels.csv` (~845 unique terms across 44+ groups).
3. Build a **keyword index**: for each taxonomy term, generate candidate keyword
   strings and compile them as word-boundary regex patterns, sorted longest-first.
4. Build an **oma/emia index**: index every `-oma` and `-emia` word found in
   taxonomy terms for a lightweight suffix-based fallback.
5. For each diagnosis row, attempt matching in order:
   - **Keyword match** — scan all keyword patterns longest-first; first match wins.
   - **Oma/emia fallback** — extract `-oma`/`-emia` words from the diagnosis text
     and look them up in the suffix index; longest word tried first.
   - **No match** — row is labeled non-cancer (Uncategorized).
6. Write `keyword_predictions.csv` and `keyword_summary.json`.

---

## Pipeline Flow (Step by Step)

The entry point is `run_keyword_scan()` in `ml/keyword_pipeline/pipeline.py`.

### Step 1: Normalize Text

All text — both taxonomy terms and diagnosis strings — passes through `_normalize()`
before any comparison:

```
"Hemangiosarcoma, NOS"  →  "hemangiosarcoma nos"
"B-cell lymphoma/leukemia"  →  "b cell lymphoma leukemia"
"MALIGNANT NEOPLASIA (SEE COMMENT)"  →  "malignant neoplasm  see comment "
```

Normalization steps (applied in order):
1. Lowercase.
2. Collapse hyphens, underscores, and slashes to spaces.
3. Strip commas, parentheses, semicolons, and colons (replaced with spaces).
4. Collapse repeated whitespace.
5. Apply synonym substitutions:
   - `neoplasia` → `neoplasm` (free-text variant not in taxonomy)
   - `plasma cell tumor` → `plasmacytoma` (common shorthand not in taxonomy)

Stripping punctuation is essential: taxonomy terms like `"Osteosarcoma, NOS"` would
otherwise require a literal comma to match a diagnosis of `"OSTEOSARCOMA NOS"`.

### Step 2: Build the Keyword Index

`_build_keyword_index()` processes every taxonomy label and generates candidate
keyword strings:

For each label, two candidates are always produced:
- **Full normalized term** — e.g. `"hemangiosarcoma nos"`
- **Core term** — qualifier words stripped from the end via `_QUALIFIER_RE`
  (strips: `nos`, `nec`, `malignant`, `benign`, `conventional`, `well differentiated`,
  `spindle cell`, `atypical`, and several other descriptor words). E.g.
  `"hemangiosarcoma nos"` → `"hemangiosarcoma"`.

For 2–3 word candidates, all **word permutations** are also added (e.g. the
taxonomy has `"Neoplasm, malignant"` → normalized `"neoplasm malignant"` →
permutation `"malignant neoplasm"` → diagnosis `"MALIGNANT NEOPLASM"` matches).

Each candidate keyword is:
- Skipped if fewer than 6 characters (avoids matching fragments).
- Skipped if already seen (first-defined label wins for duplicate keywords).
- Compiled as `re.compile(r"\b" + re.escape(kw) + r"s?\b")` — word-boundary
  pattern with optional trailing `s` for plurals.

The index is sorted by keyword length descending so the most specific (longest)
match always wins over a shorter fallback.

### Step 3: Build the Oma/Emia Index

`_build_oma_index()` indexes suffix-based tumor words as a lightweight fallback:

For every word in every normalized taxonomy term that ends in `-oma` or `-emia`
(e.g. `hemangiosarcoma`, `fibrosarcoma`, `leukemia`), the word is stored in a
dict mapping it to the label index. First occurrence wins — Preferred terms in
the taxonomy CSV appear before Synonyms, so the primary label is preferred.

The `-emia` extension covers leukemia terms, which would otherwise be completely
missed (they don't end in `-oma` and the taxonomy only lists fully qualified
leukemia names that rarely appear verbatim in diagnosis text).

### Step 4: Match Each Diagnosis Row

`_match_diagnosis()` applies the two-stage matching strategy to each row:

**Stage 1 — Keyword match:**

The normalized diagnosis string is scanned against every pattern in the keyword
index (longest first). The first pattern that matches returns its associated label.

```
diagnosis: "SUBCUTANEOUS MAST CELL TUMOR, GRADE II"
normalized: "subcutaneous mast cell tumor  grade ii"
  → keyword "subcutaneous mast cell tumor" matches → label: "Subcutaneous mast cell tumor"
```

**Stage 2 — Oma/emia fallback:**

If no keyword matches, `_OMA_RE` extracts every `-oma`/`-emia` word from the
normalized diagnosis text. Words are sorted by length descending (longer = more
specific) and each is looked up in the oma/emia index.

```
diagnosis: "BONE MARROW: LEUKEMIA AND MYELOPHTHISIS"
normalized: "bone marrow  leukemia and myelophthisis"
  → extracted ema words: ["leukemia", "myelophthisis"... wait, only -oma/-emia]
  → "leukemia" in oma_index → label: (first leukemia term in taxonomy)
```

If no suffix word matches either, the row is recorded as `no_match`.

### Step 5: Write Outputs

Results are written to `--out-dir`:
- `keyword_predictions.csv` — one row per diagnosis row; see output format below.
- `keyword_summary.json` — aggregate statistics for the run.

---

## Input Format

The pipeline reads `database/data/output/diagnoses.csv`, which contains one row
per individual diagnosis (a single pathology report can have multiple rows):

| Column | Role |
|--------|------|
| `case_id` | Links diagnosis rows back to the original report |
| `diagnosis_number` | Ordering within the report (optional, passed through if present) |
| `diagnosis` | The free-text diagnosis string — the field that is matched |

---

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--csv` | `database/data/output/diagnoses.csv` | Input diagnoses CSV path |
| `--id-col` | `case_id` | Case ID column name |
| `--diag-num-col` | `diagnosis_number` | Diagnosis number column name (optional; passed through to output if present) |
| `--text-col` | `diagnosis` | Column containing the diagnosis text to match |
| `--labels-csv` | `ml/labels/labels.csv` | Path to Vet-ICD-O taxonomy CSV |
| `--out-dir` | `ml/output/diagnoses` | Output directory |
| `--max-rows` | all | Cap on input rows (useful for quick testing) |

---

## Example Commands

**Standard run** (from repo root):
```bash
PYTHONPATH=ml ml/.venv/Scripts/python.exe -m keyword_pipeline \
  --csv database/data/output/diagnoses.csv \
  --labels-csv ml/labels/labels.csv \
  --out-dir ml/output/diagnoses
```

**Quick test on first 200 rows:**
```bash
PYTHONPATH=ml ml/.venv/Scripts/python.exe -m keyword_pipeline \
  --csv database/data/output/diagnoses.csv \
  --labels-csv ml/labels/labels.csv \
  --out-dir ml/output/diagnoses \
  --max-rows 200
```

---

## Output Files

### `keyword_predictions.csv`

One row per input diagnosis row.

| Column | Description |
|--------|-------------|
| `case_id` | Case identifier |
| `diagnosis_number` | Diagnosis ordering within the report (if present in input) |
| `diagnosis` | Original diagnosis text |
| `matched_term` | Taxonomy term matched (empty if `no_match`) |
| `matched_group` | Taxonomy group for the matched term |
| `matched_code` | Vet-ICD-O-canine-1 morphology code |
| `matched_keyword` | The specific keyword string that triggered the match |
| `method` | `keyword`, `oma_fallback`, or `no_match` |

### `keyword_summary.json`

Aggregate run statistics:

| Field | Description |
|-------|-------------|
| `csv_path` | Input file path |
| `total_rows` | Total diagnosis rows processed |
| `method_counts` | Counts for `keyword`, `oma_fallback`, and `no_match` |
| `match_rate_pct` | Percentage of rows that received any match |
| `top_matched_terms` | Top 20 taxonomy terms by match count |
| `top_matched_groups` | Top 10 taxonomy groups by match count |

---

## Current Coverage (as of 2026-03-05)

From the most recent run over 42,973 diagnosis rows:

| Method | Count | % of total |
|--------|-------|-----------|
| `keyword` | 7,247 | 16.9% |
| `oma_fallback` | 991 | 2.3% |
| `no_match` | 34,735 | 80.8% |
| **Total matched** | **8,238** | **19.2%** |

The ~81% no-match rate reflects:
1. Many submissions are genuinely non-neoplastic (inflammatory, degenerative, normal findings).
2. Incomplete keyword coverage — some cancer cases use terminology not yet in the taxonomy keyword set.
3. Each report produces multiple diagnosis rows; non-cancer secondary findings on cancer reports contribute to no-match rows.

The no-match rows are used as **negative training examples** for the PetBERT
classifier. Improving keyword coverage directly increases the number of confirmed
cancer training cases, which is the primary lever for improving classifier quality.

---

## Known Limitations

**Leukemia fallback maps to first taxonomy occurrence.**
Diagnoses that just say `"LEUKEMIA"` or `"B CELL LEUKEMIA"` are caught by the
`-emia` fallback but may be assigned the first leukemia term in the taxonomy
(e.g. `"Mast cell leukemia"`) rather than the most specific match. The group
(`Mast cell neoplasms`) may also be incorrect. A leukemia-specific sub-classifier
would be needed to resolve B/T/myeloid subtypes from free-text.

**Synonym list is manually maintained.**
The `_normalize()` substitutions (`neoplasia → neoplasm`, `plasma cell tumor → plasmacytoma`)
are hardcoded. New common shorthands not in the taxonomy must be added manually.

**No negation handling.**
Phrases like `"NO EVIDENCE OF NEOPLASIA"` or `"RULE OUT LYMPHOMA"` will match
(`neoplasm`, `lymphoma`) and be treated as cancer positives. This inflates the
false-positive rate among matched rows.

**Metastasis wording mismatch.**
The taxonomy has `"Neoplasm, metastatic"` but diagnoses often say `"METASTASIS"`.
The two words don't share a suffix and aren't synonyms in `_normalize()`, so
`"LYMPH NODE: METASTASIS (SEE COMMENT)"` will be a `no_match`. Adding a
`metastasis → metastatic neoplasm` synonym risks false positives from phrases
like `"NO EVIDENCE OF METASTASIS"` (see negation limitation above).

---

## Code Location

| File | Role |
|------|------|
| `ml/keyword_pipeline/pipeline.py` | Core logic: `_normalize`, `_build_keyword_index`, `_build_oma_index`, `_match_diagnosis`, `run_keyword_scan` |
| `ml/keyword_pipeline/cli.py` | Command-line argument parsing |
| `ml/keyword_pipeline/__main__.py` | `python -m keyword_pipeline` entry point |
| `ml/labels/taxonomy.py` | Vet-ICD-O taxonomy CSV parser (`TaxonomyLabel`, `load_labels_taxonomy`) |
| `ml/labels/labels.csv` | Vet-ICD-O-canine-1 taxonomy (~845 unique terms) |
| `ml/output/diagnoses/keyword_predictions.csv` | Output: per-row match results |
| `ml/output/diagnoses/keyword_summary.json` | Output: aggregate run statistics |
