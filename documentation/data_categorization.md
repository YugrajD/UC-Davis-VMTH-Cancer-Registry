# Data Categorization Pipeline

This document describes how clinical diagnosis data is categorized into:

- `Term` (from taxonomy)
- `Group` (from taxonomy)
- `Vet-ICD-O-canine-1 code` (from taxonomy)

## Primary Goal

Given clinical diagnosis text rows (for each `anon_id`), the pipeline predicts the best taxonomy term from:

- `ml/labels/labels.csv`

Then it maps the selected term to its corresponding group and code.

## Code Location

- Pipeline orchestration: `ml/petbert_scan/pipeline.py`
- Embedding + model loading: `ml/petbert_scan/embedding.py`
- Categorization logic: `ml/petbert_scan/categorization.py`
- Auxiliary-label policy: `ml/petbert_scan/auxiliary_policy.py`
- Taxonomy loader: `ml/labels/taxonomy.py`
- Label catalog builder: `ml/labels/catalog.py`
- Taxonomy projection: `ml/labels/projection.py`
- Auxiliary helper functions: `ml/labels/auxiliary.py`

## Inputs

- Main input dataset:
  - `ml/data/data.csv`
  - required columns: `anon_id`, `Clinical Diagnoses`
- Taxonomy label source:
  - `ml/labels/labels.csv`
- Auxiliary supervision datasets (optional but supported):
  - `ml/data/dataCarcinoma.csv`
  - `ml/data/dataSarcoma.csv`

## Label Source and Embedding Space

The label source is always `ml/labels/labels.csv`.

1. Taxonomy records are loaded from `ml/labels/labels.csv`.
2. Each taxonomy row becomes a candidate label with:
   - `term`
   - `group`
   - `code`
3. A label text is constructed per candidate term for embedding comparison.

The model used for embeddings is PetBERT (`SAVSNET/PetBERT` by default).

## Categorization Logic

For each diagnosis text row:

1. Compute text embedding with PetBERT.
2. Compute similarity between text embedding and all taxonomy label embeddings.
3. Select top label index by highest cosine similarity.
4. Apply confidence rule:
   - If similarity >= `embedding_min_sim`, use that predicted label.
   - Else mark as `Uncategorized` (while retaining the closest taxonomy candidate as reference fields).

Keyword classifier logic is not used. Categorization is embedding-based only.

## Auxiliary Labels (Carcinoma / Sarcoma)

If `--use-auxiliary-labels` is enabled:

1. Build two patient-id sets from:
   - `dataCarcinoma.csv` -> carcinoma ids
   - `dataSarcoma.csv` -> sarcoma ids
2. For rows whose `anon_id` is in one of these sets:
   - restrict candidate terms to those containing `"carcinoma"` or `"sarcoma"` (normalized text match)
   - choose the highest-scoring term within that constrained subset
3. Override final prediction for those rows.

Method flags:

- `embedding`
- `low_confidence`
- `empty`
- `auxiliary_carcinoma`
- `auxiliary_sarcoma`
- `conflict` (if an id appears in both auxiliary sets; model output is kept)

## Taxonomy Projection

After final label index is selected, it is projected to taxonomy fields:

- `predicted_term`
- `predicted_group`
- `predicted_code`

This is the main output mapping used downstream.

## Output Files

Default outputs are written to the configured `--out-dir`:

- `petbert_scan_rows.csv`
- `petbert_scan_categories.csv`
- `petbert_scan_embeddings.npz`
- `petbert_scan_summary.json`

Key output columns include:

- `predicted_term`
- `predicted_group`
- `predicted_code`
- `auxiliary_label`
- `category_method`
- `category_confidence`
- `predicted_label_index`

## Example Command

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
  --out-dir ml/output/data_taxonomy \
  --local-only
```
