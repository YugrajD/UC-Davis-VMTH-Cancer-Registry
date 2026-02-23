# PetBERT Scan (Taxonomy Mapping)

This pipeline categorizes each clinical diagnosis row to taxonomy terms from:

- `ml/labels/labels.csv`

The label source is fixed to `ml/labels/labels.csv`, and outputs:

- `predicted_term`
- `predicted_group`
- `predicted_code` (Vet-ICD-O-canine-1 code)

`ml/data/dataCarcinoma.csv` and `ml/data/dataSarcoma.csv` can be used as **additional supervision labels** for matching `anon_id`s in the unified run.

## Run

```bash
ml/.venv11/bin/python ml/scripts/petbert_scan.py \
  --csv ml/data/data.csv \
  --id-col anon_id \
  --text-col "Clinical Diagnoses" \
  --model SAVSNET/PetBERT \
  --labels-csv ml/labels/labels.csv \
  --use-auxiliary-labels \
  --carcinoma-csv ml/data/dataCarcinoma.csv \
  --sarcoma-csv ml/data/dataSarcoma.csv \
  --task categorize \
  --out-dir ml/output/data_taxonomy
```

Use `--local-only` if the model is already cached and you want to avoid network calls.

## Architecture

- `ml/petbert_scan/pipeline.py`: orchestration of loading, embedding, categorization, and writing outputs
- `ml/petbert_scan/embedding.py`: model/tokenizer loading + embedding and cosine utilities
- `ml/petbert_scan/categorization.py`: embedding-based label selection logic
- `ml/petbert_scan/auxiliary_policy.py`: applies optional carcinoma/sarcoma constraints by `anon_id`
- `ml/labels/taxonomy.py`: reads `labels.csv` into typed taxonomy records
- `ml/labels/catalog.py`: builds label catalog used for embedding comparison
- `ml/labels/projection.py`: maps selected label indices to term/group/code output fields
- `ml/labels/auxiliary.py`: shared auxiliary-label helper functions

## Outputs

- `petbert_scan_rows.csv`: row-level scan output + embedding PCA coordinates
- `petbert_scan_categories.csv`: row-level predictions including term/group/code
- `petbert_scan_embeddings.npz`: embeddings + ids + texts
- `petbert_scan_summary.json`: summary counts and run metadata

When auxiliary labels are enabled, output CSVs include:

- `auxiliary_label` (`carcinoma`, `sarcoma`, `conflict`, or empty)
- `category_method` values `auxiliary_carcinoma` or `auxiliary_sarcoma` when the auxiliary label selected the final taxonomy term
- `category_method` values `embedding`, `low_confidence`, and `empty` for standard embedding flow states
