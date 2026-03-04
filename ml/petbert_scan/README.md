# PetBERT Scan (Taxonomy Mapping)

This pipeline maps each clinical report from `reportText.csv` to taxonomy
terms from `ml/labels/labels.csv` using PetBERT embeddings and cosine
similarity.  It outputs up to 5 predictions per case ranked by confidence:

- `predicted_term`
- `predicted_group`
- `predicted_code` (Vet-ICD-O-canine-1 code)

## Run

```bash
ml/.venv/bin/python -m petbert_scan \
  --csv ml/data/report.csv \
  --id-col case_id \
  --text-cols "HISTOPATHOLOGICAL SUMMARY,FINAL COMMENT,ANCILLARY TESTS" \
  --model SAVSNET/PetBERT \
  --labels-csv ml/labels/labels.csv \
  --task categorize \
  --out-dir ml/output/report
```

Use `--local-only` if the model is already cached and you want to avoid network calls.

## Architecture

- `ml/petbert_scan/pipeline.py`: orchestration of loading, embedding, categorization, and writing outputs
- `ml/petbert_scan/embedding.py`: model/tokenizer loading, per-column mean-pooled embedding, and cosine utilities
- `ml/petbert_scan/categorization.py`: top-k embedding-based label selection with confidence thresholding
- `ml/petbert_scan/utils.py`: text cleaning, section merging, device selection
- `ml/labels/taxonomy.py`: reads `labels.csv` into typed taxonomy records
- `ml/labels/catalog.py`: builds label catalog used for embedding comparison
- `ml/labels/projection.py`: maps selected label indices to term/group/code output fields

## Outputs

- `petbert_scan_predictions.csv`: one row per (case, prediction rank) — up to 5 predictions per case ranked by confidence
- `petbert_scan_column_scores.csv`: one row per (case × column) — shows how each text column independently scored against the taxonomy and which was decisive
- `petbert_scan_provenance.csv`: per-case traceability and debug info
- `petbert_scan_similarity_scores.csv`: full cosine similarity matrix (one column per taxonomy label)
- `petbert_scan_visualization.csv`: PCA coordinates for 2-D plotting
- `petbert_scan_embeddings.npz`: compressed NumPy archive with raw 768-dim embedding vectors
- `petbert_scan_summary.json`: run metadata and aggregate counts
