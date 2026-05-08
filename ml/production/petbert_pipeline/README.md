# Production Pipeline (Taxonomy Mapping)

This pipeline maps each clinical report from `report.csv` to taxonomy terms from
`ml/ICD_labels/labels.csv` using PetBERT embeddings, the trained
`PresenceClassifier`, and group-keyword term correction. It outputs ranked
predictions per case:

- `predicted_term`
- `predicted_group`
- `predicted_code` (Vet-ICD-O-canine-1 code)

## Run

```bash
ml/.venv/bin/python ml/scripts/run_production.py \
  --csv ml/data/report.csv \
  --id-col case_id \
  --text-cols "HISTOPATHOLOGICAL SUMMARY,FINAL COMMENT,ANCILLARY TESTS" \
  --model SAVSNET/PetBERT \
  --labels-csv ml/ICD_labels/labels.csv \
  --task categorize \
  --out-dir ml/output/report
```

Use `--local-only` if the model is already cached and you want to avoid network calls.

## Architecture

- `pipeline.py`: thin orchestrator — load → text-select → embed → call each stage → write outputs
- `embedding.py`: model/tokenizer loading and per-column mean-pooled embedding
- `text_selector.py`: TF-IDF multi-column text selection
- `embedding_cache.py`: save/load cached embeddings
- `stages/case_presence_classifier.py`: Stage 1 — CasePresenceClassifier gate
- `stages/group_classifier.py`: Stage 2 — GroupClassifier
- `stages/label_presence_classifier.py`: Stage 3a — per-group LabelPresenceClassifier
- `stages/keyword_correction.py`: Stage 3b — ICD-O behavior + subtype keyword filter
- `stages/__init__.py`: per-case dispatcher (`categorize_per_case`)
- `io.py`: write all output files
- `utils.py`: text cleaning, section merging, device selection
- `ml/ICD_labels/taxonomy.py`: reads `labels.csv` into typed taxonomy records
- `ml/ICD_labels/catalog.py`: builds label catalog used for embedding comparison
- `ml/ICD_labels/projection.py`: maps selected label indices to term/group/code output fields

## Outputs

- `petbert_predictions.csv`: ranked predictions per case
- `petbert_column_scores.csv`: per-column debug breakdown
- `petbert_provenance.csv`: per-case traceability and debug info
- `petbert_similarity_scores.csv`: full label-score matrix dump
- `petbert_visualization.csv`: PCA coordinates for 2-D plotting
- `petbert_embeddings.npz`: compressed NumPy archive with saved embedding vectors
- `petbert_summary.json`: run metadata and aggregate counts
