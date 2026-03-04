# PetBERT Pipeline Revamp

It appears that the current PetBERT pipeline is not very effective.

## Problems
1. Lack of multi-diagnosis support
    - Each entry(case) in the report.csv could point to several diagnosis.
2. Outdated output format
    - The current output is not very representative of the data that was fed into the pipeline
    - The is no correlation between the different text columns within the report



## Solutions
1. Multi-diagnosis support
- Instead of matching it with the single closest label, allow the model to output several predictions at once (max 5), given that they pass the confidence level
- For each case, rank all taxonomy labels by cosine similarity and return up to 5 that exceed the `--embedding-min-sim` threshold
- `diagnosis_index` in the output is the prediction rank (1 = best match, 2 = second best, etc.)
- This replaces the current top-1 logic in `run_categorization()` in `categorization.py`

2. Refactor the output files

### File 1: Predictions (`petbert_scan_predictions.csv`)
Strictly for viewing and exporting. One row per (case, prediction rank).

| Column | Description |
|---|---|
| `case_id` | Patient identifier |
| `diagnosis_index` | Rank of this prediction for the case (1 = best, up to 5) |
| `predicted_term` | Matched taxonomy term (e.g. "Hemangiosarcoma, NOS") |
| `predicted_group` | Matched taxonomy group (e.g. "Blood vessel tumors") |
| `predicted_code` | ICD-O morphology code (blank if low_confidence) |
| `confidence` | Cosine similarity score (2 decimal places) |
| `method` | How the result was assigned: `embedding` or `low_confidence` |

**Gap vs current:** `run_categorization()` uses `np.argmax` to return only the single top-1 label. It needs to be changed to return the top-k labels (up to 5) whose scores exceed the threshold, producing one output row per qualifying label per case.

---

### File 2: Column Scores (`petbert_scan_column_scores.csv`)
Shows how each text column in `report.csv` independently scored against the taxonomy. Uses `embed_columns_separate()` (already exists in `embedding.py` but is not yet called by `pipeline.py`).

One row per (case × column). Columns absent from `--text-cols` are excluded.

| Column | Description |
|---|---|
| `row_index` | Original CSV row index (0-based) |
| `case_id` | Patient identifier |
| `column_name` | Report section name (e.g. `FINAL COMMENT`, `HISTOPATHOLOGICAL SUMMARY`) |
| `column_text` | Raw text content of this section for this case (empty string if blank) |
| `top_term` | Label with highest cosine similarity from this column's embedding alone |
| `top_group` | Taxonomy group for `top_term` |
| `top_code` | ICD-O code for `top_term` |
| `top_score` | Cosine similarity score for this column vs. `top_term` |
| `was_decisive` | `True` if this column's top score was the highest across all columns and determined the final prediction |

**Gap vs current:** `pipeline.py` builds `col_texts` per column but never embeds them separately. The fix is to call `embed_columns_separate()` in `run_scan()` and write this new file alongside the existing outputs.

---

### Other output files (keep as-is)
- `petbert_scan_provenance.csv` — per-case debug/traceability
- `petbert_scan_similarity_scores.csv` — full N×M cosine similarity matrix
- `petbert_scan_visualization.csv` — PCA 2D coordinates
- `petbert_scan_embeddings.npz` — raw embedding vectors
- `petbert_scan_summary.json` — aggregate stats
- `petbert_scan_neighbors.csv` — top-k nearest neighbor pairs (optional)
