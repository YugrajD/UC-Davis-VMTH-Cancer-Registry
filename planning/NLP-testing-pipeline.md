# NLP Testing Pipeline

In order to do additional training for the PetBERT model, we would need a metric for its performance, and that's where the NLP Testing Pipeline comes in

# Goals
1. Provide a metric to evaluate the performance of our NLP Pipeline
2. Provide the infrastructure needed for the training pipeline (WIP) to perform error calculation by comparing its prediction to the label/target

# Current NLP Pipeline
- Input: `reportText.csv` in `ml/data/`, ID column: `case_id`
- Text columns embedded (with per-column weights):
  - `FINAL COMMENT` (weight 2.0)
  - `HISTOPATHOLOGICAL SUMMARY` (weight 1.5)
  - `ANCILLARY TESTS` (weight 0.5)
  - Additional columns (`ADDENDUM`, `CLINICAL ABSTRACT`, etc.) can be added via `--text-cols`
- Model: `SAVSNET/PetBERT`
- Output files written to `ml/output/reportText/`:
  - `petbert_scan_predictions.csv` — one row per case, collapsed predictions
  - `petbert_scan_provenance.csv` — one row per sub-diagnosis, includes `predicted_category` and `embedding_similarity` score
  - `petbert_scan_similarity_scores.csv` — full cosine similarity matrix against every taxonomy label
  - `petbert_scan_visualization.csv` — PCA 2D coordinates for each sub-diagnosis
  - `petbert_scan_embeddings.npz` — raw embedding vectors
  - `petbert_scan_summary.json` — run-level stats

## How Categorization Works
Each text column is embedded independently using PetBERT (mean pooling), producing a 768-dim vector per column per case. The per-column embeddings are weighted and the label with the highest cosine similarity across any column wins. A confidence threshold (`--embedding-min-sim`, default `0.6`) is applied:
- **Score ≥ threshold** → assigned the top taxonomy label (`method = "embedding"`)
- **Score < threshold** → marked `"Uncategorized"` (`method = "low_confidence"`)
- **Empty text** → blank (`method = "empty"`)

Multi-diagnosis entries (e.g., `"1) Osteosarcoma 2) Cystitis"`) are split into sub-diagnoses, each embedded and categorized independently, then collapsed back to one row per case in the predictions CSV.

# Labels for Supervised Training
- **Source:** `matched_term` column in `keyword_predictions.csv` in `ml/output/diagnoses/`
- **Coverage:** Most entries are blank — those should be treated as ground-truth `"Uncategorized"`
- **Class imbalance:** The dataset is heavily skewed toward uncategorized, so accuracy alone is not a sufficient metric

# Proposed Evaluation Metrics
To properly evaluate the model against `matched_term` ground-truth labels, the testing pipeline should compute:

| Metric | Why it matters |
|---|---|
| **Accuracy** | Baseline — fraction of predictions matching the label |
| **Precision / Recall / F1** | Handles class imbalance better than accuracy |
| **Macro-F1** | Weights each class equally (important given sparse labels) |
| **Confusion matrix** | Reveals which categories are being confused |
| **Coverage** | Fraction of non-blank labels that received a non-Uncategorized prediction |

Since most labels are blank (= Uncategorized), per-class F1 and macro-F1 are the most meaningful single numbers to track.

# Next Steps
1. Build a comparison script that joins `petbert_scan_provenance.csv` with `keyword_predictions.csv` on `case_id`
2. Filter to rows where `matched_term` is non-blank (labeled set)
3. Compute precision, recall, F1 per class and macro-F1
4. Feed the error signal into the training pipeline (WIP)

