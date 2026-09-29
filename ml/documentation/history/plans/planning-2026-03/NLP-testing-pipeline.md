# NLP Testing Pipeline

In order to do additional training for the PetBERT model, we would need a metric for its performance, and that's where the NLP Testing Pipeline comes in

# Goals
1. Provide a metric to evaluate the performance of our NLP Pipeline
2. Provide the infrastructure needed for the training pipeline (WIP) to perform error calculation by comparing its prediction to the label/target

# Current NLP Pipeline
- Input: `report.csv` in `ml/data/`, ID column: `case_id`
- Text columns embedded independently (no weighting):
  - `FINAL COMMENT`
  - `HISTOPATHOLOGICAL SUMMARY`
  - `ANCILLARY TESTS`
  - Additional columns (`ADDENDUM`, `CLINICAL ABSTRACT`, etc.) can be added via `--text-cols`
- Model: `SAVSNET/PetBERT`
- Output files written to `ml/output/report/`:
  - `petbert_scan_predictions.csv` — one row per (case, prediction rank), up to 5 predictions per case ranked by confidence
  - `petbert_scan_column_scores.csv` — one row per (case × column), shows each column's top match and which was decisive
  - `petbert_scan_provenance.csv` — one row per case, includes `predicted_category` and `embedding_similarity` score
  - `petbert_scan_similarity_scores.csv` — full cosine similarity matrix against every taxonomy label
  - `petbert_scan_visualization.csv` — PCA 2D coordinates for each case
  - `petbert_scan_embeddings.npz` — raw embedding vectors
  - `petbert_scan_summary.json` — run-level stats

## How Categorization Works
Each text column is embedded independently using PetBERT (mean pooling), producing a 768-dim vector per column per case. The element-wise maximum similarity across all columns is used so the strongest column wins per (case, label) pair. A confidence threshold (`--embedding-min-sim`, default `0.6`) is applied:
- **Score ≥ threshold** → included as a ranked prediction (`method = "embedding"`), up to 5 per case
- **Score < threshold** → only the top-1 is kept, marked `"Uncategorized"` (`method = "low_confidence"`)
- **Empty text (all columns)** → no output row (`method = "empty"`)

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

# Apples-to-Apples Comparison

## Data Sources
| File | Key columns | Rows | Cases |
|---|---|---|---|
| `ml/output/report/petbert_scan_predictions.csv` | `case_id`, `diagnosis_index`, `predicted_term`, `predicted_group` | ~13 855 | ~2 771 |
| `ml/output/diagnoses/keyword_predictions.csv` | `case_id`, `diagnosis_number`, `matched_term`, `matched_group` | ~9 172 | ~2 783 |

~1 709 rows in `keyword_predictions.csv` have a non-blank `matched_term`, spanning ~1 273 cases. These are the ground-truth labeled set. All other rows are treated as **Uncategorized**.

## Join Key
Join on **`case_id`** only. `diagnosis_index` (petbert) and `diagnosis_number` (keyword) represent independent orderings and do **not** need to align.

## Per-Case Comparison Logic
For each `case_id` that has at least one non-blank `matched_term`:

1. Collect the **label set**: all non-blank `matched_term` values (and their `matched_group`) from `keyword_predictions.csv`.
2. Collect the **prediction set**: all `predicted_term` values (and their `predicted_group`) from `petbert_scan_predictions.csv` (up to 5 per case).
3. For each label in the label set, find the best matching prediction:

| Result | Condition |
|---|---|
| **Good** | Any `predicted_term` exactly equals the `matched_term` |
| **Slightly off** | No exact term match, but any `predicted_group` equals the `matched_group` |
| **Completely off** | Neither term nor group matches any prediction |

Because ordering doesn't matter, a single exact match anywhere in the prediction set counts as **Good**, regardless of rank.

## Aggregate Metrics
After scoring every label across all cases:

| Metric | Description |
|---|---|
| **Good %** | Labels where at least one predicted term was exact |
| **Slightly off %** | Labels with a group match but no exact term match |
| **Completely off %** | Labels with no term or group match |
| **Coverage** | Fraction of non-blank labels in cases that had any prediction |

Per-group breakdowns of Good / Slightly off / Completely off reveal which taxonomy groups the model struggles with most.

# Next Steps
1. Build a comparison script that joins the two files on `case_id`
2. Filter to labeled rows (`matched_term` non-blank)
3. Apply the three-tier scoring logic above per label
4. Output a summary table (Good / Slightly off / Completely off counts + percentages) plus a per-group breakdown
5. Feed the error signal into the training pipeline (WIP)

