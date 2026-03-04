# Classifier Development Log

## Architecture Overview

The pipeline has two stages:

1. **Keyword scan** (`keyword_scan/`) — regex/keyword matching on `diagnoses.csv` to produce ground-truth labels. Matches ~18.6% of diagnosis rows (1,537 / 9,172). Produces `keyword_predictions.csv`.

2. **PetBERT scan** (`petbert_scan/`) — embeds report text columns (HISTOPATHOLOGICAL SUMMARY, FINAL COMMENT, ANCILLARY TESTS) and taxonomy labels with [SAVSNET/PetBERT](https://huggingface.co/SAVSNET/PetBERT), then assigns labels by cosine similarity. Optionally uses a trained `PresenceClassifier` to replace raw cosine scores with presence probabilities.

3. **Presence classifier** (`model/presence_classifier.py`) — small MLP trained on (report embedding, label embedding) → binary presence probability. Trained via `run_training_cycle.py` which loops: build training pairs → train → re-run petbert scan → evaluate → log.

### Training data sources (`build_training_pairs.py`)
| Source | Description |
|--------|-------------|
| `positive` | Keyword-confirmed (case, term) pairs |
| `hard_negative` | False-positive predictions from previous eval cycle |
| `fp_extra_negative` | Additional random labels sampled for FP cases |
| `easy_negative` | Random wrong labels for keyword-confirmed cases |

### Evaluation verdicts (`evaluate_predictions.py`)
| Verdict | Meaning |
|---------|---------|
| `good` | Predicted term exactly matches a keyword-matched term |
| `slightly_off` | No exact term match but predicted group matches a keyword group |
| `completely_off` | Neither term nor group matches any keyword label for this case |
| `false_positive` | Case has no keyword labels at all (should be Uncategorized) |
| `false_negative` | Confirmed cancer case with no good/slightly_off prediction |

---

## Results History

| Timestamp | Label | Total | Good | Slightly off | Good+Slight | Completely off | FP | FN |
|-----------|-------|-------|------|-------------|-------------|----------------|-----|-----|
| 13:22 | Baseline (no classifier) | 13,855 | 0.1% | 3.2% | 3.3% | 42.6% | 54.1% | — |
| 13:34 | Classifier v1 | 13,855 | 1.0% | 1.6% | 2.6% | 43.3% | 54.1% | — |
| 13:55 | Classifier v2 | 13,652 | 1.7% | 3.5% | 5.2% | 41.4% | 53.4% | — |
| 14:17 | Classifier v3 | 10,367 | 3.3% | 1.6% | 4.9% | 53.2% | 41.9% | — |
| 14:36 | Classifier v4 | 9,530 | 2.3% | 5.2% | 7.5% | 55.1% | 37.4% | — |
| 14:58 | Classifier v5 | 9,504 | 3.5% | 2.0% | 5.5% | 57.7% | 36.8% | — |
| 15:17 | Classifier v6 | 9,525 | 2.3% | 4.4% | 6.7% | 55.7% | 37.5% | — |
| 15:59 | Classifier v7 | 11,395 | 2.2% | 1.8% | 4.0% | 50.1% | 38.3% | — |
| 16:32 | Classifier v8 | 12,405 | 1.9% | 3.6% | 5.5% | 44.7% | 43.0% | 6.9% |
| 17:07 | **Post-normalization v1** | 9,724 | 1.5% | 4.0% | 5.5% | 54.3% | 30.8% | 9.4% |
| 17:22 | Post-norm classifier v2 | 9,816 | 2.3% | 5.2% | 7.5% | 51.9% | 31.8% | 8.8% |
| 17:38 | Post-norm classifier v3 | 9,545 | 1.8% | 6.6% | **8.4%** | 52.3% | 30.1% | 9.2% |
| 17:55 | Post-norm classifier v4 | 9,708 | 2.1% | 4.4% | 6.5% | 53.3% | 31.1% | 9.1% |

All runs on 2026-03-03, device: mps.

---

## Changes Made

### Fix 1 — Per-label mean subtraction (`categorization.py`, 2026-03-03)

**Problem identified:** The pipeline was predicting only 3 groups (Blood vessel tumors, Lipomatous neoplasms, Adenomas) despite the taxonomy having 52 groups. Root cause: two labels had pathologically high mean cosine similarity across *all* cases regardless of content:

| Label | Group | Mean similarity |
|-------|-------|----------------|
| Pyogenic granuloma | Blood vessel tumors | 0.912 |
| Lipoma, NOS | Lipomatous neoplasms | 0.846 |

These labels' groups always won the `argmax`, making the pipeline useless for 12+ other cancer categories (osteosarcoma, mast cell, lymphoma, SCC, melanoma, etc.).

**Fix:** Subtract each label's mean score (computed across all cases in the current batch) from its column before taking `argmax`:

```python
finite_mask = np.isfinite(sims)
label_means = (
    np.where(finite_mask, sims, 0.0).sum(axis=0)
    / np.maximum(finite_mask.sum(axis=0), 1)
)
sims = sims - label_means[np.newaxis, :]
```

Added in `ml/petbert_scan/categorization.py` after the `sims` matrix is fully assembled, before `np.argmax`.

**Effect:** After this shift `embedding_min_sim` compares centered scores (0.0 = average similarity), so the flag should be set to ~0.05 rather than 0.5.

**Result:** Predicted groups expanded from 3 → 39. FP dropped 43% → 30.8% immediately. Good+Slightly off held at 5.5% (classifier hadn't retrained yet on new distribution).

**Note:** The `label_scores` field stored in `petbert_scan_similarity_scores.csv` now contains centered (mean-subtracted) scores, not raw cosine similarities.

---

### Fix 2 — Group-balanced training positives (`build_training_pairs.py`, 2026-03-03)

**Problem identified:** After Fix 1, the "completely off" rate plateaued at ~52-54% across four consecutive training cycles. Root cause: keyword positives are skewed toward the same 3 groups that dominated before:

| Group | Keyword positives |
|-------|------------------|
| Adenomas and adenocarcinomas | 251 |
| Blood vessel tumors | 176 |
| Lipomatous neoplasms | 104 |
| Osseous and chondromatous | 73 |
| Mast cell neoplasms | 61 |
| Melanocytoma and Melanomas | 42 |

The classifier kept reinforcing the overrepresented groups, limiting its ability to learn the correct group for rarer cancer types.

**Fix:** Added `--max-pos-per-group` argument to `build_training_pairs.py` (and wired through `run_training_cycle.py`). When set, positive examples are capped per group. Recommended starting value: `80`.

```bash
python ml/scripts/run_training_cycle.py \
  --label "group-balanced v1" \
  --max-pos-per-group 80 \
  --embedding-min-sim 0.05 \
  --device mps
```

---

## Key Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--embedding-min-sim` | 0.5 | After Fix 1 (mean subtraction), use ~0.05 instead |
| `--pos-weight` | 1.0 | BCEWithLogitsLoss weight for positives; >1 penalises false negatives more |
| `--recall-weight` | 0.5 | Checkpoint selection: 0.5 = F1, 1.0 = pure recall |
| `--max-pos-per-group` | 0 (no cap) | Cap positives per group; use 80 to balance training data |
| `--epochs` | 20 | Training epochs per cycle |

---

## Known Limitations

- **Keyword ground truth is sparse**: only 18.6% of diagnosis rows get keyword-matched. Many true cancer cases are invisible to the evaluator, inflating false negative counts.
- **Classifier trained on report-level embeddings**: the mean embedding across 3 text columns may lose fine-grained term-level signal present in individual sections.
- **Label text is minimal**: taxonomy labels are embedded as `"{term} {group}"` — richer descriptions (synonyms, ICD-O code context) could improve embedding discrimination.
