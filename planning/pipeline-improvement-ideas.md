# Pipeline Improvement Ideas

Based on current state: ~32% Good+Slight, ~33% CO floor, bank saturating at ~80k pairs (Phase 12).

Roughly ordered by expected impact.

---

## 1. More keyword-confirmed cases (highest leverage)

The data tells the story clearly: 1,273 cases → 20% Good+Slight, 5,788 cases → 33% Good+Slight. The single biggest improvement driver has been keyword coverage. Getting to ~10k confirmed cases would likely both improve the binary classifier further *and* finally make the GroupClassifier viable.

---

## 2. GroupClassifier (addresses the CO floor architecturally)

The ~33% CO floor is a structural ceiling for the binary approach — it scores labels pair-wise and picks the best, so it has no way to reason about *which group* a case belongs to globally. The GroupClassifier makes one decision per report (`report_emb → group probabilities`) which directly eliminates the argmax-across-wrong-labels failure mode. It overfits at 5,788 cases but is projected to need ~10k to generalize. This is the documented path forward.

---

## 3. PetBERT fine-tuning

The `training/finetune/` directory is a placeholder. Fine-tuning PetBERT on the corpus would close the domain gap between its pre-training distribution and veterinary pathology text. Currently, the classifier has to compensate for embeddings that weren't specialized for this domain. This would benefit both the binary and group classifier.

---

## 4. Per-column embeddings instead of mean pooling

### The problem

There is an asymmetry between how cosine similarity and the `PresenceClassifier` use the column embeddings:

- **Cosine path** ([categorization.py:66-76](../ml/petbert_pipeline/categorization.py#L66-L76)): computes a separate similarity matrix per column, then takes the **element-wise max** across columns. If any column strongly matches a label, that signal wins.
- **PresenceClassifier path** ([pipeline.py:217-218](../ml/petbert_pipeline/pipeline.py#L217-L218)): scores `classifier.score_matrix(mean_embeddings, label_embeddings)` — receives only the **averaged** 768-dim vector.

The mean is computed in [pipeline.py:124-128](../ml/petbert_pipeline/pipeline.py#L124-L128):

```python
col_emb_stack = np.stack([col_embeddings[col] for col in cols], axis=0)  # (C, N, 768)
content_counts = np.maximum(content_mask.sum(axis=0), 1.0)               # (N,)
embeddings = (col_emb_masked.sum(axis=0) / content_counts[:, None])      # (N, 768)
```

And in [train.py:157](../ml/training/binary/train.py#L157), training also reads `mean_embeddings`:

```python
report_embs_list.append(cache["mean_embeddings"][ridx])
```

So the classifier trains and infers on averaged embeddings. If HISTOPATHOLOGICAL SUMMARY says "squamous cell carcinoma" but ANCILLARY TESTS is unrelated, the mean embedding dilutes that signal before the classifier ever sees it. The cosine path would pick up the strong column hit; the classifier cannot.

### Why it matters

The three columns carry structurally different content:
- **HISTOPATHOLOGICAL SUMMARY** — pathological diagnosis, tumor type, morphology
- **FINAL COMMENT** — clinical interpretation, prognosis, treatment context
- **ANCILLARY TESTS** — IHC markers, staining, supporting test results

Averaging them treats all three as equally informative for every case. A short ANCILLARY TESTS entry (often empty) pulls the mean toward a generic, low-signal region.

The `col_embeddings` are **already stored in the cache** ([pipeline.py:102](../ml/petbert_pipeline/pipeline.py#L102)), so no PetBERT re-run is needed — this is purely a training and model architecture change.

### Option A — Concatenate column embeddings (most expressive)

Change the classifier input from `[report_emb ‖ label_emb]` (1536-dim) to `[col1 ‖ col2 ‖ col3 ‖ label_emb]` (2304+768 = 3072-dim). The classifier can learn independently which columns are diagnostic for which labels, and how to weight them. Empty columns would be zeroed before concatenation using `col_has_content`.

**Tradeoffs:**
- Model input doubles in size; parameter count of the first MLP layer increases proportionally
- Requires a `PresenceClassifier` architecture change and a cold start (old checkpoint incompatible)
- Training pairs in the cache would need to look up `col_embeddings[ridx]` instead of `mean_embeddings[ridx]`

### Option B — Per-column classifier pass, take max score (no architecture change)

Run the existing classifier on each column separately and take `max(score_col1, score_col2, score_col3)` as the final presence probability. This mirrors the cosine max-across-columns logic exactly. The existing `PresenceClassifier` model is unchanged; only inference changes.

**Tradeoffs:**
- 3× inference cost (3 classifier passes per case instead of 1) — at 12,620 cases × 857 labels this is meaningful
- Training still uses `mean_embeddings`, so there is a train/inference mismatch — the classifier was trained on means but infers on individual columns
- Would require retraining on per-column pairs to eliminate the mismatch

### Option C — Learned column attention (most principled)

A small attention layer takes the three column embeddings and the label embedding and outputs a weighted combination: `attention(col1, col2, col3, label) → weighted_report_emb`. The weights are label-dependent — e.g. ANCILLARY TESTS gets high weight when the label is an IHC-defined subtype.

**Tradeoffs:**
- More parameters, more complex to implement
- Needs enough data to learn meaningful label-conditional attention weights — 5,788 cases may be sufficient but borderline
- Most generalizable if it works

### Recommended starting point

**Option A** is the most straightforward to implement correctly (train and infer on the same representation, no mismatch) and the most likely to help given the structural column differences. It requires:
1. Update `PresenceClassifier` input dim from `2 * PETBERT_EMB_DIM` to `4 * PETBERT_EMB_DIM` (or parameterise as `n_cols * EMB_DIM + EMB_DIM`)
2. Update `train.py` to concatenate `col_embeddings[col1][ridx]`, `col_embeddings[col2][ridx]`, `col_embeddings[col3][ridx]` instead of `mean_embeddings[ridx]`
3. Update `pipeline.py` classifier call to pass concatenated column embeddings
4. Cold start required (new model architecture, new cache format for training)

---

## 5. Revisit label enrichment with a lighter blend

Fix 9's cache-based enrichment caused regression because the 50/50 blend was too aggressive — it shifted label embeddings far enough into report space to change the cosine score landscape and invalidate the CO bank. A smaller alpha (e.g., 0.1–0.2 report contribution instead of 0.5) might help without destabilizing training, and could be explored without a cold start disruption.

---

## 6. Bank pruning / learned-pair down-weighting

Phase 12 shows the bank saturating at ~80k pairs with diminishing new rows per cycle (~1,779 in c8). Once saturation is reached, the training signal stops growing. Pairs the classifier has already learned well (consistently predicting `absent` with high confidence) are taking up sampling budget without providing gradient signal. Pruning or down-weighting "easy" bank pairs would free capacity for harder, newer ones.

---

## 7. Per-group calibration / thresholds

The current `--embedding-min-sim 0.05` is a global threshold. Some groups are reliably predicted (low CO contribution) while others dominate CO errors. Per-group thresholds — or simply filtering out predictions from high-CO-rate groups at inference time until they improve — could shift the global metrics without retraining.

---

## Summary

The highest ROI path is: **more keyword data → GroupClassifier**. Everything else is incremental on the binary architecture, which is close to its ceiling at current data volumes.
