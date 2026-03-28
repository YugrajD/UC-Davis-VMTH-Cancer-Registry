# Ideas to Explore

This document records improvement ideas that were brainstormed but not yet implemented (or partially implemented). Each entry includes motivation, approach, effort/risk, and the current status.

---

## #4 — Per-Label Score Calibration

**Status:** Implemented (`--mode calibrate` in `run_training.py`)

**Problem:** After mean-centering the score matrix, different labels still have different score variances. A label the model is uncertain about (low variance, scores clustered near 0) will lose argmax to higher-variance labels even when it is the correct answer.

**Approach:** For each label `l` with ≥10 labeled cases, find a scalar offset `b_l` (via grid search over [-0.3, 0.3]) that maximizes the number of correct predictions for cases whose ground truth is label `l`:

    calibrated_score_l = (score_l - mean_l) + b_l

Offsets are saved to `ml/output/calibration/label_offsets.json` and applied at inference time by passing `--calibration-offsets` to the production pipeline.

**Notes:**
- Labels with <10 GT examples default to `b_l = 0` (no offset) to avoid overfitting.
- Optimization is greedy (one label at a time) — a reasonable first-order approximation.
- Only applies to the binary `run_categorization()` path (not group/hybrid modes).
- Potential ceiling: with no held-out test set, the projected improvement is measured on training data. Real improvement may be smaller.

---

## #5 — Cross-Encoder Re-Ranking

**Status:** Not started

**Problem:** The current bi-encoder approach embeds the report and label independently, then scores by dot product. The model never directly attends to report tokens in the context of a specific label — all cross-label reasoning happens in the small MLP head.

**Approach:** Use the existing bi-encoder (current PresenceClassifier) to retrieve the top-K candidates (K≈15), then run a cross-encoder (PetBERT fine-tuned to classify `[report_text, label_text]` pairs directly) to re-rank those candidates. Cross-encoders have full attention between report and label tokens — significantly more expressive than dot product.

Training data already exists: `training_pairs.csv` has (report_emb, label_emb, positive/negative) tuples. For a cross-encoder, we'd need the raw text pairs (`contrastive_pairs.csv` has those). Fine-tuning PetBERT as a cross-encoder requires loading its tokenizer and running it on concatenated `[report_text] [SEP] [label_text]` inputs.

**Effort:** High — new model class, new training loop, integration into production pipeline.

**Risk:** Medium — cross-encoders reliably outperform bi-encoders in retrieval literature, but the improvement may be modest at the current data scale. Main risk is that the top-15 bi-encoder candidates miss the correct label too often (in which case re-ranking can't help).

**When to try:** After accumulating ~8,000+ labeled cases; also more useful if per-label threshold calibration (#4) doesn't move the needle enough.

---

## #7 — Column Attention

**Status:** Not started (NOT the same as Phase 14/15)

**Problem:** The current architecture (`col_pair_mode=False`) concatenates all three column embeddings with the label embedding into a 3072-dim vector, then passes it through the MLP. This treats all columns equally regardless of which one is most relevant for a given label.

**What was already tried (and failed):**
- **Phase 14** (`col_pair_mode=True`, max-pool): Scored each column-label pair independently, then took the max across columns. Regressed −7.3pp because independent scoring loses cross-column interactions.
- **Phase 15** (`col_pair_mode=True`, learned combine): Replaced max-pool with a 3-weight linear combiner — still independent per-column scoring. Also regressed.

**What has NOT been tried:**
True attention where all three column embeddings are attended to *jointly* with the label embedding as the query:

    keys/values = stack([col1_emb, col2_emb, col3_emb])  # (3, 768)
    query = label_emb  # (768,)
    attn_weights = softmax(query @ keys.T / sqrt(768))  # (3,)
    context = attn_weights @ values  # (768,)
    → classifier input: [context | label_emb]  # (1536,)

The critical difference from Phase 14/15: the attention weights are *conditioned on the label*, so "which column matters" is computed jointly, not independently. Cross-column interactions are preserved in the softmax normalization.

**Effort:** Medium — new PresenceClassifier mode, full Phase experiment (~8 cycles).

**Risk:** Medium — attention over 3 elements is very small; unclear if it outperforms the current concat (which already sees all three columns + the label simultaneously, just without explicit attention weighting). Worth trying if the data ceiling rises with more labeled cases.

**When to try:** After accumulating ~8,000+ labeled cases. Run as a Phase experiment alongside calibration to isolate the effect.
