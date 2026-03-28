# Ideas to Try

Ideas brainstormed but not yet implemented. Each entry includes motivation, approach, effort/risk, and when to attempt it.

---

## #5 — Cross-Encoder Re-Ranking

**Status:** Not started

**Problem:** The current bi-encoder approach embeds the report and label independently, then scores by dot product. The model never directly attends to report tokens in the context of a specific label — all cross-label reasoning happens in the small MLP head.

**Approach:** Use the existing bi-encoder (current PresenceClassifier) to retrieve the top-K candidates (K≈15), then run a cross-encoder (PetBERT fine-tuned to classify `[report_text, label_text]` pairs directly) to re-rank those candidates. Cross-encoders have full attention between report and label tokens — significantly more expressive than dot product.

Training data already exists: `training_pairs.csv` has (report_emb, label_emb, positive/negative) tuples. For a cross-encoder, we'd need the raw text pairs (`contrastive_pairs.csv` has those). Fine-tuning PetBERT as a cross-encoder requires loading its tokenizer and running it on concatenated `[report_text] [SEP] [label_text]` inputs.

**Effort:** High — new model class, new training loop, integration into production pipeline.

**Risk:** Medium — cross-encoders reliably outperform bi-encoders in retrieval literature, but the improvement may be modest at the current data scale. Main risk is that the top-15 bi-encoder candidates miss the correct label too often (in which case re-ranking can't help).

**When to try:** After accumulating ~8,000+ labeled cases. Calibration (#4) has been tried and doesn't help at the current scale, so cross-encoder re-ranking is the next architectural lever worth exploring.

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
