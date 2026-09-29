# PresenceClassifier Optimization Ideas (post-Phase 13)

Phase 13 achieved 40.0% Good+Slight with per-column embeddings (Fix 10), up from 32.0% in Phase 12.
These are potential next steps to push further, ranked by expected impact vs. implementation effort.

---

## Current ceiling analysis

The ~30% CO floor is driven by training data, not architecture:
- Cases with rare cancers have few training examples across 44 groups
- Cases where report language doesn't match any trained pattern
- Pairwise binary scoring has no global group context

No MLP tuning alone will eliminate the CO floor — that requires either more keyword data or a
group-level architecture. But within the binary classifier paradigm, 1–3% gains may still be available.

---

## Idea A — Larger hidden_dim (highest priority, lowest effort)

**Motivation:** `hidden_dim=256` was set when input was 1536-dim (mean embedding: 2×768).
With per-column concatenation the input is now 3072-dim — a 12:1 compression ratio into the
first linear layer, vs. 6:1 before. This may be a bottleneck.

**Try:**
```bash
ml/.venv/Scripts/python.exe ml/scripts/run_training.py \
  --label "phase14-hd512 c1" \
  --hidden-dim 512 \
  --co-neg-per-case 5 \
  --fp-neg-per-case 10 \
  --embedding-min-sim 0.05 \
  --epochs 25 \
  --recall-weight 0.25 \
  --device xpu \
  --local-only
```

Also worth trying `--hidden-dim 768`. Both are backward-compatible (hidden_dim is not saved in the
checkpoint — it is a training-time hyperparameter only, reconstructed at load time from `DEFAULT_HIDDEN_DIM`
or the `--hidden-dim` flag).

**Note:** Requires a cold start only if you want to compare against a fresh bank; the embedding cache
is still valid (embeddings don't change). You can reuse the existing bank and cache.

**Expected gain:** 1–3% if the bottleneck hypothesis is correct. Zero gain if the CO floor is
entirely data-limited at this point.

---

## Idea B — More epochs for the larger model

**Motivation:** The 3072-dim input model may converge more slowly than the old 1536-dim model.
Phase 13 used epochs=25. A larger hidden_dim model may benefit from 35–40 epochs.

**Try:** `--epochs 40` combined with Idea A.

---

## Idea C — Attention-weighted column pooling (medium effort)

**Motivation:** Flat concatenation treats all 3 columns equally. In practice, HISTOPATHOLOGICAL
SUMMARY likely carries more diagnostic signal than ANCILLARY TESTS for most cases. An attention
mechanism could learn to weight columns per (case, label) pair.

**Sketch:**
```python
# Instead of: x = cat([col1, col2, col3, label])  # 3072-dim
# Do:
col_embs = stack([col1, col2, col3])         # (3, 768)
attn_scores = linear(cat([col_embs, label.expand(3,768)]))  # (3, 1)
attn_weights = softmax(attn_scores, dim=0)   # (3, 1)
report_emb = (attn_weights * col_embs).sum(0)  # (768,)
x = cat([report_emb, label])                 # 1536-dim — same as original!
```

This keeps the final classifier at 1536-dim while letting it learn column importance.
**Requires architecture changes** and a cold start (new checkpoint format, same cache valid).

**Expected gain:** Uncertain. May help for groups where one column dominates (e.g. ANCILLARY TESTS
is decisive for mast cell tumors via special staining). Could also hurt if the attention collapses.

---

## Idea D — Per-column independent scoring (medium effort)

**Motivation:** The current 3072-dim concatenation feeds all three column embeddings into a single
MLP, which must learn cross-column interactions with limited data. An alternative is to score each
column against the label independently and then aggregate:

```
concat(col1_emb ‖ label_emb) → MLP → score1
concat(col2_emb ‖ label_emb) → MLP → score2
concat(col3_emb ‖ label_emb) → MLP → score3
    ↓
aggregate (max or weighted sum) → final score
```

**Advantages:**
- Smaller input per pass (1536-dim vs 3072-dim) — easier to train at current data volumes
- Empty columns contribute no score rather than injecting a block of zeros into the input
- Per-column scores are naturally interpretable (consistent with `column_scores.csv`)
- If MLP weights are shared across columns, more parameter-efficient

**Disadvantages:**
- Loses cross-column interactions — the MLP can no longer learn "HISTOPATHOLOGICAL SUMMARY says X
  *and* FINAL COMMENT says Y → present". This may matter since FINAL COMMENT is the pathologist's
  conclusion and HISTOPATHOLOGICAL SUMMARY is the raw findings; they can be complementary.
- Requires deciding on an aggregation strategy — max is simplest but essentially replicates the
  cosine similarity approach with a learned scorer.

**Open question:** Whether cross-column interactions are learnable at 5,788 cases and a 12:1
compression ratio (3072→256). If not, the per-column approach may perform equally well with a
simpler architecture.

**Requires architecture changes** and a cold start (new checkpoint format). Cache is still valid.

---

## Idea E — Two-layer MLP (low-medium effort)

**Motivation:** Current network: `Linear(3072→256) → ReLU → Dropout → Linear(256→1)`.
Adding a second hidden layer (`256→128` or `512→256`) gives the model more representational capacity.

**Requires architecture changes** (hidden_dim2 parameter) and a cold start (different state_dict keys).
Cache is still valid.

**Expected gain:** Small. Deep networks tend to help more when the task is complex and the dataset
is large. With ~100k training pairs, a second layer is unlikely to matter much.

---

## Idea F — More keyword data (highest long-term impact, not a code change)

**Motivation:** Every phase improvement that materially broke a ceiling came from more/better data
(Phase 11: 1,273 → 5,788 cases, +17pp Good+Slight) rather than architecture changes.

At 5,788 cases across 44 groups, some groups have very few confirmed examples. If keyword coverage
improves to ~10,000+ cases, expect:
- CO floor to drop from ~30% to ~20–25%
- GroupClassifier to become viable (currently overfits at 5,788 cases)
- Re-train from cold start when new `keyword_predictions.csv` is available

---

## Recommended order of experiments

1. **Idea A (hidden_dim=512)** — run c1 with Phase 13 bank/cache still in place, compare to Phase 13 c3–c6
2. If A improves: run full cycle sequence with hidden_dim=512
3. **Idea B (epochs=40)** — combine with whichever hidden_dim wins
4. **Idea D (per-column independent scoring)** — worth testing if A+B plateau; simpler architecture may generalise better at current data volumes
5. Skip Idea C/E unless A+B+D all plateau below Phase 13 — the implementation cost isn't worth marginal gains
6. Wait for **Idea F** (more keyword data) for the next step-change improvement

---

## What a cold start requires vs. reuse

| Change | Cache valid? | Bank valid? | Checkpoint valid? |
|--------|-------------|-------------|-------------------|
| `--hidden-dim` change | Yes | Yes | No (retrain from scratch) |
| `--epochs` change | Yes | Yes | No (retrain from scratch) |
| Attention architecture (Idea C) | Yes | Yes | No (new architecture) |
| Per-column independent scoring (Idea D) | Yes | Yes | No (new architecture) |
| Two-layer MLP (Idea E) | Yes | Yes | No (new architecture) |
| New keyword data (Idea F) | No | No | No (full cold start) |

All hyperparameter experiments can reuse the existing embedding cache (`ml/data/embedding_cache.npz`)
and CO bank (`ml/output/evaluation/evaluation_co_bank.csv`). Only delete `presence_classifier_current.pt`
before starting a new hyperparameter experiment.
