# GroupClassifier Training Log

All experiments use the same dataset: 12,620 total reports, 5,788 keyword-confirmed
cancer cases across 45 groups, embedded with frozen PetBERT (Phase 13 cache).

Primary metric: **macro-averaged F1** across all groups at threshold=0.3.
Pipeline metric (for comparison with binary): **Good+Slight %** from end-to-end evaluation.

---

## Experiment 1 — Baseline (mean embedding, 2026-03-04)

**Setup:** Standard GroupClassifier on `mean_embeddings` (average of 3 columns, 768-dim).
5,788 cancer cases across 44 groups. Evaluated at thresholds 0.3 and 0.8.

**End-to-end pipeline results (for comparison with binary classifier):**

| Threshold | Good+Slight | CO% | FN% |
|-----------|-------------|-----|-----|
| 0.3 | 13.9% | 57.5% | — |
| 0.8 | 21.9% | 54.5% | 15.6% |

Binary PresenceClassifier (Phase 11, same data): 33.1% Good+Slight, 31.8% CO, 1.3% FN.

**Finding:** GroupClassifier severely overfits. Val loss >> train loss. CO rates of 54–57% far
exceed the binary classifier's floor. At threshold=0.8 the FN rate (15.6%) is unacceptable.
The model has ~132 cases/group on average — insufficient for a 45-class MLP to generalise.

---

## Experiment 2 — Priority embedding (FINAL COMMENT first, 2026-03-21) ❌

**Hypothesis:** The mean embedding dilutes the diagnostic signal. FINAL COMMENT is the
pathologist's conclusion — the most directly group-discriminating column. Use FINAL COMMENT
as the 768-dim input if present, fall back to HISTOPATHOLOGICAL SUMMARY, then ANCILLARY TESTS.

**Rationale:** No architecture change (still 768-dim input), so no increased overfitting risk.
FINAL COMMENT contains direct group-level language ("consistent with hemangiosarcoma") while
other columns contain procedural or microscopic detail.

**Implementation:** In `build_training_data.py` and `pipeline.py`, replace `mean_embeddings`
with a per-case priority selection: for each case, use the first non-empty column in order
[FINAL COMMENT, HISTOPATHOLOGICAL SUMMARY, ANCILLARY TESTS].

**Results (25 epochs, 5,864 cancer cases, device: xpu):**

| Run | Best epoch | Macro F1 | Notes |
|-----|-----------|----------|-------|
| Experiment 1 (mean) | — | 0.1020 | Previous best |
| Experiment 2 (priority) | 21 | **0.0695** | Regression — reverted |

**Per-group pattern:** Recall ≈ 1.0, precision ≈ 0 for essentially every group. The model
approves all groups for all cases — the same degenerate "everything positive" behaviour seen
when `--recall-weight` is too high in binary classifier training.

Val loss divergence: train loss 0.64 at epoch 25, val loss 2.4+. More severe overfitting
than the mean embedding baseline.

**Why it didn't work:** The priority embedding idea assumed the problem was signal quality.
It wasn't. The model still sees the same total number of training examples (~132/group average)
regardless of which column is used. The 768-dim input size is identical — no change to the
model's capacity or the data's size. The overfitting is driven purely by the data-to-parameter
ratio, not by what signal is in the embedding.

Additionally, the priority embedding introduces inconsistency across cases: some cases train
on FINAL COMMENT, others on HISTOPATHOLOGICAL SUMMARY. This may create a noisier input
distribution than the mean, which always uses all available information.

**Conclusion:** Reverted. Production checkpoint (0.1020 macro F1) unchanged. Embedding
selection is not a lever worth pursuing until the data volume constraint is resolved.

---

## What to Try Next

These are blocked on keyword coverage, not architecture:

| When | What |
|------|------|
| ~10,000 confirmed cases | Re-run baseline experiment; expect generalisation to start |
| ~10,000 confirmed cases | Try per-column concatenation (2304-dim input) as in binary Fix 10 |
| ~15,000+ confirmed cases | GroupClassifier expected to pull ahead of binary on CO% |

See [classifiers.md](classifiers.md) for the discriminating-keyword term selection idea,
which may improve within-group accuracy once the group prediction step is reliable.
