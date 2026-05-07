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

## Experiment 3 — Per-column concat + hyperparameter sweep (2026-03-23) ❌

**Hypothesis:** Switching from 768-dim `mean_embedding` to 2304-dim `col_emb_concat`
(same input as the binary PresenceClassifier) gives more signal per case. Combined with
thorough hyperparameter tuning, this could close the gap with binary at current data volume.
Also tested: per-group sample cap (`--max-group-cases=100`) to reduce overfitting in large groups.

**Architecture:** `col_emb_concat` (3 × 768 = 2304-dim) → Linear(2304→512) → ReLU →
Dropout(p) → Linear(512→17). 17 outputs (groups with ≥100 keyword-confirmed cases out of 43 total).
Loss: `BCEWithLogitsLoss` with per-class inverse-frequency weights capped at `--max-class-weight`.

**Per-group cap experiment (`--max-group-cases=100`):**

Natural class weights were too extreme (up to 119:1 for rare groups) — caused recall→1.0,
precision→0. The hypothesis was that capping the number of positive training samples per group
would reduce the imbalance. Row-removal approach: keep union of up to 100 positives per group
plus all non-cancer rows.

| max-group-cases | max-class-weight | Best macro F1 |
|----------------|-----------------|--------------|
| 100 (capped) | auto (up to 119x) | ~0.0 (all positive) |
| 100 (capped) | 20 | 0.245 |
| 0 (no cap) | 20 | 0.358 |
| 0 (no cap) | 12 | 0.409 |

**Finding:** The cap removes valid training signal. `--max-class-weight` alone is the right lever
for controlling weight explosion. Do not use `--max-group-cases`.

**Hyperparameter sweep (no cap, all data):**

| Parameter | Values tested | Best |
|-----------|--------------|------|
| `--dropout` | 0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.0 | **0.05** |
| `--max-class-weight` | 5, 10, 15, 20, 40, 12, 11, 13 | **12** |
| `--lr` | 1e-3, 5e-4, 3e-4, 2e-4, 1e-4, 7e-5, 5e-5, 3e-5 | **5e-5** |
| `--epochs` | 50, 100, 200, 300, 500, 1000, 1700, 2000 | **1700** |
| `--hidden-dim` | 256, 512 | **512** |

Key insights:
- Reducing dropout 0.3 → 0.05 gave ~+0.18 F1 — model was severely over-regularized
- Lower LR consistently helped; each halving added ~0.01–0.03 F1
- 2-layer architecture (512→256→17) regressed vs single hidden layer — reverted
- Best checkpoint: macro F1 = **0.4975** at epoch 1672

**End-to-end pipeline evaluation (vs binary Phase 16):**

| Metric | Binary (Phase 16) | GroupClassifier (best, t=0.3) |
|--------|-------------------|-------------------------------|
| Good+Slight | **41.9%** | 9.3% |
| CO% | 29.6% | ~37% |
| FP% | 27.2% | 33.3% |
| FN% | 1.2% | 16.8% |

**Why it failed:**
1. Only 17 of 43 groups trained — 26 groups unreachable at inference, causing FN spike
2. FP% (33.3%): non-cancer cases find positive neighbors in 2304-dim space regardless of threshold
3. FN% (16.8%): cases whose true group is among the 26 untrained groups are entirely missed
4. Even correct group predictions → weak term selection via within-group cosine

**Bugs found and fixed:**
- `pipeline.py` passed `embeddings` (768-dim mean) to `group_clf.predict_proba()` — should be `col_emb_concat` (2304-dim). This invalidated the Experiment 1 results (which used the wrong tensor).
- `train.py` default `--out` and `--training-data` paths included `ml/` prefix → double-nested to `ml/ml/...` when run from the `ml/` directory. Fixed to relative paths; misplaced checkpoints manually moved and `ml/ml/` deleted.
- Class weights computed before `--max-group-cases` cap remained stale (Adenoma: stored weight 11 but true post-cap ratio 92:1). Fixed by recalculating weights from capped targets. (Cap not recommended anyway.)

**Conclusion:** Hyperparameter tuning is exhausted at current data volume. The architecture is
sound (F1=0.4975 on training groups) but insufficient data coverage (17/43 groups) prevents
end-to-end performance. Do not retry until ~15,000 keyword-confirmed cases.

---

## What to Try Next

These are blocked on keyword coverage, not architecture:

| When | What |
|------|------|
| ~10,000 confirmed cases | Re-run Experiment 3 baseline; expect more groups to cross 100-case threshold |
| ~15,000+ confirmed cases | GroupClassifier expected to pull ahead of binary on CO% |

See [classifiers.md](../classifiers.md) for the discriminating-keyword term selection idea,
which may improve within-group accuracy once the group prediction step is reliable.
