# Ideas Rejected

Ideas that were implemented and evaluated but caused regression or no improvement. Preserved to avoid repeating failed experiments. Each entry records what was tried, what happened, and when (if ever) to revisit.

---

## Label Embedding Enrichment — Diagnosis Strings

**Status:** Tried (Phases 6–8, 2026-03-03) — **minimal impact; do not use**

**Idea:** Label text is minimal (`"{term} {group}"`). Blend each label's embedding 50/50 with the mean embedding of its keyword-matched diagnosis strings (e.g. `"SKIN DORSUM: SQUAMOUS CELL CARCINOMA"`), adding clinical vocabulary to close the domain gap.

**What happened:** The diagnosis strings and label strings already occupy nearly the same compact region of PetBERT's embedding space — both are short anatomic phrases. Blending two vectors that are already close together barely moved the label embedding. The domain gap to full clinical report embeddings (HISTOPATHOLOGICAL SUMMARY, FINAL COMMENT, ANCILLARY TESTS) remained. Severe oscillation (±13pp) emerged as old CO bank pairs from the unenriched space conflicted with enriched-space predictions. Even after a bank reset and full warm-up, the Phase 8 ceiling was 22.1% Good+Slight — not better than Phase 5 (22.9%) without enrichment. The enrichment did not help.

**Code:** `ml/labels/enrichment.py` — original diagnosis-text version replaced by Fix 9 (see below), then abandoned entirely after Phase 10.

---

## Label Embedding Enrichment — Report Embedding Blend (50/50)

**Status:** Tried (Phase 10, 2026-03-04) — **regressed; do not use**

**Idea:** Instead of blending with diagnosis strings, blend each label's embedding 50/50 with the centroid of cached report embeddings for keyword-confirmed cases. Report embeddings are already in the cache — no extra PetBERT inference needed. This directly reduces the distance the classifier must bridge between report and label representations.

**What happened:** The hybrid embeddings shifted the cosine score landscape enough to invalidate the accumulated CO bank, requiring a full cold start. After 18 cycles (including co=5 warm-up and co=10 main phase), the ceiling was ~13% Good+Slight — significantly below Phase 9 (17.5–20.4%) without enrichment. The 50/50 blend creates a point in embedding space that is neither a good label representative nor a good report representative. The classifier's input (mean_report_embedding) is now compared against a blended label embedding that was pulled toward report space, but report embeddings vary widely per case while the blend uses a centroid — the result is a noisy midpoint that hurts, not helps.

**Conclusion:** Label enrichment in any form does not help with the current PetBERT backbone. The right fix for the domain gap is contrastive fine-tuning of the backbone itself (Phase 17), which directly optimises the embedding space for (report, label) alignment.

---

## Per-Pair Architecture with Max-Pooling

**Status:** Tried (Phase 14, 2026-03-21) — **REGRESSION −7.3pp; do not use**

**Idea:** Instead of concatenating all columns with the label (`[col1 | col2 | col3 | label]` → 3072-dim), score each column-label pair independently with a shared MLP (`[colN | label]` → 1536-dim), then max-pool across the three per-column logits. Motivation: halves the input compression ratio (12:1 → 6:1) and separates column-label relationships.

**What happened:** Phase 14 peaked at 32.7% Good+Slight vs Phase 13's 40.0% (−7.3pp) after 7 cycles.

**Why it failed:**
1. **Shared MLP weights across columns** — FINAL COMMENT, HISTOPATHOLOGICAL SUMMARY, and ANCILLARY TESTS have fundamentally different linguistic patterns; a shared MLP can't learn column-specific representations.
2. **Max-pooling amplifies noise** — if one column incidentally mentions a label (differential, ruled-out finding), max-pooling promotes that noisy score above the signal from the correct column.
3. **Loss of cross-column interaction** — the concat model learns "col1 says X *and* col2 says Y → label Z"; per-pair treats columns as independent and discards those joint patterns.

**Backup:** `ml/output/checkpoints/binary/presence_classifier_best_phase14_perpair.pt`

---

## Per-Pair Architecture with Learned Column Combination

**Status:** Tried (Phase 15, 2026-03-22) — **REGRESSION −5.2pp; do not use**

**Idea:** Same per-pair shared MLP as Phase 14, but replaces hard max-pooling with a learned `Linear(3 → 1)` combiner — 4 extra parameters that learn globally which columns carry the most diagnostic signal.

**What happened:** Phase 15 peaked at 34.8% Good+Slight (c5). Better than Phase 14 (32.7%) — the learned combiner correctly deweights noisy columns vs hard max — but still −5.2pp below Phase 13 (40.0%). The shared MLP weights remain the fundamental problem: column identity is lost regardless of how scores are combined.

**Backup:** `ml/output/checkpoints/binary/presence_classifier_best_phase15_learned.pt`

---

## KNN Group Gating (Hybrid Architecture)

**Status:** Tried (2026-03-23) — **massive regression; do not retry until ~15k cases**

**Idea:** Use the existing KNN group selector as a gate: only allow argmax over groups that receive ≥ threshold vote fraction from KNN neighbors. Cases with no group above threshold → Uncategorized. Both checkpoints already exist — no new training required. Hypothesis: reduces CO by restricting the label search space to plausible groups.

**What happened:** Tested at thresholds 0.1, 0.2, 0.3. All caused massive regression vs binary-only baseline (37.8% Good+Slight):

| Mode | Good+Slight | CO% | FP% | FN% |
|------|------------|-----|-----|-----|
| Binary only (Phase 16) | 37.8% | 30.1% | 30.3% | 1.8% |
| Hybrid t=0.1 | 5.6% | 37.6% | 53.0% | 3.8% |
| Hybrid t=0.2 | 7.5% | 31.9% | 47.7% | 13.0% |
| Hybrid t=0.3 | ~6.1% | ~28% | ~39% | 26.6% |

**Why it failed:**
1. **Slight% collapsed** — binary-only's Slight predictions come from the argmax naturally landing in the right group. The KNN gate excludes whole groups; if the wrong groups are voted in, the correct group is fully removed → CO or FN. Slight rate: 27.5% → 1.9–2.5%.
2. **KNN can't gate non-cancer cases** — even at t=0.1, FP jumped to 53%. Non-cancer cases find cancer neighbors in embedding space regardless; KNN cannot distinguish them.
3. **KNN sparsity causes FN** — reference set built from LLM predictions (~30% cancer prevalence, ~150 cases/group). Many true cancer cases don't have enough same-group neighbors to pass any threshold.

**Code preserved:** `run_categorization_hybrid()` in `categorization.py`. Revisit when database grows past ~15k cases.

---

## Hard-Negative Fine-Tuning of PetBERT Backbone

**Status:** Tried (Phases 20–21, 2026-03-27) — **regression at all weights; cannot break the ~70% ceiling**

**Idea:** After InfoNCE fine-tuning (Phase 17), augment backbone training with margin loss on hard-negative triplets from the CO bank: `(report, correct_label, wrong_label)`. The CO bank contains ~24.3k wrong-group predictions — these are exactly the cases where the model confuses labels across groups. Push those apart directly in embedding space during backbone training.

**What happened:** Two weight settings tried, both regressing vs Phase 18's 70.4%:

| Config | Best Good+Slight | CO% | vs Phase 18 |
|--------|-----------------|-----|------------|
| weight=0.5, margin=0.3 (Phase 20) | 68.5% | 6.9% | −1.9pp |
| weight=0.25, margin=0.3 (Phase 21) | 69.4% | 6.1% | −1.0pp |

**Why it failed:**
- Hard-neg loss consistently raised Good% (+0.7pp) but suppressed Slight% (−1.4 to −2.1pp) — net regression. The model became more conservative: it pushed away from wrong groups but also over-shot, dropping borderline Slight matches.
- CO floor marginally improved (6.1% at weight=0.25 vs Phase 18's 6.5%) but this gain was offset by the Slight% loss.
- The ~70% ceiling is a **data ceiling**, not an embedding-space ceiling. Borderline Slight cases require more labelled examples to resolve correctly, not a stronger push-away signal.

**Backups:** `model_phase20_round3_backup.safetensors` (weight=0.5), `model_phase19_round2_backup.safetensors` (Round 2, pre-hard-neg). Phase 18 best checkpoint remains `presence_classifier_best.pt` (70.4%).

**When to retry:** Not before ~8,000+ confirmed cancer cases. The margin and weight values may need tuning based on label distribution at that scale.

---

## More LLM Annotation for Minority Groups

**Status:** Not viable — **annotation pipeline is fixed; do not pursue**

**Idea:** Re-run LLM annotation on cases where the current pipeline predicts "Unidentified Cancer" to add training signal for minority groups where GroupClassifier is weakest.

**Why it won't work:** The annotation pipeline is already operating on all available data. There is no mechanism to generate additional ground-truth labels beyond what the existing pipeline produces — more annotation runs would not yield new cases, only re-annotate existing ones. The minority-group data ceiling is a fixed constraint of the current dataset size, not an annotation coverage gap.

**Alternative:** LB2 (Round 2 backbone hard-neg fine-tune on the current post-TF-IDF backbone) addresses the same CO problem from the embedding side rather than the data side.

---

## GroupClassifier — Lower Learning Rate (lr=2e-5)

**Status:** Tried (Phase 26, 2026-05-04) — **worse than lr=5e-5; do not use**

**Idea:** After training GroupClassifier at lr=5e-5 (macro F1=0.4335 at epoch 219), try a lower lr=2e-5 to see if slower convergence finds a better minimum.

**What happened:**

| Run | Best epoch | Macro F1 |
|-----|-----------|----------|
| lr=5e-5 | 219 | **0.4335** |
| lr=2e-5 | 278 | 0.4249 |

Slower convergence did not find a better minimum — lr=5e-5 is better calibrated to the data volume. `group_classifier_best.pt` was not overwritten.

**Conclusion:** Stick with `--lr 5e-5`. Hyperparameter sweep on weight-decay and max-class-weight are also unlikely to outperform Tier 3a given this F1 pattern.

---

## Lower Case Presence Threshold (gate=0.5 → 0.4)

**Status:** Tried (Phase 26, 2026-05-04) — **FP increase outweighs FN gain; do not use**

**Idea:** Lower the CasePresenceClassifier gate threshold from 0.5 to 0.4 to catch more cancer cases (reduce FN) at the cost of more false positives.

**What happened (combined with group-t=0.85):**

| Config | G+S | CO | FP | FN | Total |
|--------|-----|----|----|-----|-------|
| gate=0.5 group-t=0.85 | 52.3% | 24.0% | 4.9% | 18.8% | 9,335 |
| gate=0.4 group-t=0.85 | 51.7% | 24.1% | **5.8%** | 18.4% | 9,467 |

FP rises +0.9pp for only −0.4pp FN and G+S falls. Gate=0.5 remains better. The CasePresenceClassifier trained with recall_weight=0.85 already handles recall adequately at threshold=0.5.

---

## Per-Label Score Calibration

**Status:** Tried (2026-03-28) — **does not help with current data; do not use in production**

**Problem:** After mean-centering the score matrix, different labels still have different score variances. A label the model is uncertain about (low variance, scores clustered near 0) will lose argmax to higher-variance labels even when it is the correct answer.

**Approach:** For each label `l` with ≥10 labeled cases, find a scalar offset `b_l` (via grid search) that maximizes Good+Slight accuracy for cases whose ground truth is label `l`:

    calibrated_score_l = (score_l - mean_l) + b_l

Three objective functions were tried — all regressed in production against Phase 18 best (70.4%):
- Exact-match objective → 59.7% Good+Slight (−9.7pp) — Slight% collapsed as offsets stole wins across groups
- Group-level per-label objective → harmful even in-sample (93.7% → 89.2% on annotation set)
- Net-gain across all annotated cases → 63.9% Good+Slight (−5.5pp) — Off% controlled but Slight% still fell; total predictions dropped ~5k as calibrated argmax interacts poorly with embedding_min_sim threshold

**Conclusion:** Score calibration does not help. The variance bias is not the binding constraint — the ~70% ceiling is a data ceiling. Do not apply `--calibration-offsets` in production until more labelled cases are available.

**Code:** `ml/training/binary/calibrate.py` (v3 net-gain objective, kept for reference). Offsets file `ml/output/calibration/label_offsets.json` is empty (`{}`). See training-log-finetune.md for full results.

**When to retry:** If the dataset grows to ~8,000+ confirmed cancer cases, recalibrate from scratch against the new production best checkpoint.
