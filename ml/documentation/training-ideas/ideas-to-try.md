# Ideas to Try — 4-Stage Pipeline Improvement Plan

Structured improvement plan for pushing Good+Slightly off on the 4-stage pipeline
(CasePresenceClassifier → GroupClassifier → LabelPresenceClassifier → KW correction)
toward 65% G+S on the held-out test set.

**Current baseline (Phase 28, 2026-05-07, lp-t=0.5, group-t=0.85, 4-stage):**
G+S = 57.9% | CO = 23.4% | FP = 8.0% | FN = 9.2% | 16,902 test cases

**Target:** 65% G+S (+7.1 pp).

Run `run_evaluation.py --stage all --test-cases ml/output/splits/test_cases.txt`
after each tier to record the marginal change before moving on. Per
CLAUDE.md "Embedding & Classifier Versioning", archive checkpoints under
`ml/output/archive/YYYY-MM-DD_<short>/` before any retrain that invalidates
downstream classifiers.

---

## Bottleneck Analysis (verified on Phase 28 test set)

### Verdict distribution (16,902 test cases)
| Verdict | % | Owning stage |
|---|---|---|
| Good | 25.7% | — |
| **Slightly Off** | **33.8%** | **Stage 3a/3b** (correct group, wrong label within group) |
| **Completely Off** | **23.4%** | **Stage 2** (wrong group) |
| False Positive | 8.0% | Stage 1 gate |
| False Negative | 9.2% | Stage 1 gate (~5%) + Stage 2 no-group fallback (~4%) |

### Per-stage health
- **Stage 1 (CasePresence):** P=0.91 / R=0.95 / F1=0.93 / AUC=0.98 — saturated, not a bottleneck.
- **Stage 2 (Group):** macro F1=0.78, top-1=82.7%, **top-3=97.5%**. The right group is in the top 3 nearly always but ranked below threshold or beaten by a neighbor. Hyperparameter tuning exhausted (focal/cosine-LR/lr=2e-5/27-group all failed — see ideas-rejected.md and training-log-group.md).
- **Stage 3a (LP):** validation F1 0.76–0.96, **production precision 0.29–0.61**. The 1536-dim `[report ‖ label] → 512 → 1` MLP overfits. Worst groups: Osseous P=0.30 (894 FP / 402 TP), Gliomas P=0.18, Uncommon P=0.11 (8,645 FP / 1,183 TP).
- **Stage 3b (KW):** filter-only — never suppresses. Adenomas group has no organ-specific subtype keywords → effectively a no-op for 17.6% of the test set.

### Worst groups by Good-rate (most addressable error)
| Group | Test cases | % of test | Good % | Notes |
|---|---|---|---|---|
| **Adenomas and adenocarcinomas** | 2,969 | 17.6% | **18.1%** | Largest single drag; missing subtype keywords |
| Soft tissue tumors and sarcomas | — | — | 46.8% Good, P=0.50 | Over-predicts |
| Uncommon | 421 | 2.5% | 3.1% | Inherent — 25 merged groups |

### Verified root causes (file:line)
1. `ml/training/label_presence/train.py:138-140` — `train_test_split(stratify=targets)` is target-stratified but **NOT case-disjoint**. Same case can appear in train and val with different labels → optimistic val score, masking the val/prod gap.
2. `ml/training/label_presence/build_training_pairs.py:101` — `rng.sample(neg_pool, k)` picks **random** within-group negatives. The MLP learns lexical label cues rather than discriminative report features.
3. `ml/training/label_presence/train.py:62, 168` — `dropout=0.3`, `weight_decay=1e-4` hardcoded. Fixed `epochs=25`, no early stopping.
4. `ml/production/petbert_pipeline/stages/__init__.py:111-122` — Single global `label_presence_threshold=0.5` across 17 LPs whose precisions span 0.18–0.96.
5. `ml/production/petbert_pipeline/stages/keyword_correction.py:38-45` — Adenomas has no entry in `subtype_keywords.py`; `filter_by_subtype` returns the input pool unchanged.

---

## Tier 1 — Quick Wins (1–2 days each, no embedding regen)

### QW1 — Hard within-group negative mining ★ FIRST MOVE

**Status:** Not started.

**Problem:** Random within-group negatives (`rng.sample`, `negs_per_pos=5`) make the LP task too easy in training. The MLP overfits to lexical cues from label strings and fails in production where confusable subtypes co-occur (Osseous: 894 FP / 402 TP; Gliomas: P=0.18).

**Hypothesis:** Mining negatives by cosine similarity between the *positive label embedding* and *other in-group label embeddings* forces the MLP to learn discriminative report features.

**Approach:**
1. Modify `ml/training/label_presence/build_training_pairs.py:96-103` — replace `rng.sample(neg_pool, k)` with cosine-ranked top-k from cached `label_embeddings`. Reuse `production.petbert_pipeline.embedding_cache.load_cache` (already returns `label_embeddings`) and `production.petbert_pipeline.embedding.cosine_similarity_matrix`.
2. Add `--label-presence-hard-neg-fraction` flag (default 0.7) to `ml/scripts/run_training.py` `_train_label_presence` helper. Mix hard + random negatives at the configured fraction.
3. Retrain all 17 LP classifiers (~30 min total on XPU).

**Cost:** 1 day. No embedding regen. Existing data sufficient.

**Expected gain:** +2.5 to +4.0 pp G+S (recovers 10–15% of Slightly-Off, target macro precision 0.5–0.75).

**Risk:** Per-group recall could dip on rare labels. **Watchpoint:** if any per-group recall drops >10 pp, dial fraction to 0.5.

---

### QW2 — Case-disjoint LP val split + dropout/weight-decay tuning + early stopping

**Status:** Not started.

**Problem:** Current target-stratified split (`train.py:138-140`) leaks the same case across train and val with different labels, so val scores are optimistic. Combined with hardcoded `weight_decay=1e-4` and `dropout=0.3`, regularization is weak and the val→prod precision collapse is unsurprising.

**Hypothesis:** A case-disjoint split + stronger regularization + early stopping closes the val/prod gap.

**Approach:**
1. `ml/training/label_presence/train.py:137-140` — replace with `GroupShuffleSplit(groups=case_ids)` so train and val never share a case.
2. Add `--weight-decay` (try 1e-2) and `--patience` (5) plumbed through `ml/scripts/run_training.py`.
3. Lower default `dropout` from 0.3 → 0.2 (line 62).
4. Change `recall_weight` default 0.5 → 0.35 so checkpoint selection prefers precision (matches the production failure mode).
5. Retrain all 17 LP classifiers.

**Cost:** 0.5 day. 17 LP retrains.

**Expected gain:** +1.0 to +2.0 pp G+S (cumulative with QW1).

**Risk:** Smaller groups (Osseous, Gliomas) lose effective positives under a case-disjoint split. **Mitigation:** k-fold averaging of the *score*, not the weights.

---

### QW3 — Per-group LP threshold calibration (no retrain)

**Status:** Not started.

**Problem:** A single `label_presence_threshold=0.5` across 17 heterogeneous LPs (production precision span 0.18–0.96) is wrong by construction. The optimal sigmoid threshold is group-specific.

**Hypothesis:** Pure post-hoc calibration on a held-out fold lifts precision without retraining.

**Approach:**
1. New script `ml/scripts/calibrate_label_presence.py` (read-only on weights). For each LP: load weights, run on a train-cases holdout fold, sweep thresholds 0.3–0.9, pick the threshold that maximizes F0.7 (precision-weighted).
2. Persist as `{group}.threshold.json` next to each `.pt` in `output/checkpoints/label_presence/`.
3. Modify `ml/production/petbert_pipeline/stages/label_presence_classifier.py::score_within_group` to read per-group thresholds, falling back to the CLI default when missing.
4. Wire through `categorize_per_case` — pass per-group threshold dict alongside the existing `label_presence_threshold` arg.

**Cost:** 0.5 day. No retrain, no embedding regen.

**Expected gain:** +0.7 to +1.5 pp G+S.

**Risk:** Calibration set overfit. Use a fold disjoint from training; verify on test.

---

### QW4 — Adenomas keyword expansion (Stage 3b uplift)

**Status:** Not started.

**Problem:** Adenomas at 18.1% Good × 17.6% test share = ~14% of total error budget addressable here. ICD-O behavior digit alone (benign/malignant) does not disambiguate organ-of-origin (mammary vs apocrine vs sebaceous adenocarcinoma). `subtype_keywords.py` has no entry for this group, so Stage 3b is a no-op for it.

**Hypothesis:** Adding organ-keyword filters tightens label selection without retraining.

**Approach:**
1. Curate organ markers against `ml/ICD_labels/labels.csv` for the "Adenomas and adenocarcinomas" group: mammary, apocrine, sebaceous, hepatoid, perianal, thyroid, ceruminous, hepatocellular, biliary, pancreatic, renal, prostatic, etc.
2. Add the entry to `ml/ICD_labels/subtype_keywords.py`. `filter_by_subtype` already preserves the no-match → original-pool fallback (verified at `keyword_correction.py:36-37`) — keep that contract.
3. Re-run production; no retraining.

**Cost:** 1 day, mostly curation.

**Expected gain:** +1.5 to +2.5 pp G+S.

**Risk:** Empty-pool over-filter. Always preserve the fallback to the original pool when no keyword matches.

---

### QW5 — Soft-tissue threshold tightening at Stage 2

**Status:** Not started.

**Problem:** Soft tissue tumors and sarcomas has F1=0.63, P=0.50 — Stage 2 over-predicts this group. The fixed `group_classifier_threshold=0.85` is uniform across 25 groups of widely varying base rates and class confusion.

**Hypothesis:** A per-group Stage 2 threshold (one off-diagonal value for soft tissue) reduces FP at this group without retraining.

**Approach:**
1. Modify `ml/production/petbert_pipeline/stages/group_classifier.py` to accept a per-group threshold dict (default to scalar threshold).
2. Sweep soft-tissue threshold 0.85–0.95 on training holdout; pick best by per-group F1.
3. If Stage 2 retrain becomes necessary, cap soft-tissue `pos_weight` more aggressively via a custom override array threaded through `train_groups`.

**Cost:** 0.5 day for sweep; +0.5 day if retrain required.

**Expected gain:** +0.5 to +1.0 pp G+S.

**Risk:** Trades soft-tissue recall. Verify per-group recall on test.

---

## Tier 2 — Medium investments (week-scale)

### MI1 — Bilinear / low-rank LP scoring head

**Status:** Not started. **Run after Tier 1 has landed.**

**Problem:** `[report ‖ label] → 1536 → 512 → 1` is a generic feed-forward head. It cannot natively model pairwise interaction between report and label embeddings, and on small per-group positive pools it overfits.

**Hypothesis:** A bilinear scoring head `report_proj^T (U V^T) label_proj` (low rank, shared) has fewer parameters yet richer pairwise structure → better calibration.

**Approach:**
1. Add `head_type ∈ {mlp, bilinear}` constructor arg to `ml/model/label_presence_classifier.py`. Persist in checkpoint dict — follow the existing `col_pair_mode`/`col_combine` versioning pattern so old checkpoints still load.
2. Use a low-rank factorization `W = U V^T`, rank ≤ 64.
3. Add `--head-type` flag through `ml/scripts/run_training.py` and `ml/training/label_presence/train.py`. Default stays `mlp` for backward compatibility.
4. Retrain all 17 LP classifiers and compare to MLP baseline.

**Cost:** 3–4 days (model + training plumbing + 17 retrains + comparison).

**Expected gain:** +1.5 to +3.0 pp G+S **stacking on Tier 1**.

**Risk:** Bilinear with full `hidden_dim²` weights overfits on small groups. Use the low-rank factorization above.

---

### MI2 — Per-group softmax LP head (replaces N×binary)

**Status:** Not started.

**Problem:** 17 separate binary classifiers see only within-group signal and have no normalization across labels. The sigmoid threshold has to be tuned per-group (QW3) precisely because there is no cross-label calibration. A single per-group softmax over `(K_g labels + 1 "none")` would calibrate naturally.

**Hypothesis:** Per-group softmax head removes threshold tuning and produces better-calibrated within-group probabilities.

**Approach:**
1. New module `ml/model/label_presence_softmax.py` (per-group softmax classifier, K_g + 1 outputs).
2. New training entry `train_label_presence_softmax` in `ml/training/label_presence/` with cross-entropy loss.
3. Dispatcher in `ml/production/petbert_pipeline/stages/__init__.py::categorize_per_case` checks for a softmax checkpoint first, binary fallback when absent. KISS — build alongside, don't rip out the binary path.

**Cost:** 1 week. No embedding regen.

**Expected gain:** +2 to +3 pp G+S (partially overlaps QW1 + MI1).

**Risk:** "None" class spec — case has the group but the gold label is outside the taxonomy. Easy to mis-spec. Build a small audit set first.

---

### MI3 — Adenomas-bucket sub-group split at Stage 2

**Status:** Not started. **Run only if QW4 is insufficient.**

**Problem:** "Adenomas and adenocarcinomas" is too heterogeneous (2,969 test cases, 18.1% Good) for a single GroupClassifier output. Phase 28c's 27-group expansion failed because it was *upward* (Thymic/Myxomatous *promotion* with sparse data). This split is *downward* inside an already large group — children stay >500 cases each.

**Hypothesis:** Splitting Adenomas by anatomic site (mammary / cutaneous-adnexal / GI / endocrine) directly attacks the largest Slightly-Off contributor.

**Approach:**
1. Add a sub-group remapping table; expose via `--subgroup-map` CSV in `ml/training/group/build_training_data.py`.
2. Retrain Stage 2 GroupClassifier with the new groups.
3. Retrain LPs for affected children. `keyword_correction.py` becomes the soft splitter when GroupClassifier is unsure.
4. **Gate:** only split a child if it has ≥300 train cases.

**Cost:** 1 week (taxonomy remap + Stage 2 retrain + child LP retrains + evaluation).

**Expected gain:** +1.5 to +3.0 pp G+S on top of QW4.

**Risk:** Repeats the 27-group degradation pattern. Strictly enforce the ≥300-cases gate per child.

---

## Tier 3 — Larger bets (2–3 weeks)

### LB1 — Generative label decoder (Flan-T5-small) replacing Stage 3a

**Status:** Not started. **Reserve until Tier 1 + MI1 are exhausted.**

**Problem:** Cosine + binary heads are fundamentally limited — they score `(report, label)` pairs independently. A small seq2seq decoder constrained to the taxonomy vocabulary by trie-decoding could replace Stage 3a entirely with one model that reads the report and outputs the label string.

**Approach:**
1. Encoder = frozen adapted PetBERT in `output/checkpoints/contrastive/`.
2. Decoder = Flan-T5-small init; constrained beam search over the label-string trie built from `ml/ICD_labels/labels.csv`.
3. New training pipeline; new stage `ml/production/petbert_pipeline/stages/generative_decoder.py` that supplants Stage 3a.

**Cost:** 2–3 weeks. No PetBERT regen.

**Expected gain:** +3 to +6 pp G+S if it works; could land flat.

**Risk:** Highest of any idea here. Hallucination on Uncommon. Reserve until Tier 1 + MI1 results are in hand.

---

### LB2 — Re-do contrastive backbone with Phase-28-mined hard negatives

**Status:** Not started. **(Refreshed from previous Tier 4a.)** Run only if Tier 1 + MI1 stalls below 62%.

**Problem:** The Phase 17 contrastive backbone wasn't tuned against the specific Phase 28 confusion matrix. Mining hard-neg triplets from current production errors would push embeddings of confusable labels apart. (CO is now 23.4%, mostly within adjacent groups.)

**Approach:**
1. New script `ml/scripts/mine_phase28_confusions.py` — read `output/production/contrastive/petbert_predictions.csv` + annotation CSV, emit `output/training/contrastive/hard_neg_pairs.csv` with (report, correct_label, wrong_label) triplets.
2. Run `--mode adapt-backbone --hard-neg-csv …`.
3. Cold-start of cache + Stage 2 + Stage 3a per CLAUDE.md "Cold Start Protocol":
   ```bash
   rm -f ml/output/training/embedding_cache.npz
   ```
4. Archive the full old generation (embeddings + backbone + all classifiers) under `ml/output/archive/YYYY-MM-DD_<short>/` before starting (CLAUDE.md "Embedding & Classifier Versioning").

**Cost:** 2 weeks (PetBERT regen on 58k reports is the long pole).

**Expected gain:** +2 to +4 pp G+S, but heavily overlaps QW1.

**Risk:** Cold-start cost is large; if QW1 already addresses the same hard negatives at the LP layer, marginal gain is small. Run only after QW1 results land.

---

## Recommended Sequencing

1. **Week 1:** QW1 (hard negatives) + QW2 (case-disjoint split + regularization) bundled into one LP retrain cycle. Land QW3 (threshold calibration) on top of the new LPs.
2. **Week 1 (parallel, no retrain conflict):** QW4 (Adenomas keywords) + QW5 (soft-tissue threshold).
3. **Re-evaluate.** If G+S ≥ 63%, push to MI1 (bilinear head) for the 65% target.
4. **Tier 3 only if Tier 1 + MI1 stalls below 62%.** LB2 is most attractive once QW1 hard-neg patterns are known. LB1 is the riskiest swing — reserve.

---

## Evaluation Reference

Standard test-set evaluation command after any change:
```bash
ml/.venv/Scripts/python.exe ml/scripts/run_evaluation.py \
  --stage all --test-cases ml/output/splits/test_cases.txt \
  --out-dir ml/output/evaluation/contrastive_test \
  --label "<tier and description>"
```

Per-stage isolation when an end-to-end change moves the needle:
```bash
ml/.venv/Scripts/python.exe ml/scripts/run_evaluation.py --stage case-presence  --test-cases ml/output/splits/test_cases.txt
ml/.venv/Scripts/python.exe ml/scripts/run_evaluation.py --stage groups         --test-cases ml/output/splits/test_cases.txt
ml/.venv/Scripts/python.exe ml/scripts/run_evaluation.py --stage label-presence --test-cases ml/output/splits/test_cases.txt
```

Results append to `ml/output/evaluation/contrastive_test/evaluation_history.csv` and the per-stage `*_history.csv` files.

**Baseline to beat:** G+S = 57.9% (Phase 28, 2026-05-07, 4-stage, lp-t=0.5, group-t=0.85).
