# Ideas to Try — 4-Stage Pipeline Improvement Plan

Structured improvement plan for pushing Good+Slightly off on the 4-stage pipeline
(CasePresenceClassifier → GroupClassifier → LabelPresenceClassifier → KW correction)
toward 65% G+S on the held-out test set.

**Current baseline (concat-3 + per-section contrastive + 4-stage with per-LP thresholds + tail-gate, verified 2026-05-13, eval-half unbiased):**
G+S = **62.1%** (Good 46.1% + Slightly 16.0%) | CO = 14.7% | FP = 2.3% | FN = 20.8% | n=4,414 eval-half (`pipeline_eval_half/evaluation_history.csv`, full test n=8,835 → G+S 62.3%).

> Historical references: the prior 17-LP TF-IDF baseline (now preserved at `ml-tfidf/`) measured G+S = 59.5% (Good 25.7% + Slightly 33.8%) / CO = 23.4% / FN = 9.2% / 16,902 test cases. The original 25-group Phase 28 setup (pre-cold-start) was 57.9% G+S. The bottleneck table below is the 17-LP analysis preserved for failure-mode reference; verdict ownership has shifted under concat-3 (much more error now lives in Stage 1 gate / FN than in Stage 3a / Slightly Off).

**Target:** 65% G+S (+2.9 pp from the current 62.1% baseline).

Run `run_evaluation.py --stage all --test-cases ml/output/splits/test_cases.txt`
after each tier to record the marginal change before moving on. Per
CLAUDE.md "Embedding & Classifier Versioning", archive checkpoints under
`ml/output/archive/YYYY-MM-DD_<short>/` before any retrain that invalidates
downstream classifiers.

---

## Bottleneck Analysis (verified on Phase 28 / 17-LP TF-IDF test set — historical reference)

> Preserved as failure-mode reference. Under concat-3 (G+S 62.1%) verdict ownership shifted: Slight collapsed 33.8% → 16.0% and CO collapsed 23.4% → 14.7%, but FN grew 9.2% → 20.8% (stricter gate at 0.85). Rerun a fresh bottleneck pass on `pipeline_eval_half/evaluation.csv` before quoting these percentages as current. The verified file:line root causes below have *partially shipped* — see the per-item status updates.

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

### Verified root causes (file:line — shipping status as of 2026-05-13)
1. ~~`ml/training/label_presence/train.py:138-140` — `train_test_split(stratify=targets)`~~ — **SHIPPED**. `train.py:151` now uses `GroupShuffleSplit(groups=case_ids)` (case-disjoint). Same-case leakage closed; val scores no longer inflated by the old split.
2. `ml/training/label_presence/build_training_pairs.py:163` — `rng.sample(remaining, n_random)` is still the random-negative path, but **hard-neg mining infrastructure is in place** behind `--label-presence-hard-neg-fraction` (precomputed `hard_rank` at L107-114, mixed in at L153-156). Default fraction is 0.0 (random-only) after QW1 at fraction=0.7 net-negative — see "QW1 followups" below.
3. `ml/training/label_presence/train.py:62, 168` — `dropout`, `weight_decay`, `patience` are now flag-plumbed (`--label-presence-{dropout,weight-decay,patience}`); defaults remain at the Phase 28 values pending a cleaner sweep.
4. ~~Single global `label_presence_threshold=0.5`~~ — **SHIPPED**. `lp_thresholds.json` is the production default (see `ideas-accepted.md → Per-LP threshold calibration`); production loads per-LP thresholds in `stages/label_presence_classifier.py` with the CLI default as fallback.
5. `ml/production/petbert_pipeline/stages/keyword_correction.py` — Adenomas still has no entry in `subtype_keywords.py`. QW4 below remains open.

---

## Tier 1 — Quick Wins (1–2 days each, no embedding regen)

### QW1 — Hard within-group negative mining ★ FIRST MOVE

**Status:** Attempted 2026-05-10 at fraction=0.7, both alone and bundled with
QW2 — **net-negative in both configurations, REVERTED**. Best G+S:
- QW1 alone, LP-t=0.5: 58.4% (−1.1 pp vs Phase 28 baseline 59.5%)
- QW1+QW2 bundle, LP-t=0.5: 58.6% (−0.9 pp)

Hard-neg mining preserved recall but compressed sigmoid scores toward 0.5
(calibration shift), inflating prediction count ~20 % and Slightly-Off.
QW2 regularization didn't undo the shift. Phase 28 LPs restored to live
checkpoints; bundle and QW1-only LPs archived. Code infrastructure
(`--label-presence-hard-neg-fraction`, case-disjoint split, early stopping)
left in place as opt-in; defaults flipped back to Phase 28 values. Full
breakdown: `training-log-label-presence.md` Phase 30 / Phase 30b.

**Open follow-ups that may still work:**
- fraction=0.3 or 0.5 (less aggressive) — untried.
- QW3 (per-group threshold calibration) on top of QW1-bundle — directly
  counters the calibration shift; the bundle's LP-t=0.8 result (Good 27.4 %,
  CO 22.8 % — both *beat* Phase 28) suggests per-group thresholds could
  recover the lost FN budget.

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

**Status:** Implemented 2026-05-10 alongside QW1 (Phase 30b bundle). Code landed:
`GroupShuffleSplit` in `train.py`, new flags
`--label-presence-{dropout,weight-decay,patience}` in `run_training.py`,
optional early-stopping loop. Case-disjoint split confirmed Phase 29's val
scores were inflated by case leakage (same-model val score dropped on the
new split). Combined with QW1 at fraction=0.7 + wd=1e-2 + dropout=0.2 +
patience=5 + recall_weight=0.35, the bundle was **net-negative on G+S** —
QW2's regularization didn't undo QW1's calibration shift. Defaults reverted
to Phase 28 values; the flags remain available as opt-in. The case-disjoint
split itself is strictly better than the old target-stratified split and
stays the new default.

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

### QW3 — Per-group LP threshold calibration (no retrain) — **SHIPPED 2026-05-10**

Implemented as `ml/scripts/sweep_lp_thresholds.py` (one consolidated JSON, not per-LP files). Produces `ml/output/checkpoints/label_presence/lp_thresholds.json`; production auto-loads it via `--label-presence-thresholds-json` with the CLI default as fallback. Measured gain on test eval-half: +8.7 pp top-1 exact-term Good (+0.082 macro F1 / +0.19 micro F1). See `ideas-accepted.md → Per-LP threshold calibration` for the full entry.

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

### LB2 — Round 2 backbone fine-tuning (hard-neg) on the post-TF-IDF backbone

**Status:** Not started. **(Refreshed from previous Tier 4a.)** Now more attractive given QW1 (LP-layer hard-neg) was net-negative at fraction=0.7 (Phase 30, 2026-05-10) — the same hard-neg signal at the *embedding* layer is the next place to try.

**Problem:** The current production backbone (`ml/output/checkpoints/contrastive/model.safetensors`, 2026-05-07 05:18) was trained **Round 1 only** — single 3-epoch InfoNCE pass with in-batch negatives on TF-IDF-selected text, no hard-negative loss. The last hard-neg backbone pass on record is Phase 23 Run 8 (2026-04-28), which predates the TF-IDF text-selector format and is no longer the production backbone. The on-disk `hard_neg_pairs.csv` (2026-04-28 12:35) was built from pre-TF-IDF report text — it must be re-mined before any Round 2 attempt.

**Approach:**
1. Mine fresh hard-neg triplets from current Phase 28 production errors. New script `ml/scripts/mine_phase28_confusions.py` — read `output/production/contrastive/petbert_predictions.csv` + annotation CSV, emit `output/training/contrastive/hard_neg_pairs.csv` with (report, correct_label, wrong_label) triplets where `report` is the same TF-IDF-selected text the current backbone was trained on (use `text_selection.get_selector(...)`).
2. Warm-start from current backbone: `--mode adapt-backbone --model ml/output/checkpoints/contrastive --skip-pair-build --hard-neg-csv … --hard-neg-weight 0.25 --hard-neg-margin 0.3 --lr 1e-5 --epochs 2`. Phase 21 found weight=0.25 the right setting; weight=0.5 (Phase 20) regressed.
3. Cold-start cache + Stage 1 + Stage 2 + Stage 3a per CLAUDE.md "Cold Start Protocol":
   ```bash
   rm -f ml/output/training/embedding_cache.npz
   ```
4. Archive the full old generation (embeddings + backbone + all classifiers) under `ml/output/archive/YYYY-MM-DD_<short>/` before starting (CLAUDE.md "Embedding & Classifier Versioning").

**Cost:** ~1 day backbone fine-tune + ~3h embed cache regen on 58k reports + downstream retrains. Total ~2 days, not 2 weeks — Round 1 only takes ~33 minutes; Round 2 is the same shape.

**Expected gain:** +1 to +3 pp G+S. Lower-bounded estimate now that QW1 demonstrated the LP layer cannot absorb the hard-neg signal cleanly — pushing it into the embedding space is the alternative lever.

**Risk:** Phase 20 showed Round 2 with weight=0.5 *regresses* G+S; Phase 21 with weight=0.25 was −0.5pp on the old data. Use the Phase 21 weight setting and verify against current Phase 28 baseline before locking in.

---

## Recommended Sequencing

QW1 was attempted at fraction=0.7 (Phase 30, 2026-05-10) and was **net-negative**: G+S 59.5% → 58.4%. The LP-layer hard-neg signal squeezed sigmoid scores toward 0.5 and inflated Slightly-Off; per-group recall mostly held but macro precision fell. **Phase 28 LPs have been restored to live `ml/output/checkpoints/label_presence/`** (byte-identical to `archive/2026-05-10_pre-QW1-hardneg/label_presence/`); QW1-only and QW1+QW2 bundle outputs are archived at `archive/2026-05-10_QW1-fraction-0.7/` and `archive/2026-05-10_QW1+QW2-bundle/` respectively. Sanity check 2026-05-10 reproduced the 59.5% baseline (`evaluation_history.csv` row #33).

1. **Decide on QW1 first** — either revert to Phase 28 LPs (rollback target on disk) or try the QW1 fallback at fraction=0.5 / bundle with QW2. Both paths are cheap.
2. **No-retrain wins in parallel:** QW3 (per-group threshold calibration), QW4 (Adenomas keywords), QW5 (soft-tissue threshold). None invalidate downstream.
3. **Re-evaluate.** If G+S ≥ 63%, push to MI1 (bilinear head) for the 65% target.
4. **Tier 3 if Tier 1 + MI1 stalls below 62%.** LB2 is now the most attractive Tier-3 move — QW1's LP-layer hard-neg failure is itself evidence that the hard-neg signal needs to live in the embedding space, not the LP head. LB1 is the riskiest swing — reserve.

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

**Baseline to beat:** G+S = **59.5%** (Phase 28 LPs restored 2026-05-10, 17-group LPs on Phase 29 cold-start backbone, 4-stage, lp-t=0.5, group-t=0.85). Verified `evaluation_history.csv` row #33.
