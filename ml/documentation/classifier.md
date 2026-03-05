# Classifier Development Log

## Architecture Overview

The system has **two pipelines with distinct roles**:

### Production Pipeline — PetBERT Scan

Maps **full pathology report text → cancer group + term + ICD code**. This is the only pipeline that runs in production.

| Stage | Input | Output |
|-------|-------|--------|
| Embedding | Report text columns (HISTOPATHOLOGICAL SUMMARY, FINAL COMMENT, ANCILLARY TESTS) | 768-dim mean-pooled embedding per case via frozen PetBERT |
| Scoring | Report embedding vs. taxonomy label embeddings | Score per (case, label) pair — cosine similarity, or learned probability when a classifier is present |
| Classification | Scores across all labels | Top-k predictions with confidence threshold → term + group + ICD code |

The current scorer is a binary `PresenceClassifier` MLP (`model/presence_classifier.py`) trained on `(report_embedding, label_embedding) → present/absent`. It replaces raw cosine similarity scores but is evaluated one pair at a time, which introduces a group-assignment ceiling (~42% completely-off). See [multiclass-classifier-plan.md](multiclass-classifier-plan.md) for the planned replacement.

### Training Pipeline — Keyword Scan

Maps **diagnosis field text → cancer label**, for the **sole purpose of generating ground-truth training labels for the PetBERT pipeline**. Does not run in production.

The keyword scan matches structured diagnosis strings (e.g. `"SKIN DORSUM: SQUAMOUS CELL CARCINOMA"`) against a curated keyword dictionary to assign Vet-ICD-O labels. It currently covers ~18.6% of diagnosis rows (~1,273 unique cases across 39 cancer groups). The keyword pipeline is actively being improved by a domain expert.

**Ground-truth assumption:** Cases not matched by the keyword scan are treated as **non-cancer (Uncategorized)**. This is valid for a general veterinary clinic population where ~18% cancer prevalence is expected. As keyword coverage improves, training label quality will improve accordingly.

### Training data sources (`build_training_pairs.py`) — binary PresenceClassifier only

These training sources apply to the current binary `PresenceClassifier`. They will not be used by the planned multi-class group classifier (see [multiclass-classifier-plan.md](multiclass-classifier-plan.md)).

| Source | Description |
|--------|-------------|
| `positive` | Keyword-confirmed (case, term) pairs — from the keyword scan (training pipeline only) |
| `hard_negative` | False-positive predictions from previous eval cycle |
| `fp_extra_negative` | Additional random labels sampled for FP cases |
| `co_negative` | Completely-off predictions from the rolling CO bank — the specific wrong-group (case, label) pairs that fool cosine similarity |
| `easy_negative` | Random wrong labels for keyword-confirmed cases |

### Evaluation verdicts (`training/binary/evaluate.py`)

| Verdict | Meaning |
|---------|---------|
| `good` | Predicted term exactly matches a keyword-matched term |
| `slightly_off` | No exact term match but predicted group matches a keyword group |
| `completely_off` | Neither term nor group matches any keyword label for this case |
| `false_positive` | Case has no keyword labels at all (should be Uncategorized) |
| `false_negative` | Confirmed cancer case with no good/slightly_off prediction |

---

## How to Run

> **Note:** Use `ml/.venv/bin/python3` — the project venv is at `ml/.venv/`. Plain `python` or `python3` will not find the required packages.

**Current recommended command** (bank exists at `ml/output/evaluation/evaluation_co_bank.csv`):

```bash
ml/.venv/bin/python3 ml/scripts/run_training.py \
  --label "..." \
  --co-neg-per-case 10 \
  --fp-neg-per-case 10 \
  --embedding-min-sim 0.05 \
  --epochs 25 \
  --recall-weight 0.25 \
  --device mps \
  --local-only
```

If starting fresh (no bank yet), use `--co-neg-per-case 5` for the first cycle until the bank exceeds ~20k pairs, then switch to `--co-neg-per-case 10`. With the current dataset (5,788 cancer cases), the bank fills to >20k after c1 alone — switch to co=10 from c2.

---

## Cold Start (after resetting embeddings or classifier)

A cold start is required any time the embedding space changes — e.g. after updating PetBERT, changing label enrichment logic, or switching `--enrich-labels-csv`. Old bank pairs are anchored to the old cosine space and will add noise; the cache is no longer valid either (see Fix 7).

### Prerequisites

1. `ml/output/diagnoses/keyword_predictions.csv` must exist. If not, run the keyword scan first:
   ```bash
   ml/.venv/bin/python3 -m keyword_pipeline
   ```

2. `ml/data/report.csv` must exist (the input data).

### Files to delete

```bash
rm -f ml/data/embedding_cache.npz                          # stale cache — rebuilt on first cycle
rm -f ml/output/evaluation/evaluation_co_bank.csv          # old-space bank — must start fresh
rm -f ml/model/checkpoints/presence_classifier_best.pt     # old checkpoint
```

### Warm-up phase

The first cycle's Step 0 detects the missing cache and runs PetBERT on all reports and labels. This is the only time PetBERT runs — all subsequent cycles load from cache. It takes several minutes.

```bash
ml/.venv/bin/python3 ml/scripts/run_training.py \
  --label "cold-start c1" \
  --co-neg-per-case 5 \
  --fp-neg-per-case 10 \
  --embedding-min-sim 0.05 \
  --epochs 25 \
  --recall-weight 0.25 \
  --device mps \
  --local-only
```

Check `ml/output/evaluation/evaluation_co_bank.csv` row count after c1. With the current dataset (5,788 cancer cases), the bank exceeds 20k after c1 alone (large case count × ~36% CO rate) — switch to co=10 immediately for c2. With smaller datasets, bank grows ~3–6k pairs/cycle and may take 5–6 cycles to reach 20k.

### Switch to `--co-neg-per-case 10` once bank exceeds ~20k pairs

Switch to the [standard command](#how-to-run) above. Expect a noticeable Good+Slight jump on the first co=10 cycle.

### Expected trajectory (5,788 cancer cases)

| Cycles completed | Expected Good+Slight | Notes |
|-----------------|---------------------|-------|
| c1 (co=5) | ~16% | Cache rebuilt; bank fills to >20k in this cycle |
| c2 (co=10) | ~14% | Slight dip before classifier adjusts |
| c3–c4 | ~26–31% | Rapid improvement |
| c5+ (plateau) | ~32–33% | Stable; CO floor ~33% |

---

## Key Parameters

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `--embedding-min-sim` | `0.05` | After Fix 1 (mean subtraction), scores are centered — use 0.05, not 0.5 |
| `--co-neg-per-case` | `10` (bank >20k) / `5` (bank <20k) | Raising to 10 once the bank exceeds ~20k pairs was the key unlock: Good+Slight jumped 13.6% → 21.0% |
| `--fp-neg-per-case` | `10` | Keep at 10; reducing to 5 weakens FP rejection |
| `--epochs` | `25` | Beyond 25 shows diminishing returns |
| `--pos-weight` | `1.0` | Do not increase; the sampler already balances training |
| `--recall-weight` | `0.25` | Score = `(1-rw)·P + rw·R`. At rw=0.5, epoch-1 degenerate checkpoints (R≈0.95, P≈0.10) could win and produced bad cycles. At rw=0.25 they score ~0.31 vs mid-training balanced checkpoints at ~0.39 — they can no longer win. Do not raise above 0.5. |
| `--max-pos-per-group` | `0` (no cap) | Do not cap — removes signal from already-good groups |

---

## Development History

All runs on 2026-03-03–04, device: mps.

### Phase 1 — Initial classifier (pre-normalization)

Basic MLP classifier without CO negatives. The pipeline predicted only 3 groups because two labels ("Pyogenic granuloma", "Lipoma NOS") had pathologically high mean cosine similarity across all cases, monopolising `argmax`. FP rate stayed near 54%.

| Timestamp | Label | Total | Good | Slightly off | Good+Slight | CO% | FP% | FN% |
|-----------|-------|-------|------|-------------|-------------|-----|-----|-----|
| 13:22 | Baseline (no classifier) | 13,855 | 0.1% | 3.2% | 3.3% | 42.6% | 54.1% | — |
| 13:34 | Classifier v1 | 13,855 | 1.0% | 1.6% | 2.6% | 43.3% | 54.1% | — |
| 13:55 | Classifier v2 | 13,652 | 1.7% | 3.5% | 5.2% | 41.4% | 53.4% | — |
| 14:17 | Classifier v3 | 10,367 | 3.3% | 1.6% | 4.9% | 53.2% | 41.9% | — |
| 14:36 | Classifier v4 | 9,530 | 2.3% | 5.2% | 7.5% | 55.1% | 37.4% | — |
| 14:58 | Classifier v5 | 9,504 | 3.5% | 2.0% | 5.5% | 57.7% | 36.8% | — |
| 15:17 | Classifier v6 | 9,525 | 2.3% | 4.4% | 6.7% | 55.7% | 37.5% | — |
| 15:59 | Classifier v7 | 11,395 | 2.2% | 1.8% | 4.0% | 50.1% | 38.3% | — |
| 16:32 | Classifier v8 | 12,405 | 1.9% | 3.6% | 5.5% | 44.7% | 43.0% | 6.9% |

### Fix 1 — Per-label mean subtraction (`categorization.py`)

**Problem:** Two labels dominated `argmax` regardless of report content:

| Label | Group | Mean similarity |
|-------|-------|----------------|
| Pyogenic granuloma | Blood vessel tumors | 0.912 |
| Lipoma, NOS | Lipomatous neoplasms | 0.846 |

**Fix:** Subtract each label's mean score (computed across all cases in the batch) before `argmax`:

```python
finite_mask = np.isfinite(sims)
label_means = (
    np.where(finite_mask, sims, 0.0).sum(axis=0)
    / np.maximum(finite_mask.sum(axis=0), 1)
)
sims = sims - label_means[np.newaxis, :]
```

**Result:** Predicted groups expanded from 3 → 39. FP dropped from 43% → 30.8% immediately.

**Note:** `label_scores` in `petbert_scan_similarity_scores.csv` now stores centered (mean-subtracted) scores, not raw cosine similarities. Set `--embedding-min-sim` to ~0.05 rather than 0.5.

### Phase 2 — Post-normalization training

After Fix 1, FP dropped substantially but completely-off remained high.

| Timestamp | Label | Total | Good | Slightly off | Good+Slight | CO% | FP% | FN% |
|-----------|-------|-------|------|-------------|-------------|-----|-----|-----|
| 17:07 | Post-normalization v1 | 9,724 | 1.5% | 4.0% | 5.5% | 54.3% | 30.8% | 9.4% |
| 17:22 | Post-norm v2 | 9,816 | 2.3% | 5.2% | 7.5% | 51.9% | 31.8% | 8.8% |
| 17:38 | Post-norm v3 | 9,545 | 1.8% | 6.6% | **8.4%** | 52.3% | 30.1% | 9.2% |
| 17:55 | Post-norm v4 | 9,708 | 2.1% | 4.4% | 6.5% | 53.3% | 31.1% | 9.1% |
| 18:14 | Group-balanced v1 (max-pos=80) | 9,298 | 1.7% | 4.9% | 6.6% | 54.8% | 29.0% | 9.6% |
| 21:27 | Group-balanced v2 | 9,868 | 2.2% | 4.4% | 6.6% | 52.5% | 32.4% | 8.6% |
| 21:31 | Group-balanced v3 | 10,013 | 1.4% | 8.2% | 9.6% | 52.1% | 30.3% | 8.0% |
| 21:37 | Dropped group cap | 10,309 | 1.9% | 7.3% | 9.2% | 51.3% | 31.8% | 7.8% |
| 21:41 | Recall-boosted v1 ❌ | 10,462 | 0.3% | 4.8% | 5.1% | 54.3% | 30.8% | 9.8% |

**Fix 2 (abandoned) — Group-balanced positives:** Capping keyword positives per group (`--max-pos-per-group 80`) did not help — it removed signal from correctly-learned groups without adding any for rarer ones. Do not use.

**Recall-boosted failure:** Raising `--recall-weight 0.7` and `--pos-weight 2.0` caused the classifier to approve essentially everything (val recall 0.93), flooding rare groups with wrong predictions. Do not raise `--recall-weight` above 0.5.

### Fix 3 — CO negatives (`build_training_pairs.py`)

**Problem:** Completely-off was stuck at ~51–54%. The classifier had no training examples of the form *"this case has cancer X, but label Y — which cosine similarity scored highly — is wrong."*

**Fix:** Added `co_negative` training source: for each `completely_off` prediction in `evaluation.csv`, add `(case, wrong_label) → target=0`, capped per case via `--co-neg-per-case`.

**Result:** Immediately improved to ~11–12% Good+Slight. However, exposed a feedback oscillation.

### Phase 3 — CO negatives with single-cycle feedback (oscillating)

| Timestamp | Label | Total | Good | Slightly off | Good+Slight | CO% | FP% | FN% |
|-----------|-------|-------|------|-------------|-------------|-----|-----|-----|
| 21:46 | CO-negatives v1 (co=3, fp=10) | 10,458 | 2.1% | 9.1% | 11.2% | 48.6% | 33.3% | 6.8% |
| 21:48 | CO-negatives v2 (co=5, fp=5) | 10,114 | 2.4% | 9.0% | 11.4% | 49.8% | 31.3% | 7.6% |
| 21:49 | **CO-negatives v3 (co=5, fp=10)** | 10,984 | 1.6% | 10.8% | **12.4%** | **45.0%** | 36.0% | 6.7% |
| 21:50 | CO-negatives v4 | 10,009 | 2.1% | 9.3% | 11.4% | 50.6% | 30.5% | 7.6% |
| 21:51 | CO-negatives v5 | 10,814 | 1.7% | 10.2% | 11.9% | 46.2% | 35.1% | 6.8% |
| 21:53 | CO-negatives v6 (dual-source) ❌ | 10,158 | 1.9% | 8.2% | 10.1% | 51.2% | 31.2% | 7.6% |
| 21:55 | Higher-threshold v1 (min-sim=0.10) | 10,805 | 1.5% | 10.7% | 12.2% | 45.8% | 35.2% | 6.8% |
| 21:55 | Higher-threshold v2 (min-sim=0.10) | 10,090 | 1.6% | 7.8% | 9.4% | 52.1% | 30.5% | 8.0% |
| 22:06 | CO-negatives cycle A | 11,046 | 1.6% | 10.1% | 11.7% | 45.4% | 36.4% | 6.6% |
| 22:07 | CO-negatives cycle B | 10,516 | 1.4% | 7.8% | 9.2% | 50.4% | 32.4% | 8.0% |

**Oscillation root cause:** Using only the most recent `evaluation.csv` for CO negatives creates a self-defeating loop:
- Good cycle (low CO%) → few CO negatives → weak training signal → next cycle regresses
- Bad cycle (high CO%) → many CO negatives → strong signal → next cycle improves

The dual-source attempt (`--co-neg-extra-csv`) failed — both files were from adjacent cycles and were nearly identical. Raising the threshold (min-sim=0.10) made no difference. The ceiling was ~12% Good+Slight.

### Fix 4 — Rolling CO-negative bank (`update_co_bank.py`)

**Problem:** Single-cycle CO feedback oscillates because a good cycle depletes its own training signal.

**Fix:** After each evaluate step (step 4.5), `update_co_bank.py` appends the current cycle's `completely_off` rows to a persistent bank file (`ml/output/evaluation/evaluation_co_bank.csv`), deduplicating on `(case_id, predicted_term)`. Step 1 of every cycle reads CO negatives from the bank instead of `evaluation.csv`, so the signal always includes all previous cycles.

New scripts/changes:
- `ml/scripts/utils/update_co_bank.py` — appends CO rows to bank, deduplicates
- `ml/scripts/utils/build_training_pairs.py` — `--co-neg-bank-csv` arg: when provided, uses bank as CO source instead of `--evaluation-csv`
- `ml/scripts/run_training_cycle.py` — step 4.5 updates bank; passes `--co-neg-bank-csv` to step 1

### Phase 4 — Rolling bank (co=5, bank growing)

| Timestamp | Label | Total | Good | Slightly off | Good+Slight | CO% | FP% | FN% | Bank size |
|-----------|-------|-------|------|-------------|-------------|-----|-----|-----|-----------|
| 22:15 | Rolling bank cycle 1 | 9,511 | 3.9% | 9.9% | 13.8% | 51.1% | 28.8% | 6.3% | 9,732 |
| 22:16 | Rolling bank cycle 2 | 10,243 | 2.5% | 10.4% | 12.9% | 47.8% | 32.6% | 6.7% | 14,096 |
| 22:18 | Rolling bank cycle 3 | 9,441 | 4.4% | 9.3% | 13.7% | 51.4% | 29.1% | 5.9% | 17,285 |
| 22:19 | Rolling bank cycle 4 | 9,960 | 3.0% | 12.5% | 15.5% | 46.8% | 31.2% | 6.4% | 19,591 |
| 22:20 | Rolling bank cycle 5 | 9,468 | 4.7% | 11.4% | 16.1% | 48.5% | 29.7% | 5.7% | 21,461 |
| 22:22 | Rolling bank cycle 6 | 9,886 | 3.4% | 10.2% | 13.6% | 48.9% | 31.0% | 6.5% | 23,186 |

Oscillation amplitude dropped from ~5pp (pre-bank) to ~1–2pp in Good+Slight. Good+Slight ceiling rose from ~12.4% to ~16%, and all metrics trended upward across cycles. However, a residual oscillation persisted because `--co-neg-per-case 5` caps each cycle at ~6,200 CO pairs regardless of how large the bank grows — leaving most of its accumulated diversity untapped.

### Fix 5 — Raise `--co-neg-per-case` to 10

Once the bank exceeded ~20k pairs, raising the cap from 5 → 10 approximately doubled the CO signal to ~12,200 pairs/cycle, drawn from 23k+ unique pairs.

### Phase 5 — Rolling bank (co=10, oscillation resolved)

| Timestamp | Label | Total | Good | Slightly off | Good+Slight | CO% | FP% | FN% | Bank size |
|-----------|-------|-------|------|-------------|-------------|-----|-----|-----|-----------|
| 22:23 | **Rolling bank cycle 7 (co=10)** | 9,609 | **7.0%** | **14.0%** | **21.0%** | **43.0%** | 32.1% | **3.9%** | 25,511 |
| 22:24 | **Rolling bank cycle 8 (co=10)** | 9,427 | **7.7%** | **15.2%** | **22.9%** | **42.5%** | **31.0%** | **3.7%** | 27,199 |

Cycle 7 → 8: all metrics improved again with no regression. The oscillation is resolved. With ~12k diverse CO pairs per cycle from a 27k-pair bank, the classifier sees a stable, sufficient training signal every cycle and no longer depends on the previous cycle being "bad" to get a strong update.

---

### Fix 6 — Label embedding enrichment (`labels/enrichment.py`) ⚠️ revised in Fix 9

**Motivation:** Label text is minimal (`"{term} {group}"`), which limits how well PetBERT can match report language to taxonomy labels. The keyword scan already provides confirmed (case, label) pairs. The idea was to blend each label's embedding with the mean embedding of its keyword-matched diagnosis strings to add clinical vocabulary.

**Original implementation (diagnosis-text-based — had minimal impact):**
- `ml/labels/enrichment.py` — for each label term in `keyword_predictions.csv`, embedded all keyword-matched diagnosis strings, took the mean, and averaged 50/50 with the original label embedding.
- Enriched embeddings stored in the cache under `enriched_label_embeddings`. Both `petbert_scan` and `train_classifier.py` use enriched embeddings when present.
- CLI: `--enrich-labels-csv <path_to_keyword_predictions.csv>` on `petbert_scan` and `run_training_cycle.py`.
- Cache invalidation: passing `--enrich-labels-csv` sets `require_enriched=True` in `load_cache`; cache is rebuilt automatically if enriched embeddings are missing.

**Why it had minimal impact:** The `diagnosis` column in `keyword_predictions.csv` contains short anatomic phrases (e.g. `"SKIN DORSUM: SQUAMOUS CELL CARCINOMA"`), which live in nearly the same region of PetBERT's embedding space as the label texts (`"Squamous cell carcinoma NOS Squamous cell neoplasms"`). Blending two vectors that are already close together barely moves the label embedding. Meanwhile the classifier matches against `mean_embeddings` — mean PetBERT embeddings of full clinical report columns (HISTOPATHOLOGICAL SUMMARY, FINAL COMMENT, ANCILLARY TESTS) — which are in a very different part of the embedding space. The enrichment never bridged that gap.

**Bug found and fixed (2026-03-03):** In the first enriched cycle, `score_matrix()` was still receiving the original `label_embeddings` instead of `active_label_embeddings`. The classifier trained on enriched embeddings but scored with original ones — garbage presence probabilities. Fixed by passing `active_label_embeddings` consistently to both `run_categorization` and `classifier.score_matrix`.

### Phase 6 — Label embedding enrichment (ongoing)

| Timestamp | Label | Total | Good | Slightly off | Good+Slight | CO% | FP% | FN% | Bank size |
|-----------|-------|-------|------|-------------|-------------|-----|-----|-----|-----------|
| 23:14 | enriched labels v1 ❌ (bug: wrong embs at inference) | 4,437 | 0.1% | 1.3% | 1.4% | 37.3% | 34.0% | 27.4% | 33,446 |
| 23:31 | enriched v2 (cycle 1, bug fixed) | 10,189 | 3.3% | 13.6% | 16.9% | 43.9% | 33.1% | 6.1% | 40,660 |
| 23:32 | enriched v3 (cycle 2) | 8,998 | 0.4% | 3.8% | 4.2% | 58.7% | 26.3% | 10.8% | 44,577 |
| 23:34 | enriched v4 (cycle 3) | 10,095 | 0.5% | 8.0% | 8.5% | 52.1% | 30.4% | 9.0% | — |
| 23:34 | **enriched v5 (cycle 4)** | 9,818 | **4.7%** | **14.0%** | **18.7%** | **43.8%** | 31.9% | **5.6%** | — |
| 23:34 | enriched v6 (cycle 5) | 8,946 | 0.2% | 5.0% | 5.2% | 58.0% | 25.3% | 11.4% | ~55k |

**Observation:** Severe oscillation (±13pp) despite 44k+ bank. ~75% of bank pairs from old (unenriched) embedding space. Old pairs are valid negatives but shift the decision boundary each cycle because the "hardest" negatives in the old cosine space are different from the enriched space. Good cycles improving each time (16.9% → 18.7%), but oscillation is not resolving.

**Fix 7 — Reset CO bank.** Same logic as Fix 1 (normalization changed the score distribution): when the embedding space changes, old bank pairs introduce noise. Reset bank and rebuild from enriched-space predictions only.

### Phase 7 — Bank reset, co=5 (bank rebuilding)

| Timestamp | Label | Total | Good | Slightly off | Good+Slight | CO% | FP% | FN% | Bank size |
|-----------|-------|-------|------|-------------|-------------|-----|-----|-----|-----------|
| 23:36 | bank-reset c1 | 10,400 | 1.3% | 8.4% | 9.7% | 50.4% | 32.3% | 7.6% | — |
| 23:36 | bank-reset c2 | 10,297 | 1.8% | 6.2% | 8.0% | 52.3% | 31.6% | 8.0% | — |
| 23:36 | bank-reset c3 | 9,754 | 3.3% | 8.5% | 11.8% | 51.4% | 29.8% | 7.0% | — |
| 23:37 | bank-reset c4 | 10,007 | 2.5% | 8.4% | 10.9% | 50.8% | 31.0% | 7.4% | — |
| 23:37 | bank-reset c5 | 8,770 | 0.4% | 4.9% | 5.3% | 58.0% | 25.6% | 11.2% | — |
| 23:37 | **bank-reset c6** | 9,837 | **3.8%** | **8.9%** | **12.7%** | **49.6%** | 31.8% | **5.9%** | 24,379 |

Oscillation persists but amplitude is narrowing (±7pp vs ±15pp before reset). c6 peak of 12.7% mirrors Phase 4 cycle 6 before the co=10 switch (which jumped to 21%). Bank now exceeds 20k → switching to co=10.

### Phase 8 — Enriched, co=10 (bank grown from reset)

| Timestamp | Label | Total | Good | Slightly off | Good+Slight | CO% | FP% | FN% | Bank size |
|-----------|-------|-------|------|-------------|-------------|-----|-----|-----|-----------|
| 23:37 | co10 c1 | 9,163 | 1.0% | 5.1% | 6.1% | 57.4% | 26.3% | 10.2% | 28,081 |
| 23:38 | co10 c2 | 10,243 | 6.0% | 11.8% | 17.8% | 42.8% | 34.9% | 4.5% | 30,679 |
| 23:38 | co10 c3 | 9,590 | 1.0% | 4.9% | 5.9% | 57.0% | 27.8% | 9.3% | 33,422 |
| 23:38 | co10 c4 | 9,918 | 6.8% | 12.1% | 18.9% | 43.1% | 34.0% | 4.1% | 34,649 |
| 23:38 | **co10 c5** | 9,381 | **7.4%** | **14.7%** | **22.1%** | **43.3%** | **30.5%** | **4.1%** | 35,804 |
| 23:39 | co10 c6 | 9,510 | 0.9% | 5.3% | 6.2% | 56.3% | 28.2% | 9.3% | 38,052 |
| 23:39 | co10 c7 | 10,120 | 6.7% | 11.8% | 18.5% | 42.6% | 34.7% | 4.2% | 38,665 |
| 23:39 | co10 c8 | 9,127 | 1.1% | 6.8% | 7.9% | 55.5% | 26.6% | 10.1% | 40,714 |
| 23:46 | co10 c9 | 10,303 | 6.3% | 11.8% | 18.1% | 42.2% | 35.6% | 4.2% | 41,101 |
| 23:48 | **co10 c10** | 9,149 | **7.7%** | **12.6%** | **20.3%** | 46.1% | 29.2% | 4.4% | 41,923 |
| 23:49 | co10 c11 | 9,592 | 1.1% | 9.8% | 10.9% | 51.6% | 28.6% | 8.9% | 43,233 |
| 23:51 | co10 c12 | 9,840 | 7.1% | 12.0% | 19.1% | 43.4% | 33.4% | 4.2% | 43,636 |
| 23:52 | co10 c13 | 8,915 | 0.5% | 4.5% | 5.0% | 57.7% | 26.5% | 10.7% | 45,738 |

**Oscillation pattern:** Good cycles consistently land 18–22%; bad cycles drop to ~5–8%. Bad cycles are caused by the checkpoint selection metric being gamed by epoch-1 degenerate models (see Fix 8). Bad cycles trended upward over time (c6=6.2% → c8=7.9% → c11=10.9%) as the bank filled, but a bad c13 (5.0%) confirmed the root cause was the metric, not the bank size.

### Fix 8 — Reduce `--recall-weight` to 0.25

**Problem:** The checkpoint score formula is `(1 - rw) * precision + rw * recall`. At rw=0.5, an epoch-1 degenerate checkpoint with R=0.957 and P=0.095 scores `0.5×0.095 + 0.5×0.957 = 0.526`, beating all later balanced epochs. When this checkpoint is used for inference the classifier accepts nearly every (case, label) pair → CO and FN spike → bad cycle.

**Fix:** Set `--recall-weight 0.25`. At rw=0.25, that same epoch-1 checkpoint scores `0.75×0.095 + 0.25×0.957 = 0.310`, while typical mid-training epochs (P≈0.20, R≈0.82) score `0.75×0.20 + 0.25×0.82 = 0.355` — balanced late checkpoints always win.

**Result:** Zero bad cycles across 7 consecutive cycles. Oscillation eliminated.

### Phase 9 — rw=0.25 (oscillation resolved)

| Timestamp | Label | Total | Good | Slightly off | Good+Slight | CO% | FP% | FN% | Bank size |
|-----------|-------|-------|------|-------------|-------------|-----|-----|-----|-----------|
| 23:55 | **rw025 c14** | 9,854 | **6.2%** | **11.3%** | **17.5%** | 44.6% | 32.9% | 4.9% | 46,081 |
| 23:56 | **rw025 c15** | 9,197 | **7.6%** | **11.8%** | **19.4%** | 46.7% | 29.4% | 4.5% | 46,480 |
| 23:57 | **rw025 c16** | 9,847 | **6.6%** | **12.6%** | **19.2%** | 43.5% | 32.7% | 4.6% | 46,843 |
| 23:58 | **rw025 c17** | 9,433 | **7.3%** | **13.1%** | **20.4%** | 44.7% | 30.6% | 4.3% | 47,121 |
| 00:00 | **rw025 c18** | 9,761 | **7.1%** | **13.2%** | **20.3%** | 42.8% | 32.6% | 4.3% | 47,492 |
| 00:01 | **rw025 c19** | 9,501 | **6.6%** | **12.2%** | **18.8%** | 45.7% | 30.8% | 4.6% | — |
| 00:02 | **rw025 c20** | 9,872 | **7.2%** | **12.7%** | **19.9%** | **42.7%** | 33.2% | 4.2% | 47,492 |

**Phase 9: 7 consecutive good cycles — system converged.** All rw=0.25 cycles produced 17.5–20.4% Good+Slight with no bad cycle (vs. max 2 consecutive with rw=0.5). CO is at 42.7%, hitting the documented floor. FN stable at ~4%. Further improvement in Good+Slight or CO is unlikely under the current architecture without group-level re-ranking or better label embeddings (see Known Limitations).

---

### Fix 9 — Cache-based label enrichment (revised `labels/enrichment.py`)

**Problem:** Fix 6's diagnosis-text enrichment barely moved label embeddings because diagnosis strings and label strings already occupy the same compact region of PetBERT's space. The domain gap to full clinical report embeddings remained.

**Fix:** Replace diagnosis-text embedding with cached report embeddings. For each keyword-confirmed `(case_id, label_term)` pair in `keyword_predictions.csv`, look up that case's `mean_embedding` from the embedding cache, average them per label term, and blend 50/50 with the original label embedding.

```python
# old: re-embed short diagnosis strings through PetBERT
mean_diag = mean(embed(diagnosis_texts_for_label))
enriched[label] = (label_emb + mean_diag) / 2

# new: look up full-report embeddings already in the cache — no PetBERT call needed
mean_report = mean(cache["mean_embeddings"][confirmed_case_indices])
enriched[label] = (label_emb + mean_report) / 2
```

**Why this should help:** The classifier input is `(mean_report_embedding, label_embedding)`. Pulling the label embedding toward the centroid of confirmed-case report embeddings directly reduces the distance the classifier must bridge. No new PetBERT inference is needed — the embeddings are already in the cache.

**Changes:**
- `ml/labels/enrichment.py` — complete rewrite: removed `tokenizer`/`model`/`device` params, added `case_ids` and `mean_report_embeddings` params; reads cache row indices instead of embedding text
- `ml/petbert_scan/pipeline.py` — updated call site to pass `ids` and `embeddings` (already computed at that point)

**Requires cold start:** The embedding cache and CO bank must be reset before the first cycle. The cache's `enriched_label_embeddings` will now be computed from report embeddings instead of diagnosis text, so the old cache is stale. See the [Cold Start](#cold-start-after-resetting-embeddings-or-classifier) section.

### Phase 10 — Cache-based enrichment (in progress)

All runs on 2026-03-04, device: mps. Cold start performed (cache, bank, checkpoint deleted before c1).

**Warm-up phase (co=5, bank building):**

| Timestamp | Label | Total | Good | Slightly off | Good+Slight | CO% | FP% | FN% | Bank size |
|-----------|-------|-------|------|-------------|-------------|-----|-----|-----|-----------|
| 00:49:24 | cold-start c1 | 10,667 | 1.4% | 5.9% | 7.3% | 51.5% | 33.2% | 8.0% | ~5.5k |
| 00:51:49 | cold-start c2 | 10,492 | 1.9% | 5.3% | 7.2% | 51.8% | 33.0% | 8.0% | ~10.9k |
| 00:54:15 | cold-start c3 | 9,592 | 3.5% | 5.7% | 9.2% | 54.5% | 29.3% | 7.0% | — |
| 00:54:48 | cold-start c4 | 9,868 | 2.4% | 6.0% | 8.4% | 53.3% | 30.6% | 7.7% | — |
| 00:55:19 | cold-start c5 | 9,918 | 2.7% | 5.3% | 8.0% | 53.9% | 31.0% | 7.1% | ~20.8k → switched to co=10 |

**co=10 phase (bank >20k):**

| Timestamp | Label | Total | Good | Slightly off | Good+Slight | CO% | FP% | FN% | Bank size |
|-----------|-------|-------|------|-------------|-------------|-----|-----|-----|-----------|
| 00:56:09 | **cold-start c6 (co=10)** | 10,240 | **4.6%** | **6.8%** | **11.4%** | 49.0% | 33.9% | 5.7% | — |
| 00:56:52 | **cold-start c7 (co=10)** | 9,977 | **5.0%** | **8.1%** | **13.1%** | 48.7% | 32.8% | 5.4% | — |
| 00:57:25 | cold-start c8 (co=10) | 9,877 | 4.4% | 6.0% | 10.4% | 51.7% | 31.5% | 6.4% | ~27.3k |
| 01:00:29 | **Phase 10 c9 (co=10)** | 10,008 | **5.7%** | **8.4%** | **14.1%** | 47.6% | 33.3% | 5.0% | ~28.2k |
| 01:03:00 | Phase 10 c10 (co=10) | 10,026 | 4.7% | 7.0% | 11.7% | 49.9% | 32.5% | 5.9% | ~28.9k |
| 01:03:58 | Phase 10 c11 (co=10) | 9,214 | 4.5% | 3.5% | 8.0% | 55.6% | 28.7% | 7.6% | ~30.4k |
| 01:05:17 | Phase 10 c12 (co=10) | 10,473 | 4.3% | 7.6% | 11.9% | 47.3% | 35.1% | 5.7% | ~31.2k |
| 01:06:20 | Phase 10 c13 (co=10) | 9,722 | 3.3% | 3.1% | 6.4% | 54.8% | 31.0% | 7.7% | ~32.3k |
| 01:07:59 | **Phase 10 c14 (co=10)** | 10,343 | **5.2%** | **7.9%** | **13.1%** | 47.2% | 34.5% | 5.2% | ~32.6k |
| 01:09:05 | Phase 10 c15 (co=10) | 9,926 | 4.8% | 6.9% | 11.7% | 49.8% | 32.8% | 5.6% | ~33.2k |
| 01:10:16 | Phase 10 c16 (co=10) | 10,002 | 5.3% | 6.4% | 11.7% | 49.8% | 32.7% | 5.8% | ~33.4k |
| 01:11:37 | Phase 10 c17 (co=10) | 10,109 | 4.7% | 8.2% | 12.9% | 48.1% | 33.6% | 5.5% | ~34.0k |
| 01:12:52 | Phase 10 c18 (co=10) | 9,759 | 5.2% | 6.5% | 11.7% | 51.4% | 31.0% | 5.9% | ~34.3k |

**Phase 10 plateau confirmed (13 co=10 cycles, 18 total):** Last 5 cycles (c14–c18): 13.1%, 11.7%, 11.7%, 12.9%, 11.7% — locked in 11–13% with no bad collapses and no improvement. Bank growth slowing to ~260–570 rows/cycle. **Conclusion:** Phase 10 with Fix 9 enrichment has converged at ~12% Good+Slight, significantly below Phase 9's 17.5–20.4%. The 50/50 label-report blend in Fix 9 creates hybrid embeddings that shift the cosine score landscape sufficiently to lower the effective ceiling. Further cycles under current settings are unlikely to break through 14.1%.

---

### Phase 11 — New keyword data (5,788 confirmed cases, 44 groups)

All runs on 2026-03-04, device: mps. New `keyword_predictions.csv` and `report.csv` delivered with 5,788 keyword-confirmed cancer cases (up from 1,273) across 44 groups (up from 39) and 12,620 total reports. Full cold start performed (cache, bank, checkpoint deleted).

**Context:** Cache rebuild covers all 12,620 reports. CO bank hit ~20k rows after c1 alone (54k predictions × 36.7% CO rate), allowing co=10 from c2 onward. No warm-up phase needed.

| Timestamp | Label | Total | Good | Slightly off | Good+Slight | CO% | FP% | FN% |
|-----------|-------|-------|------|-------------|-------------|-----|-----|-----|
| 03:26:12 | cold-start c1 (co=5) | 54,058 | 3.3% | 13.0% | 16.3% | 36.7% | 42.7% | 4.3% |
| 10:57:22 | new-data c2 (co=10) | 48,655 | 3.6% | 10.0% | 13.6% | 43.1% | 37.6% | 5.7% |
| 10:58:54 | **new-data c3** | 45,445 | **8.7%** | **17.4%** | **26.1%** | **36.1%** | 35.6% | **2.2%** |
| 11:00:17 | **new-data c4** | 42,151 | **11.5%** | **19.5%** | **31.0%** | **35.4%** | 32.1% | **1.5%** |
| 11:01:37 | **new-data c5** | 43,299 | **11.5%** | **19.7%** | **31.2%** | **33.2%** | 34.2% | **1.4%** |
| 11:02:55 | **new-data c6** | 42,026 | **11.9%** | **21.2%** | **33.1%** | **33.2%** | 32.4% | **1.3%** |
| 11:04:48 | **new-data c7** | 43,891 | **11.2%** | **20.9%** | **32.1%** | **31.8%** | 34.8% | **1.3%** |
| 11:06:10 | **new-data c8** | 41,908 | **12.0%** | **21.1%** | **33.1%** | 33.4% | 32.0% | **1.4%** |
| 11:07:31 | **new-data c9** | 43,677 | **11.3%** | **20.8%** | **32.1%** | **32.0%** | 34.6% | **1.4%** |
| 11:08:47 | **new-data c10** | 42,399 | **11.5%** | **19.3%** | **30.8%** | 34.9% | 32.4% | 1.9% |

**Phase 11: stable plateau at ~32% Good+Slight, ~33% CO.** Both metrics dramatically exceed Phase 9 (20.4% Good+Slight, 42.7% CO). The old 42% CO floor is broken — CO now fluctuates 32–35%. FN is very low (~1.3–1.9%). No degenerate cycles. The improvement is driven entirely by the increased keyword coverage (1,273 → 5,788 confirmed cancer cases), which provides richer CO-negative training signal and better-calibrated positive examples.

**GroupClassifier comparison (same new data):**
- GC @ threshold 0.3: 13.9% Good+Slight, 57.5% CO — worse than binary
- GC @ threshold 0.8: 21.9% Good+Slight, 54.5% CO, 15.6% FN — worse than binary c6+

Binary PresenceClassifier is the clear winner at current data volumes. GroupClassifier still overfits (val loss >> train loss) despite 5,788 cases across 44 groups.

---

### Windows Unicode fix (2026-03-05)

When running on Windows (cp1252 console), `print()` calls containing `→` (U+2192) and `★` (U+2605) raise `UnicodeEncodeError` because those characters are not in the cp1252 codepage. Fixed by replacing them with ASCII equivalents (`->`, `*`, `--`) in:
- `ml/training/binary/update_co_bank.py` — `→` in bank update message
- `ml/training/binary/run_cycle.py` — `★` in best-checkpoint message
- `ml/training/group/train.py` — `★` in best-F1 message
- `ml/petbert_pipeline/embedding_cache.py` — `→` in all cache-miss messages

### Phase 12 — XPU, cold start, same dataset (2026-03-05)

All runs on 2026-03-05, device: xpu (Intel). Cold start performed (cache, bank, checkpoint deleted). Same `report.csv` and `keyword_predictions.csv` as Phase 11 (12,620 reports, 5,788 confirmed cancer cases, 44 groups).

**Note on c1 bank anomaly:** c1's step 4.5 shows `39,104 -> 50,823` unique pairs, implying the bank already held 39,104 rows at the start of c1's update step. This is because two prior aborted runs (during Windows Unicode debugging) each wrote CO rows to the bank before failing on the print statement — the bank file was written *before* the failing `print()` call. This pre-seeded the bank with CO pairs from those evaluation runs, which used the same embedding cache and therefore the same cosine space. The pairs are valid; this explains why c1's training pairs included CO negatives despite the cold start, and why c1 started unusually high.

| Timestamp | Label | Total | Good | Slightly off | Good+Slight | CO% | FP% | FN% | Bank size |
|-----------|-------|-------|------|-------------|-------------|-----|-----|-----|-----------|
| 00:05:45 | **cold-start c1 (co=5)** | 42,948 | **9.0%** | **19.4%** | **28.4%** | 37.1% | 32.0% | 2.5% | 50,823 |
| 00:09:01 | **c2 (co=10)** | 43,061 | **11.1%** | **18.0%** | **29.1%** | 36.0% | 33.3% | 1.6% | 61,391 |
| 00:10:15 | **c3 (co=10)** | 43,462 | **11.5%** | **20.1%** | **31.6%** | 32.7% | 34.3% | 1.4% | 66,557 |
| 00:11:25 | **c4 (co=10)** | 42,897 | **11.6%** | **20.4%** | **32.0%** | 33.1% | 33.6% | 1.3% | 70,087 |
| 00:12:38 | c5 (co=10) | 42,786 | 11.2% | 19.0% | 30.2% | 34.7% | 33.3% | 1.7% | 73,596 |
| 00:13:50 | c6 (co=10) | 42,971 | 11.7% | 19.8% | 31.5% | 33.3% | 33.7% | 1.5% | 76,249 |
| 00:15:15 | c7 (co=10) | 43,805 | 11.3% | 20.6% | 31.9% | 32.1% | 34.6% | 1.4% | 78,410 |
| 00:16:26 | c8 (co=10) | 43,385 | 11.0% | 19.3% | 30.3% | 34.3% | 33.6% | 1.8% | ~80,189 |

**Phase 12 plateau confirmed (8 cycles).** Best Good+Slight: 32.0% (c4). Oscillation amplitude ±2pp around 31–32%, with no degenerate cycles. CO floor: 32–35%. FN: 1.3–2.5%. These results are consistent with Phase 11 (33.1% best, 31.8% CO) on the same dataset; Phase 12 is slightly below Phase 11's best, likely because the pre-seeded bank from aborted debug runs biased c1 training positively but did not improve the plateau.

**Bank saturation:** New rows added per cycle declining rapidly (c2: +10,568 → c4: +3,530 → c7: +2,161 → c8: ~+1,779). Bank approaching saturation at ~80k unique CO pairs. Further cycles are unlikely to improve results.

**Parameters confirmed stable:**

| Parameter | Value | Notes |
|-----------|-------|-------|
| `--co-neg-per-case` | 10 | Optimal; bank >20k from c1 in this run |
| `--fp-neg-per-case` | 10 | Unchanged |
| `--recall-weight` | 0.25 | No degenerate cycles in 8 consecutive runs |
| `--epochs` | 25 | Unchanged |
| `--embedding-min-sim` | 0.05 | Unchanged |

---

## Summary of Progress

| Phase | Best Good+Slight | Best CO% | Best FN% |
|-------|-----------------|----------|----------|
| No classifier | 3.3% | 42.6% | — |
| Phase 1 (basic classifier) | 7.5% | 44.7% | 6.9% |
| Phase 2 (post-normalization) | 9.2% | 51.3% | 7.8% |
| Phase 3 (CO negatives, single-cycle) | 12.4% | 45.0% | 6.7% |
| Phase 4 (rolling bank, co=5) | 16.1% | 46.8% | 5.7% |
| Phase 5 (rolling bank, co=10) | 22.9% | 42.5% | 3.7% |
| Phase 6–8 (diagnosis-text enrichment, oscillating) | 22.1% | 42.2% | 4.1% |
| Phase 9 (rw=0.25, oscillation resolved) | 20.4% | 42.7% | 4.2% (stable, 7 consecutive cycles) |
| Phase 10 (cache-based enrichment — Fix 9) | 14.1% | 47.6% | 5.0% (regression — do not use) |
| **Phase 11 (new keyword data, 5,788 cases)** | **33.1%** | **31.8%** | **1.3%** (stable, 8+ consecutive cycles) |
| **Phase 12 (XPU, cold start, same dataset)** | **32.0%** | **32.1%** | **1.3%** (stable, 8 cycles — confirms Phase 11 plateau) |

---

## Known Limitations of the Binary PresenceClassifier

- **Completely-off floor (~33%)**: with 5,788 training cases the CO floor dropped from ~42% to ~33%. Further reduction requires either more keyword-confirmed cases or a group-level architecture (GroupClassifier).
- **Classifier trained on report-level embeddings**: the mean embedding across 3 text columns may lose fine-grained term-level signal present in individual sections.
- **GroupClassifier still overfits at 5,788 cases**: needs ~10,000+ confirmed cases across 44 groups to generalize reliably. Re-train whenever keyword coverage improves.

The planned multi-class group classifier (see [multiclass-classifier-plan.md](multiclass-classifier-plan.md)) directly addresses the CO floor by replacing pair-wise binary scoring with a single global group decision per report.
