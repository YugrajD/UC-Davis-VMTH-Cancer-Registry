# Classifier Development Log

## Architecture Overview

The pipeline has three stages:

1. **Keyword scan** (`keyword_scan/`) — regex/keyword matching on `diagnoses.csv` to produce ground-truth labels. Matches ~18.6% of diagnosis rows (1,537 / 9,172). Produces `keyword_predictions.csv`.

2. **PetBERT scan** (`petbert_scan/`) — embeds report text columns (HISTOPATHOLOGICAL SUMMARY, FINAL COMMENT, ANCILLARY TESTS) and taxonomy labels with [SAVSNET/PetBERT](https://huggingface.co/SAVSNET/PetBERT), then assigns labels by cosine similarity. Uses a trained `PresenceClassifier` to replace raw cosine scores with presence probabilities.

3. **Presence classifier** (`model/presence_classifier.py`) — small MLP trained on (report embedding, label embedding) → binary presence probability. Trained via `run_training_cycle.py` which loops: build training pairs → train → re-run petbert scan → evaluate → update CO bank → log.

### Training data sources (`build_training_pairs.py`)

| Source | Description |
|--------|-------------|
| `positive` | Keyword-confirmed (case, term) pairs |
| `hard_negative` | False-positive predictions from previous eval cycle |
| `fp_extra_negative` | Additional random labels sampled for FP cases |
| `co_negative` | Completely-off predictions from the rolling CO bank — the specific wrong-group (case, label) pairs that fool cosine similarity |
| `easy_negative` | Random wrong labels for keyword-confirmed cases |

### Evaluation verdicts (`evaluate_predictions.py`)

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

**Current recommended command** (bank already exists at `ml/output/evaluation/evaluation_co_bank.csv`, 27k+ pairs):

```bash
ml/.venv/bin/python3 ml/scripts/run_training_cycle.py \
  --label "..." \
  --co-neg-per-case 10 \
  --fp-neg-per-case 10 \
  --embedding-min-sim 0.05 \
  --epochs 25 \
  --device mps \
  --local-only
```

If starting fresh (no bank yet), use `--co-neg-per-case 5` for the first ~6 cycles until the bank exceeds ~20k pairs, then switch to `--co-neg-per-case 10`.

---

## Key Parameters

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `--embedding-min-sim` | `0.05` | After Fix 1 (mean subtraction), scores are centered — use 0.05, not 0.5 |
| `--co-neg-per-case` | `10` (bank >20k) / `5` (bank <20k) | Raising to 10 once the bank exceeds ~20k pairs was the key unlock: Good+Slight jumped 13.6% → 21.0% |
| `--fp-neg-per-case` | `10` | Keep at 10; reducing to 5 weakens FP rejection |
| `--epochs` | `25` | Beyond 25 shows diminishing returns |
| `--pos-weight` | `1.0` | Do not increase; the sampler already balances training |
| `--recall-weight` | `0.5` | Do not raise above 0.5 — causes the classifier to approve everything |
| `--max-pos-per-group` | `0` (no cap) | Do not cap — removes signal from already-good groups |

---

## Development History

All runs on 2026-03-03, device: mps.

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

**Summary of progress:**

| Phase | Best Good+Slight | Best CO% | Best FN% |
|-------|-----------------|----------|----------|
| No classifier | 3.3% | 42.6% | — |
| Phase 1 (basic classifier) | 7.5% | 44.7% | 6.9% |
| Phase 2 (post-normalization) | 9.2% | 51.3% | 7.8% |
| Phase 3 (CO negatives, single-cycle) | 12.4% | 45.0% | 6.7% |
| Phase 4 (rolling bank, co=5) | 16.1% | 46.8% | 5.7% |
| **Phase 5 (rolling bank, co=10)** | **22.9%** | **42.5%** | **3.7%** |

---

## Known Limitations

- **Keyword ground truth is sparse**: only 18.6% of diagnosis rows are keyword-matched. Many true cancer cases are invisible to the evaluator, inflating false negative counts.
- **Classifier trained on report-level embeddings**: the mean embedding across 3 text columns may lose fine-grained term-level signal present in individual sections.
- **Label text is minimal**: taxonomy labels are embedded as `"{term} {group}"` — richer descriptions (synonyms, ICD-O codes, clinical context) could improve embedding discrimination and reduce the completely-off rate upstream.
- **Completely-off floor (~42%)**: the presence classifier can only accept/reject individual (case, label) pairs — it cannot redirect a wrong-group cosine match to the correct group. Breaking below ~40% likely requires group-level re-ranking or better label embeddings.
