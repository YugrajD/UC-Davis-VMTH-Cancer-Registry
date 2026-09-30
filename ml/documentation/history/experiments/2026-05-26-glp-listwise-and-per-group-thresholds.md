# Global label-presence classifier (GLP): listwise loss, levers and per-group thresholds

**Date:** 2026-05-26 · **Verdict:** inconclusive (near-Pareto at best, never migrated); per-group thresholds reverted

## Hypothesis
One model scoring every (case, label) pair (a GLP) could replace or augment the 25-way group classifier. Tested with eight levers (A-H, plus a diagnostic) in an experimental clone of `ml/`, since deleted, then with per-group thresholds on the GLP.

## Setup
- Baseline (clone): group classifier + per-group LPs + per-LP thresholds + tail gate: G+S 60.1% (Good 45.7, Slight 14.4), held-out test split.
- Levers: hard-negative retrain, BCE ensemble, all-groups LP, 3-input architecture, term-only label text, BCE blend, softmax at inference, listwise softmax cross-entropy loss.
- Threshold follow-ups on BCE-cycle3 and listwise GLPs (eval half, about 4,070 rows): per-group F1 thresholds (variant A, alpha) and coordinate descent on joint G+S (variant beta).

## Result
| Config | Good | Slight | G+S | Off | FP | FN | n |
|---|---|---|---|---|---|---|---|
| Baseline | 45.7% | 14.4% | 60.1% | 15.0% | 2.3% | 22.6% | 8,818 |
| `listwise_ensemble_t001` | 46.7% | 13.6% | 60.3% | 14.2% | 2.2% | 23.3% | 8,585 |
| `listwise_ensemble_t050` | 49.5% | 8.4% | 57.9% | n/a | n/a | 25.4% | n/a |

Levers: A loss (hard-negative retrain), B mixed win (+4.3 G+S, -8 Good), C no-op, D loss (-21 pp), E tiny mixed, F mixed (+1.2 Good, flat G+S), G Good-only, H biggest win (listwise loss).

Per-group thresholds (eval half):

| Config | Good | Slight | G+S | Fallback rate |
|---|---|---|---|---|
| BCE-cycle3 baseline | 46.9% | 13.9% | 60.8% | n/a |
| + per-group F1 thresholds | 48.7% | 11.9% | 60.6% | 43.8% |
| + coordinate descent on G+S | 48.7% | 11.9% | 60.6% | 2.88% |
| Listwise baseline | 46.0% | 14.1% | 60.1% | n/a |
| Listwise + per-group F1 | 48.3% | 11.7% | 60.0% | 64.3% |

## Why
- BCE per-pair training saturates scores (mean of per-case max sigmoid 0.997, 5-13 labels at or above 0.5 spanning several groups); this is structural, not a tuning issue. Listwise loss concentrated mass on the top label (top-1 group mass 0.199 to 0.715).
- Pair-level validation F1 (about 0.88 for both losses) does not predict pipeline G+S, which differed by about 10 pp.
- Per-group thresholds were redundant with raw argmax: 91.7% of newly routed cases picked the same group the fallback already chose, and calibration gains overfit by about 1.5 pp. The Good +2-3 / Slight -2-3 swing is a bucket swap on the same predictions, not new coverage. Show Good and Slight separately: the near-Pareto win depends on weighting Good over Slight and FN.

## Don't retry unless…
Never train GLP-style heads with per-pair BCE when downstream selection is rank-based; use softmax cross-entropy. Per-group thresholding on a GLP is closed as a lever.

## Where it went
Nothing migrated to `ml/`; the clone tree was deleted on 2026-09-14. The remaining headroom toward the promoted `ml/` (62.1%) was judged to sit in stage 3 (per-group LP heads), not stage 2 or thresholding. See the stacking result in [slight bank + CO-bank](2026-05-27-slight-bank-and-cobank-stacking.md).
