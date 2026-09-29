# Lowering the Uncommon-merge threshold (27 and 30 groups, Phases 28b-28d)

**Date:** 2026-05-07 · **Verdict:** reverted

## Hypothesis
Groups with fewer than 200 cases are merged into "Uncommon". Promoting a few of them (Thymic, Myxomatous, Cystic, Germ cell, Ductal; 83-161 unique train cases each) to their own group should raise accuracy on those groups.

## Setup
- Baseline: Phase 27 GroupClassifier (25 groups, macro F1 0.4475) plus Phase 28 LPs, 57.9% G+S (old evaluator, test set).
- 28b: threshold 100, 30 groups. 28c: 27 groups (Thymic and Myxomatous only). 28d: retrain the affected LPs on the 27-group structure.

## Result
| Run | Macro F1 | G+S | Notes |
|---|---|---|---|
| Phase 27 baseline | 0.4475 | 55.3% (three-stage) / 57.9% (four-stage) | |
| 28b, 30 groups | 0.4050 | 55.1% (three-stage) | Ductal, Cystic, Germ cell did not learn (F1 below 0.31) |
| 28c, 27 groups, four-stage | 0.4330 | 56.3% | Uncommon LP mislabelled aggressively (Ductal: 330 predictions at 2% good) |
| 28d, LP realigned | 0.4330 | 56.2% | Thymic has one label, so no LP can be trained |

Good and Slight were not split out.

## Why
The 27-group classifier was weaker than the 25-group one, and LP realignment could not overcome that. Thymic and Myxomatous themselves worked well (87% and 68% good in 28b). A counting caveat: the threshold counted annotation rows, not unique cases, so multi-label cases inflated group counts (Ductal had 83 unique train cases but passed 100 rows).

## Don't retry unless…
The 27-group GroupClassifier reaches at least the macro F1 of the 25-group one (0.4475 at the time). The current rule (`uncommon_threshold` 200, "Neoplasms, NOS" always merged) is the recipe value.

## Where it went
The `--label-presence-groups` filter was added for this experiment and later removed with the old tree. Checkpoint archive was deleted in the 2026-05-25 and 2026-09-14 cleanups.
