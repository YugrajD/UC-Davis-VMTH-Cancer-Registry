# LabelPresence hard-negative mining and regularisation (QW1, QW2, Phase 30/30b)

**Date:** 2026-05-10 · **Verdict:** reverted (the case-disjoint validation split from QW2 was kept)

## Hypothesis
- QW1: random in-group negatives make the LP task too easy. Mining negatives by cosine similarity to the positive label (fraction 0.7, so 4 hard and 1 random per positive) should force discriminative features. Expected +2.5 to +4.0 pp G+S.
- QW2: a case-disjoint validation split, dropout 0.2, weight decay 1e-2, early stopping (patience 5) and recall weight 0.35 should close the gap between validation and production precision.

## Setup
- Baseline: Phase 28 heads (17-group set after the phase-29 cold start), LP threshold 0.5, group threshold 0.85, test set n=16,902 prediction rows, old evaluator. Later archived as `2026-05-10_pre-QW1-hardneg`.
- QW1 alone, then the QW1 + QW2 bundle; both evaluated at LP thresholds 0.5-0.8.

## Result
| Config | rows | Good | Slight | G+S | CO | FP | FN |
|---|---|---|---|---|---|---|---|
| Baseline, LP 0.5 | 16,902 | 25.7% | 33.8% | 59.5% | 23.4% | 8.0% | 9.2% |
| QW1, LP 0.5 | 20,231 | 19.3% | 39.1% | 58.4% | 24.0% | 9.5% | 8.1% |
| QW1 + QW2, LP 0.5 | 20,017 | 19.2% | 39.4% | 58.6% | 23.6% | 9.7% | 8.2% |
| QW1 + QW2, LP 0.8 | 13,611 | 27.4% | 28.9% | 56.3% | 22.8% | 8.7% | 12.3% |

LP-only macro precision fell from 0.537 to 0.444 at threshold 0.5 while recall held (0.918 to 0.916).

## Why
Training against cosine-similar in-group negatives compresses sigmoid scores toward 0.5, so the LP fires on about 20% more (case, label) pairs at any fixed threshold, inflating Slight. QW2's regularisation tightened generalisation but did not undo the compression. The bundle at LP 0.8 had the best Good and CO of any config, but FN climbed enough to pull G+S below baseline. The case-disjoint split confirmed that earlier validation scores were inflated by same-case leakage.

## Don't retry unless…
You change the architecture (for example a softmax over in-group labels instead of independent sigmoids). The fraction 0.3 retry also failed ([retry record](2026-05-13-lp-hard-neg-retry-and-top3-rerank.md)); fraction 0.5 was never tried.

## Where it went
`--label-presence-hard-neg-fraction` and `--label-presence-patience` were deleted in the 2026-05-25 cleanup. The case-disjoint `GroupShuffleSplit` stayed and is the current LP validation split. Recipe values today are the Phase 28 ones (dropout 0.3, weight decay 1e-4, recall weight 0.5, 5 random negatives per positive).
