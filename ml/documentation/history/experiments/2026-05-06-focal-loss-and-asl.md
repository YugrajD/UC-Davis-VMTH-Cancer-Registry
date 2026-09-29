# Focal loss and Asymmetric Loss (ASL) for the GroupClassifier

**Date:** 2026-05-06 (focal) and 2026-05-07 (ASL) · **Verdict:** reverted

## Hypothesis
The group head over-predicts (high recall, low precision per group). Down-weighting easy examples (focal loss, gamma 2) or using separate positive and negative focusing (ASL, gamma+ 1, gamma- 2 or 4, margin 0.05) should sharpen precision.

## Setup
- Baseline: BCE with `pos_weight` capped at 50. Macro F1 0.4475 (dropout 0.1) for focal, 0.4435 (dropout 0.05, 26 groups) for ASL.
- Metric: validation macro F1; runs that collapsed were not carried to the pipeline.

## Result
| Run | Macro F1 | Outcome |
|---|---|---|
| Focal (gamma 2), dropout 0.05, cosine LR, 600 epochs | 0.2300 | collapsed |
| ASL gamma+ 1.0, gamma- 4.0, with pos_weight | 0.0772 | collapsed (all groups recall about 1, precision about 0) |
| ASL gamma+ 1.0, gamma- 2.0, with pos_weight | 0.1836 | collapsed |
| ASL gamma+ 1.0, gamma- 4.0, no pos_weight | 0.0930 | collapsed |
| BCE reference | 0.4435-0.4475 | unbeaten |

The metric here is macro F1, not the G+S / Good / Slight pipeline numbers.

## Why
- pos_weight (up to 50x) plus a large gamma- double-suppresses negatives and drives the model to "always positive".
- Removing pos_weight did not help: with groups of 114-3,322 positives against about 40k negatives, gamma- 4 drives the negative gradient to about zero below p 0.3, so predicting positive everywhere is the minimum-loss strategy. ASL was designed for labels with thousands of positives and negatives (for example MS-COCO).

## Don't retry unless…
You have a per-group positive count in the thousands, or a different imbalance strategy. The original notes said not to retry focal loss without first diagnosing the instability.

## Where it went
`--focal-loss`, `--focal-gamma`, `--asl*` flags and their loss classes were deleted in the 2026-05-25 production cleanup. Production loss is BCE with capped `pos_weight`. Related: [epochs, dropout and LR](2026-05-04-groupclassifier-epochs-dropout-lr.md).
