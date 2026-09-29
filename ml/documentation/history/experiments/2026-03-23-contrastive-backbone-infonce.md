# Contrastive backbone adaptation (InfoNCE, Phase 17)

**Date:** 2026-03-23 · **Verdict:** adopted

## Hypothesis
Frozen PetBERT puts report text and label text in different regions, so the similarity between a report and its correct label is not reliably higher than against wrong labels. Fine-tuning the backbone with InfoNCE on `(report_text, label_text)` positive pairs should align them.

## Setup
- Baseline: Phase 16, frozen PetBERT, hidden 512, 41.9% G+S (old evaluator, keyword ground truth).
- Round 1: 7,398 pairs, 3 epochs, batch 32, lr 2e-5, temperature 0.07 (InfoNCE loss 1.90, 1.36, 1.22), then a cold start of the classifier (5 CO negatives, 10 FP negatives per case).

## Result
| Run | Good | Slight | G+S | CO | FP | FN |
|---|---|---|---|---|---|---|
| Phase 16, frozen backbone (c2) | 12.7% | 29.2% | 41.9% | 29.6% | 27.2% | 1.2% |
| Phase 17, adapted backbone (c8) | 20.4% | 48.6% | 69.0% | 6.9% | 23.7% | 0.3% |

Same old evaluator for both rows; not comparable with post-May numbers. A later run of the same idea on the train/test split (Phase 22) gave the first honest out-of-sample figures; see `history/training-log/training-log-finetune.md`.

## Why
Adapting the embedding space, not the head, was what removed most wrong-group predictions (CO -22.7 pp). Everything after it builds on the adapted backbone.

## Don't retry unless…
Not applicable (adopted). Variants: hard-negative margin loss did not help ([two attempts](2026-03-27-backbone-hard-neg-margin.md)); training on whole-report text rather than per section was later found to be the largest lever left on the table ([per-section backbone](2026-05-12-per-section-contrastive-backbone.md)).

## Where it went
Kept as `report_mapping/training/backbone.py` (InfoNCE, per-section pairs; see [../../concepts/report-mapping.md](../../concepts/report-mapping.md)). Decision: [0002](../decisions/0002-per-section-contrastive-backbone.md).
