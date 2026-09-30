# Backbone hard-negative margin loss (three attempts, 2026-03 and 2026-05)

**Date:** 2026-03-27 (Phases 20-21) and 2026-05-20 (per-section retry, "LB2") · **Verdict:** reverted (a milder April variant, noted below, was used for a few weeks and then dropped)

## Hypothesis
After InfoNCE, add a margin loss on hard-negative triplets `(report, correct label, wrong label)` mined from the CO bank (wrong-group predictions) to push confusable labels apart in embedding space.

## Setup
- 2026-03-27: baseline Phase 18, 70.4% G+S (old evaluator). Triplet weight 0.5 (Phase 20) and 0.25 (Phase 21), margin 0.3, about 24.3k triplets.
- 2026-05-20: baseline `ml/` production, G+S 62.1% (eval half, n=4,414). Fresh triplets re-mined per section: 56,109 section pairs and 12,285 triplets; weight 0.5, margin 0.3, 3 epochs from the base PetBERT (about 8 h on an RTX 5070 Ti); cold start of gate, group and LP heads, then a threshold sweep (best gate 0.60-0.70, group 0.90).

## Result
| Attempt | Good | Slight | G+S | CO | FP | FN |
|---|---|---|---|---|---|---|
| 2026-03 baseline, Phase 18 c16 (old evaluator) | 20.3% | 50.1% | 70.4% | 6.5% | 22.7% | 0.4% |
| weight 0.5, Phase 20 c2 | 21.9% (+1.6) | 46.6% (-3.5) | 68.5% (-1.9) | 6.9% | 24.2% | 0.3% |
| weight 0.25, Phase 21 c4 | 22.1% (+1.8) | 47.3% (-2.8) | 69.4% (-1.0) | 6.1% | 24.1% | 0.4% |
| 2026-05 baseline (eval half) | 46.1% | 16.0% | 62.1% | 14.7% | 2.3% | 20.8% |
| 2026-05 per-section retry, best operating point | 45.5% (-0.6) | 16.1% | 61.6% (-0.5) | 13.1% (-1.6) | 3.2% (+0.9) | 22.1% (+1.3) |

The March rows are each phase's best cycle, taken from `training-log/training-log-finetune.md`. The older ideas-rejected summary quoted "Good +0.7 pp, Slight -1.4 to -2.1 pp", which does not match these best-cycle columns (Good +1.6 and +1.8, Slight -3.5 and -2.8); the two sources conflict and the summary probably averaged over cycles. Both agree on the direction (Good up, Slight down), which matches May.

April variant: a "Round 2" backbone (257,079 triplets, weight 0.25, lr 1e-5, 2 epochs) raised GroupClassifier macro F1 from 0.1815 to 0.1922 in Phase 23 and stayed in the production backbone until the May 2026 backbone retrain, which used InfoNCE only.

## Why
- The loss worked as designed (CO fell 1.6 pp in May) but the embedding space became more recall-leaning; FP and FN rose more than CO fell, and no operating point recovered it. Per-stage F1 was flat or slightly up, so the stages themselves were not broken.
- In March the loss raised Good but suppressed Slight (model over-shot and dropped borderline matches). The original reading was "a data ceiling, not an embedding-space ceiling".
- The multi-stage gate structure does not absorb a recall shift the way a single-stage pipeline does (see [broad-vs-picker diagnostic](2026-05-24-broad-vs-picker-diagnostic.md)).

## Don't retry unless…
Only with a different loss or training design; the plain margin loss has now failed on two backbones. The original notes also suggested not before about 8,000 confirmed cancer cases (the corpus is now larger than that, and the May run still failed).

## Where it went
Code removed in the 2026-05-25 production cleanup (`--hard-neg-csv`, `--hard-neg-weight`, `--hard-neg-margin`, `HardNegPairDataset`, `build_hard_neg_pairs`). The failed May generation was archived locally as `2026-05-20_backbone-hardneg-FAILED/` and later deleted in the 2026-09-14 output prune. Related: [LP hard-negatives](2026-05-10-lp-hard-negatives-qw1-qw2.md).
