# End-to-end fine-tuned PetBERT group classifier (FinetuneGroupClassifier)

**Date:** 2026-05-07 to 2026-05-08 · **Verdict:** reverted (abandoned)

## Hypothesis
Fine-tuning PetBERT end to end with a linear head would give better group probabilities than the frozen-embedding MLP, and the gain would carry through to G+S.

## Setup
- Baseline: Phase 28, 57.9% G+S (LP threshold 0.5) and 55.9% (0.9), old evaluator, test set.
- Backbone plus linear head, 25 groups, BCE with `pos_weight`, 46,626 cases (37,301 train / 9,325 validation), batch 8, lr 2e-5.
- Phase A: standalone benchmark. Phase B: swap into the four-stage pipeline. Then a threshold sweep and an 8-epoch retrain.

## Result
| Run | Good | Slight | G+S | CO | FP | FN |
|---|---|---|---|---|---|---|
| Standalone, no gate/LP/KW, 3 epochs | 3.3% | n/a | 32.2% | 27.0% | 36.3% | 4.5% |
| Swapped in, defaults (group 0.5) | 18.4% | 24.4% | 42.8% | 44.9% | 8.5% | 3.8% |
| 3 epochs, group 0.85, LP 0.5 | 25.4% | 30.2% | 55.6% | 26.4% | 8.8% | 9.3% |
| 8 epochs, group 0.85, LP 0.5 | 26.2% | 30.2% | 56.4% | 23.0% | 9.1% | 11.6% |
| 8 epochs, group 0.85, LP 0.9 | 35.6% | 20.3% | 55.9% | 19.7% | 7.8% | 16.7% |

Validation macro F1 rose from 0.4475 (frozen MLP) to 0.4789 (3 epochs) and 0.5774 (8 epochs, still climbing).

## Why
The F1 gain (+0.10) turned into only +0.8 pp G+S. The LP heads still read frozen contrastive-backbone embeddings, so they were not aligned with the new group choice: 8 of 25 groups had no matching LP and fell back to cosine (Acinar cell 84% CO, Mature T/NK 85%). Closing that gap ("Phase C": regenerate the cache, retrain gate and all LPs, about 5 h) had an estimated 60% chance of +1-3 pp. Inference cost was the deciding factor: about 9 min per run against about 10 s, and every group retrain became a 90 min job.

## Don't retry unless…
You have a separate reason to own a fine-tuned backbone (similar-case retrieval, dedup). If revisited: train at least 10 epochs, gate before tokenising, and do Phase C first; threshold sweeps without it are wasted.

## Where it went
Model, trainers, `finetune_pipeline/` and the `--finetune-group-classifier` flag were removed. Note: three stray files (`ml/training/finetune/{build_dataset,dataset,train}.py`) came back into git with the 2026-09-27 merge from another branch; they import modules that no longer exist and are dead code. Artifacts were archived locally as `2026-05-08_finetune-stage2-abandoned/` (later pruned).
