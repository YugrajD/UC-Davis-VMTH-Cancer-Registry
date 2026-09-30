# GroupClassifier training recipe: epochs, dropout, learning rate, LR schedule

**Date:** 2026-05-04 to 2026-05-07 (Phases 26-29) · **Verdict:** adopted (epochs 300, lr 5e-5, dropout 0.1); lr 2e-5, dropout 0.05 and cosine schedule reverted

## Hypothesis
The GroupClassifier was still improving when it stopped (best epoch 120 of 150) and dropout 0.3 might over-regularise at 46k training cases.

## Setup
- Metric: macro F1 over groups on the validation split (0.3 threshold), plus end-to-end sweeps on the test set.
- Fixed guards: `max_class_weight` 50, `weight_decay` 1e-3.

## Result
| Experiment | Macro F1 | Verdict |
|---|---|---|
| Phase 24, 150 epochs, dropout 0.3 | 0.3136 | baseline |
| 300 epochs, lr 5e-5 (epoch 219) | 0.4335 | adopted (+0.120) |
| lr 2e-5 (epoch 278) | 0.4249 | reverted |
| dropout 0.1 (epoch 192) | **0.4475** | adopted |
| dropout 0.05 (epoch 221) | 0.4399 | reverted |
| dropout 0.05 + cosine warm restarts, 600 epochs | 0.4393 | reverted |

End-to-end on the test set (gate 0.5, group 0.85): 300 epochs moved G+S from 53.6% to 54.6% and CO from 23.4% to 22.3% (Good and Slight not split out); dropout 0.1 with the new subtype keywords gave 54.9% G+S, CO 21.1%, FP 5.1%, FN 18.9% at group threshold 0.85 (0.90: G+S 56.1%, CO 15.9%, FN 23.0%). These figures use the old evaluator.

On the later concat-3 backbone the same recipe reached macro F1 0.5712 (epoch 258), so the hyperparameters carried over unchanged. A Phase 29 sweep on an intermediate embedding space preferred dropout 0.05 (F1 0.4435 against 0.4308 for 0.1), but the production recipe stayed at 0.1.

## Why
Lower dropout helped once data volume grew; the LR schedule made no difference once dropout was tuned; slower learning rates found no better minimum.

## Don't retry unless…
The data volume or the embedding space changes materially (the dropout optimum moved once already). See also [focal loss and ASL](2026-05-06-focal-loss-and-asl.md).

## Where it went
Values are pinned in `report_mapping/training/recipe.py` (`GROUP`: epochs 300, lr 5e-5, dropout 0.1, weight decay 1e-3, max class weight 50) and by `tests/test_recipe.py`.
