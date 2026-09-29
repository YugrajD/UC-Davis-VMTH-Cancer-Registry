# Per-section contrastive backbone

**Date:** 2026-05-12 · **Verdict:** adopted

## Hypothesis
The InfoNCE backbone was trained on TF-IDF-merged text, but concat-3 inference embeds each section separately. Per-section vectors were off-manifold for that backbone. Training on per-section `(section_text, label_text)` pairs aligns training with inference.

## Setup
- Baseline: default PetBERT with concat-3 (G+S 13.1%) in the three-stage prototype tree (experimental sibling tree, since deleted), test n=11,661 cases.
- Pairs: 56,109 (histopathology 25,804; final comment + comment 26,122; ancillary 4,183); sections under 10 characters skipped; 44 of 21,411 annotated train cases dropped as all-empty.
- Same InfoNCE trainer: 3 epochs, lr 2e-5, batch 32, temperature 0.07, max length 256 (about 44 min on the XPU; loss 1.97, 1.58, 1.45, still falling).

## Result
| Metric | Default PetBERT | Per-section backbone | Delta |
|---|---|---|---|
| Good | 2.0% | 6.2% | +4.2 |
| Slight | 11.1% | 17.8% | +6.7 |
| G+S | 13.1% | 24.0% | +10.9 |
| CO | 30.7% | 20.2% | -10.5 |
| FP | 51.1% | 53.9% | +2.8 |
| FN | 5.0% | 1.9% | -3.1 |

These numbers are from the ungated prototype (FP is high because no case-presence gate existed yet). The full-stack result is in the [concat-3 record](2026-05-12-concat-3-text-representation.md).

## Why
Good tripled, CO fell by a third and FN halved. Each section vector is now label-aware on its own, so the downstream head sees three aligned views instead of three off-manifold ones. The small FP uptick is the classifier being more confident. This was the largest single lever found.

## Don't retry unless…
Not applicable (adopted). Loss was still falling at epoch 3; more epochs were never tested. A hard-negative variant failed ([backbone hard-neg](2026-03-27-backbone-hard-neg-margin.md)).

## Where it went
Kept: `report_mapping/training/backbone.py` builds per-section pairs and runs InfoNCE (`BACKBONE` recipe: 3 epochs, batch 32, lr 2e-5, temperature 0.07, max length 256). The backbone lives inside each generation directory as `petbert/`. Decision: [0002](../decisions/0002-per-section-contrastive-backbone.md).
