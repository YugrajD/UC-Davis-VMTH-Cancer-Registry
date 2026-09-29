# 0002 Adapt the PetBERT backbone with per-section contrastive pairs

**Status:** accepted · **Date:** 2026-05-12

## Context
Base PetBERT is pretrained with masked-language modelling on veterinary records; its geometry does not by itself put a report near its correct label. An InfoNCE adaptation on whole-report text (Phase 17) helped a lot, but concat-3 embeds each section separately, so the sections were off-manifold for a backbone trained on merged text.

## Decision
Train the backbone with InfoNCE on `(section_text, "term group")` pairs built per section, matching what inference embeds. Recipe (`ml/report_mapping/training/recipe.py`): 3 epochs, batch 32, lr 2e-5, temperature 0.07, max length 256 in training and 512 at inference, seed 42, starting from the base `SAVSNET/PetBERT`. The adapted backbone is stored inside each generation as `petbert/`, and `--model` on `predict.py` and `train.py` accepts any HuggingFace directory or name (the default for prediction and heads-only training is the generation's own backbone).

## Consequences
- Backbone weights are part of the embedding fingerprint, so retraining the backbone invalidates every head and cached embedding. The backbone stage therefore only runs when explicitly requested (`retrain_cycle.py --backbone`) and lands in `candidate/`.
- A hard-negative margin loss on top of it failed twice and is not in the recipe.
- Loss was still falling at epoch 3; more epochs are untested.

## Evidence
[per-section backbone](../experiments/2026-05-12-per-section-contrastive-backbone.md) (G+S 13.1% to 24.0% on the ungated prototype), [first InfoNCE adaptation](../experiments/2026-03-23-contrastive-backbone-infonce.md), [hard-negative attempts](../experiments/2026-03-27-backbone-hard-neg-margin.md). Current description: [../../concepts/report-mapping.md](../../concepts/report-mapping.md).
