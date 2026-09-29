# Demographic features for the gate and group heads

**Date:** 2026-05-29 · **Verdict:** reverted

## Hypothesis
Age, sex and breed (from a demographics table, 58,313 rows, 100% canine) carry information beyond the report text and would help the gate (stage 1) and group head (stage 2).

## Setup
- Non-text features bypass PetBERT: a fit-on-train encoder produced a 45-dim block (age z-score plus missing flag, sex one-hot, species one-hot, top-30 breeds plus Other plus Missing, two zip-present flags), late-fused onto the 2304-dim vector (input 2349).
- Baseline: promoted `ml/`, G+S 62.1% (Good 46.1, Slight 16.0, CO 14.7, FP 2.3, FN 20.8, eval half). Test split n about 9,097, group 0.85, gate 0.80.

## Result
| Metric | Baseline | With demographics |
|---|---|---|
| Good | 46.1% | 46.2% |
| Slight | 16.0% | 15.8% |
| CO | 14.7% | 15.2% |
| FP | 2.3% | 3.6% |
| FN | 20.8% | 19.2% |
| Group macro F1 | 0.5712 | 0.5736 |
| Gate validation F1 | 0.942 | 0.941 |

Good flat and Slight -0.2 pp fails the bar of "Good and Slight each at least baseline". The FN-down, FP-up swing is a retrained-gate operating-point shift, not a demographics gain.

## Why
Species is a dead feature (all canine). Breed and age add little beyond what histology text already encodes for ICD-O group choice. Stage 3 (LabelPresence) was not attempted because it was gated on a positive stage 1+2 result that never appeared.

## Don't retry unless…
You drop species, keep only age and breed, and target the gate's recall/precision trade rather than the Good rate. Plain late-fusion concatenation is a known dead end.

## Where it went
Reverted to the canonical checkpoints; the flag and encoder were later removed with the old tree. At inference the pipeline keyed on the checkpoint's own metadata, not the CLI flag, so rollback was just which checkpoint sat at the production path. Experiment archives were deleted in the 2026-09-14 output prune.
