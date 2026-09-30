# KNN group gating (hybrid architecture)

**Date:** 2026-03-23 · **Verdict:** reverted

## Hypothesis
A k-nearest-neighbour group selector (k=10, built from LLM predictions, 2304-dim) can restrict which groups the binary classifier may pick from, cutting completely-off errors. Both checkpoints already existed, so no training was needed.

## Setup
- Baseline: Phase 16 binary classifier, 37.8% G+S against LLM ground truth (about 12.6k cases, about 150 per group).
- Gate thresholds 0.1, 0.2 and 0.3 on the KNN vote fraction.

## Result
| Mode | G+S | CO | FP | FN |
|---|---|---|---|---|
| Binary only | 37.8% | 30.1% | 30.3% | 1.8% |
| Hybrid t=0.1 | 5.6% | 37.6% | 53.0% | 3.8% |
| Hybrid t=0.2 | 7.5% | 31.9% | 47.7% | 13.0% |
| Hybrid t=0.3 | about 6.1% | about 28% | about 39% | 26.6% |

The Slight rate fell from 27.5% to about 2%. The source table mixes two ground truths (the t=0.3 row was measured against keyword labels), so the last row is approximate.

## Why
- If KNN votes for the wrong groups the correct group is removed entirely (CO or FN).
- Non-cancer reports find cancer neighbours regardless, so FP rose to 53%.
- The sparse reference set (about 150 cases per group) meant many true cases could not clear any threshold.

## Don't retry unless…
The registry has roughly 15k or more labelled cases, as the original note said. Today a learned gate plus group classifier already does the job.

## Where it went
`run_categorization_hybrid()` was preserved for a while and then deleted with the old tree. See [three-stage gate, group and keyword correction](2026-04-29-three-stage-gate-group-keyword.md) for what replaced it.
