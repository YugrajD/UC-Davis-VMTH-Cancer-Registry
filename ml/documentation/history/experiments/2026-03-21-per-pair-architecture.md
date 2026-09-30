# Per-pair architecture (shared MLP per column, max-pool or learned combine)

**Date:** 2026-03-21 to 2026-03-22 · **Verdict:** reverted

## Hypothesis
Scoring each (column, label) pair with a shared MLP and combining the three logits (max-pool in Phase 14, learned `Linear(3->1)` in Phase 15) halves the input compression and separates column-label relationships.

## Setup
- Baseline: Phase 13 concat architecture, 40.0% G+S.
- Old evaluator, keyword ground truth, all on XPU.

## Result
| Variant | Good | Slight | G+S | CO | FP | FN | Delta G+S vs Phase 13 |
|---|---|---|---|---|---|---|---|
| Phase 13 concat baseline (c6) | 12.8% | 27.2% | 40.0% | 30.4% | 28.5% | 1.1% | n/a |
| Phase 14, max-pool (c6) | 10.1% | 22.6% | 32.7% | 32.7% | 32.9% | 1.8% | -7.3 pp |
| Phase 15, learned combine (c5) | 10.7% | 24.1% | 34.8% | 32.3% | 31.4% | 1.6% | -5.2 pp |

Good fell 2.7 and 2.1 pp and Slight fell 4.6 and 3.1 pp against Phase 13 (both from `training-log/training-log-binary.md`).

## Why
1. One shared MLP cannot learn column-specific language (histopathology, comment and ancillary text differ).
2. Max-pool lets one column that merely mentions a label (a differential, a ruled-out finding) outrank the true signal.
3. Scoring columns independently drops joint patterns such as "column 1 says X and column 2 says Y".

The learned combiner recovered part of the loss but shared weights remained the core problem.

## Don't retry unless…
You give each column its own weights. Note that the later per-group LabelPresence head (`n_cols=3`, shared MLP, learned combine) does use this shape and works, because it runs inside one group on the contrastive-adapted concat-3 embeddings.

## Where it went
Code removed with the old binary tree. Checkpoint backups were local and are gone. Old paths are historical.
