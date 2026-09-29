# Per-column embeddings and a wider hidden layer (Phases 13 and 16)

**Date:** 2026-03-05 and 2026-03-22 · **Verdict:** adopted (later replaced by [concat-3](2026-05-12-concat-3-text-representation.md))

## Hypothesis
- Phase 13: averaging the report columns into one 768-dim vector dilutes a strong diagnostic column, so feeding each column separately should help.
- Phase 16: with three columns the input is 3072-dim; a 256-unit hidden layer is a 12:1 bottleneck.

## Setup
- Baseline: Phase 12 mean embedding, hidden 256 (G+S 32.0%).
- Phase 13: concatenate the three column embeddings with the label embedding. Phase 16: hidden 256 to 512, no cold start.
- Old evaluator, keyword ground truth.

## Result
| Phase | Good | Slight | G+S | CO | FP | FN |
|---|---|---|---|---|---|---|
| 12 mean embedding | n/a | n/a | 32.0% | 32.1% | 33.6% | 1.3% |
| 13 per-column concat (c6) | 12.8% | 27.2% | 40.0% | 30.4% | 28.5% | 1.1% |
| 16 hidden 512 (c2) | 12.7% | 29.2% | 41.9% | 29.6% | 27.2% | 1.2% |

## Why
- Per-column input stopped the report mean from washing out a strong column (+8.0 pp G+S).
- Doubling the hidden width gave +1.9 pp immediately, confirming the compression was a bottleneck. It mostly helped term-level discrimination, not group confusion (CO barely moved).
- In Phase 13, `co-neg-per-case` 10 regressed (30.4% to 26.6%) and 5 was kept.

## Don't retry unless…
Not applicable to adopted steps. A later test of hidden 2048 on the 2304-dim gate stack found no gain ([case-presence gate](2026-05-12-case-presence-gate-2304.md)): capacity was no longer the bottleneck.

## Where it went
The idea survives as concat-3 (three per-section embeddings concatenated to 2304 dims) and the per-group LabelPresence head's `n_cols=3` design. Old paths are historical. Two variants that did not work are in [per-pair architecture](2026-03-21-per-pair-architecture.md).
