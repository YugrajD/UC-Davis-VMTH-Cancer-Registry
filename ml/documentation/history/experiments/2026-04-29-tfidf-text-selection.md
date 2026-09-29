# TF-IDF multi-column text selection (and the fallback chain before it)

**Date:** 2026-04-27 to 2026-04-29 (Phase 24) · **Verdict:** superseded by [concat-3](2026-05-12-concat-3-text-representation.md); removed 2026-05-20

## Hypothesis
- Fallback chain (2026-04-27): 826 cases (1.4% of 58,313) had no text in the three columns the pipeline read; taking the first non-empty column from a priority list would cover them.
- TF-IDF selection (Phase 24): the chain discarded secondary columns, losing signal. Concatenate HISTOPATHOLOGICAL SUMMARY + FINAL COMMENT + COMMENT and, when it overflows 512 tokens, keep the highest-scoring sentences by TF-IDF.

## Setup
- Column fill rates over 58,313 cases: histopathological summary 97.1%, comment 66.4%, final comment 31.5%; gross description and clinical abstract excluded as low value.
- 73.3% of cases fit in 512 tokens as-is; 26.7% overflowed and were compressed.
- The selector had to be applied identically in training and inference or the embedding space drifts silently.

## Result
| Step | Result |
|---|---|
| Fallback chain | 825 of 826 silent cases now covered (2 truly empty) |
| TF-IDF selection | Binary G+S about 47% (Phase 23 plateau) to 63.8% at cycle 10; GroupClassifier macro F1 0.192 to 0.3136 |

These are old-evaluator, train-side numbers. The Phase 25 three-stage result is recorded in [the three-stage record](2026-04-29-three-stage-gate-group-keyword.md); Good and Slight were not split out for this step.

## Why
Recovering secondary columns added diagnostic signal, but 26.7% of cases still lost sentence-level detail to truncation and the merged string flattened section structure. Concat-3 fixed both.

## Don't retry unless…
Not applicable; the truncation problem is what removed it.

## Where it went
`ml/text_selection/`, `--text-cols` and `--tfidf-vectorizer` were deleted on 2026-05-20, and the cache key `tfidf_selected` became `concat_3`. A TF-IDF baseline tree was preserved as a sibling directory and later deleted (2026-09-14). Old paths are historical.
