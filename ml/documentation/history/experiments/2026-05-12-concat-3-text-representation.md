# concat-3 text representation

**Date:** 2026-05-12 (prototype), promoted to `ml/` 2026-05-13 · **Verdict:** adopted (supersedes [TF-IDF text selection](2026-04-29-tfidf-text-selection.md))

## Hypothesis
TF-IDF selection merged three columns into one 512-token string and 26.7% of cases overflowed and lost sentence-level detail. Embedding each section on its own (its own 512-token budget) and concatenating the three 768-dim vectors into 2304 dims keeps section structure that later heads can weight.

## Setup
- Sections: histopathological summary; final comment plus comment; ancillary tests.
- Sweep in a three-stage prototype tree (no LP heads; experimental sibling tree, since deleted): arms {tfidf, mean_pool, mean_5, mean_6, concat_3, concat_5, concat_6} on default PetBERT, test n=61,203 prediction-cases.
- Promotion: the four-stage prototype (concat-3 + per-section backbone + 2304-dim gate + per-group LPs with `n_cols=3`) against the preserved TF-IDF four-stage baseline, same evaluator.

## Result
| Comparison | Good | Slight | G+S | CO | FP | FN | n |
|---|---|---|---|---|---|---|---|
| Prototype sweep, TF-IDF arm | n/a | n/a | 11.6% | n/a | n/a | n/a | 61,203 |
| Prototype sweep, concat_3 arm | 2.0% | 11.1% | 13.1% | 30.7% | 51.1% | 5.0% | 61,203 |
| TF-IDF four-stage baseline | 37.3% | 19.3% | 56.6% | 16.8% | 7.8% | 18.7% | 10,334 |
| Promoted four-stage, eval half | 46.1% | 16.0% | 62.1% | 14.7% | 2.3% | 20.8% | 4,414 |
| Promoted four-stage, full test | 46.6% | 15.7% | 62.3% | 14.4% | 2.3% | 21.0% | 8,835 |

Alone, concat-3 gave +1.5 pp G+S over TF-IDF on default PetBERT; most of the promoted gain came from the stack ([per-section backbone](2026-05-12-per-section-contrastive-backbone.md), [2304-dim gate](2026-05-12-case-presence-gate-2304.md), LPs with `n_cols=3`). GroupClassifier macro F1 rose from 0.4475 to 0.5712. The promotion was +5.5 pp G+S, +8.8 pp Good, FP -5.5 pp, FN +2.1 pp (stricter gate).

Caveat on the 62.1%: it does not reproduce from files on disk today. The reproducible reference is 61.76% ([decision 0005](../decisions/0005-ml-rewrite-and-cutover.md)).

## Why
Concat beat mean-pooling at 3 sections but lost at 5, so section quality matters more than capacity. Section-aware inputs also let the LP head learn which section matters per group.

## Don't retry unless…
Not applicable (adopted). About 0.25% of cases have all three sections empty (text lives in gross description or clinical abstract) and get a near-zero vector; a fallback was proposed but not built.

## Where it went
`report_mapping/sections.py` defines the sections (`SECTION_SPEC_VERSION = 1`) once; the cache key `tfidf_selected` was renamed `concat_3` on 2026-05-20 when TF-IDF code was removed. Decision: [0001](../decisions/0001-concat-3-per-section-text-representation.md). Current description: [../../concepts/report-mapping.md](../../concepts/report-mapping.md).
