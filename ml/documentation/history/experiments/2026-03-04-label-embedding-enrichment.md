# Label embedding enrichment (diagnosis strings, then report centroids)

**Date:** 2026-03-03 to 2026-03-04 · **Verdict:** reverted

## Hypothesis
Label text is only `"{term} {group}"`. Blending each label embedding 50/50 with extra clinical vocabulary would close the domain gap between labels and full-report embeddings.

## Setup
- Fix 6 (Phases 6-8): blend with the mean embedding of each label's keyword-matched diagnosis strings.
- Fix 9 (Phase 10): blend with the centroid of cached report embeddings for keyword-confirmed cases (no extra PetBERT pass).
- Baseline: the unenriched single all-label classifier (see [that record](2026-03-03-single-presence-classifier-co-bank.md)). Old evaluator, keyword ground truth.

## Result
| Variant | Best cycle Good | Slight | G+S | Compared with (Good / Slight / G+S) |
|---|---|---|---|---|
| Diagnosis-string blend (Phases 6-8, co10 c5) | 7.4% | 14.7% | 22.1% | 7.7% / 15.2% / 22.9% in Phase 5 without it |
| Report-centroid blend (Phase 10, best c9 of 18 cycles) | 5.7% | 8.4% | 14.1% | 7.3% / 13.1% / 20.4% in Phase 9 without it (best cycle; range 17.5-20.4%) |

Per-cycle Good and Slight come from `training-log/training-log-binary.md`; the summary in that log rounds Phase 10 down to about 13-14.1%.

## Why
- Diagnosis strings and label strings already sit close together in PetBERT space, so blending barely moved the label vector.
- A 50/50 report-centroid blend changed the cosine landscape enough to invalidate the accumulated CO bank (full cold start needed, ceiling still lower) and produced a point that represents neither a label nor a report.
- The real fix for the domain gap was contrastive adaptation of the backbone ([Phase 17](2026-03-23-contrastive-backbone-infonce.md)).
- The first enriched cycle also scored with the un-enriched label embeddings by mistake (fixed 2026-03-03).

## Don't retry unless…
You use a much lighter blend (about 0.1-0.2 report share) and can afford to reset any negative bank. The original notes flagged this as untested.

## Where it went
`ml/labels/enrichment.py` and the `--enrich-labels-csv` flag were removed with the old tree. Old paths are historical.
