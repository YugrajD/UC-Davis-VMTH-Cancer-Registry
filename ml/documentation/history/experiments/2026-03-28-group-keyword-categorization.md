# Group-keyword categorization (behavior-digit filter)

**Date:** 2026-03-28 · **Verdict:** adopted

## Hypothesis
At 70.4% G+S about 42% of rank-1 predictions were "slightly off": the right ICD group but the wrong term inside it. Inside a group the main disambiguator is the ICD-O behavior digit (`/0` benign, `/1` borderline, `/2` in situ, `/3` malignant, `/6` metastatic), which maps to plain vocabulary and needs no training.

## Setup
- Baseline: default top-k categorization on the Phase 18 checkpoint.
- New mode: predicted group first, then a behavior-word filter picks the term. Old evaluator, top-k rows per case, about 17.8k rows.

## Result
| Mode | Good | Slight | G+S | CO | FP | FN |
|---|---|---|---|---|---|---|
| Default top-k | 30.6% | 55.9% | 86.5% | 9.4% | 1.9% | 2.2% |
| Group-keyword | 56.5% | 30.0% | 86.5% | 9.3% | 1.9% | 2.2% |

Per-group Good rose sharply: Blood vessel tumors 21% to 70%, Adenomas and adenocarcinomas 32% to 71%, Squamous cell 24% to 79%, Osseous 18% to 52%. These are top-k figures and not comparable with the later top-1 numbers.

## Why
The step only redistributes Slight into Good; it cannot change which cases pass or fail, hence identical G+S. Groups where terms differ by topography or histologic subtype (Meningiomas 80% Slight, Odontogenic 82%, Osseous 46%, Gliomas 46%) needed subtype vocabularies, which came next ([subtype keywords](2026-05-04-argmax-fallback-threshold-and-subtype-keywords.md)).

## Don't retry unless…
Not applicable (adopted).

## Where it went
Kept as stage 4 of report mapping: `report_mapping/inference/keyword_correction.py` with `taxonomy/behavior.py` (see [../../concepts/report-mapping.md](../../concepts/report-mapping.md)). Decision: [0003](../decisions/0003-four-stage-report-mapping.md).
