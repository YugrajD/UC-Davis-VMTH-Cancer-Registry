# Argmax fallback, group threshold 0.85 and subtype keywords (Phase 26)

**Date:** 2026-05-04 to 2026-05-06 · **Verdict:** adopted (gate threshold 0.4 reverted)

## Hypothesis
- Every "Unidentified Cancer" output is a guaranteed miss, yet many gate-passed cases have a top-group probability of 0.82-0.89, just under the 0.90 threshold, where the group is often right.
- The group threshold and the gate threshold were untuned.
- Groups whose terms differ by histologic subtype (Meningiomas, Osseous, Gliomas) can be split by keyword rules after the fact.

## Setup
Test set, old evaluator, group classifier fixed. Steps applied cumulatively.

## Result
| Config | G+S | CO | FP | FN | Rows |
|---|---|---|---|---|---|
| group 0.90 (baseline) | 51.8% | 19.3% | 4.7% | 24.1% | 8,744 |
| group 0.85 | 52.3% | 24.0% | 4.9% | 18.8% | 9,335 |
| group 0.80 | 50.4% | 29.2% | 5.0% | 15.4% | 10,096 |
| 0.85 plus argmax fallback | 53.6% | 23.4% | 4.9% | 18.1% | 9,256 |
| plus subtype keywords (3 groups) | 53.6% | 23.4% | 4.9% | 18.1% | 9,255 |
| gate 0.4 instead of 0.5 (group 0.85) | 51.7% | 24.1% | 5.8% | 18.4% | 9,467 |

Good and Slight were not split out. Subtype keywords changed the total by only 0.1 pp but lifted Meningiomas Good from 3% to 10%. Gate 0.4 raised FP by 0.9 pp for a 0.4 pp FN gain, so 0.5 stayed. The keyword set later grew to 6 groups (May 2026) and is 7 groups in the current code.

## Why
The fallback eliminated "Unidentified Group" outputs for gate-passed cases (+1.3 pp G+S). Below 0.85 low-confidence groups flood in as CO.

## Don't retry unless…
Not applicable to adopted steps. Note the original "CLI default is 0.3, always pass 0.85" trap is gone: thresholds now come from the generation's calibrated `thresholds.json`.

## Where it went
Argmax fallback and subtype filters live in `report_mapping/inference/stages.py` and `keyword_correction.py`, subtype rules in `taxonomy/subtype.py` (7 groups). The fallback is always on. Thresholds are refit by `calibrate.py`; see [../../concepts/report-mapping.md](../../concepts/report-mapping.md). An Adenomas subtype set was never built (see [open ideas](../open-ideas.md)).
