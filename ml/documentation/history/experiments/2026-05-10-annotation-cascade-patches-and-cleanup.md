# Annotation cascade patches, model bake-off and ensemble cleanup

**Date:** 2026-05-09 to 2026-05-10 · **Verdict:** adopted (cascade patches and ensemble cleanup; cleanup is built and on by default, but its full-scale pass was estimated at about 150 h and killed, so it has not completed at full scale, and whether it ran for `silver-0-legacy` is unknown because that manifest has `cleanup_enabled` null)

## Hypothesis
A 90-row stratified audit of the LLM annotation (188,775 rows, 58,208 cases) found about 7% flat-wrong rows and 25-30% over-broad labels, so label noise might cap the model's ceiling. Patch the cascade and add an ensemble verification pass.

## Setup
- Five cascade patches: a pre-tier-1 negation gate, behavior-aware tier 2 (threshold 0.7 when a behavior word is present), masking of "non-X" compounds, token-overlap group identification, and injected anatomic-site context in the tier-3 prompt.
- Cleanup: two diverse local models vote on each confirmed tier 1/2/3 match; unanimous wins, otherwise "Uncertain" (or an optional tiebreaker).
- Model bake-off: six local models on a 26-row tier-3 sample, adjudicated +1/0/-1 over the 17 rows where models disagreed.

## Result
| Item | Result |
|---|---|
| Cascade re-run, matched rows | 37,344 to 36,623 (-721) |
| Uncertain rows | 411 to 744 |
| Tier-3 calls / no-match | 3,251 / 1,061 to 6,279 / 4,162 |
| Bake-off | gemma-4-31b +9, qwen3.6-27b +7, nemotron-nano +6, gemma-4-e4b +6, medgemma-27b -1, llama-3.3-70b -1 |
| Cleanup pass at full scale | about 150 h estimated (2 calls x about 8 s x 36,623 rows); killed |

G+S effect of the cleaned labels was never measured.

## Why
- The patches route more ambiguous rows to tier 3 and reject them as no-match or uncertain instead of force-matching: slightly lower recall, expected higher precision.
- Headline match rate misleads: the two highest-match-rate models (medgemma 62%, llama 65%) hallucinated subtypes (first-plausible-candidate matching). The sample is only 26 rows and the adjudication was not a veterinary pathologist, so the +7/+6/+6 models are within noise.
- Cleanup needs trimming (subset of rows, one verifier, or batching) before it is tractable.

## Don't retry unless…
You trim the cleanup input or speed up verification first. Preferred verifier pair from the bake-off: gemma-4-31b and qwen3.6-27b; avoid medgemma and llama-3.3-70b for this task.

## Where it went
The patched cascade is the tier 1-3 cascade in `diagnosis_mapping/` (see [../../concepts/diagnosis-mapping.md](../../concepts/diagnosis-mapping.md)); the cleanup pass runs by default with the two-model pair `DEFAULT_CLEANUP_MODELS`. The annotation design itself was later replaced by the gold/silver plan ([annotation-redesign-plan](../plans/annotation-redesign-plan.md)).
