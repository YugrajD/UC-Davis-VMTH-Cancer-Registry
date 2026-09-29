# ml-merge dual backbone: broad-versus-picker diagnostic

**Date:** 2026-05-24 · **Verdict:** inconclusive (never adopted; the merge tree was deleted 2026-09-14)

## Hypothesis
An experimental merge tree ("ml-merge") combined `ml/`'s group classifier with a second backbone from a colleague's pipeline whose presence-style classifier picked candidate groups. It beat `ml/` on G+S. Was that win only an operating-point artefact that `ml/` could reproduce by running its own group head "broad" (low threshold, many groups per case)?

## Setup
- Sweep of `ml/` production at group thresholds 0.1 and 0.3, tail cap 5/20/50, gap cull off and a global LP threshold, scored per prediction on the test split.
- Compared at operating points matched to the merge tree's G+S frontier.

## Result
| Item | Value |
|---|---|
| `ml/` canonical anchor reproduced | G+S 62.3% against 62.2% recorded |
| `ml/`-broad G+S shortfall at matched points | 27-54 pp |
| CO (Off) when broad | 69-85% (merge tree: 25-30%) |
| Group-rows per case at threshold 0.1 | about 13 (about 12 completely off) |

The merge tree's own Good and Slight numbers are not recorded in the sources I could read; only the shortfall of the broad `ml/` run is.

## Why
`ml/`'s 25-way group head is a sharp sigmoid trained around 0.85. Forced to threshold 0.1 it emits nearly every group, so the extra recall is wrong-group noise. The other pipeline's picker forwarded clean candidate groups even when broad. So the merge tree's advantage was group-picker quality, not an operating-point artefact.

## Don't retry unless…
You retrain a presence-style (text, label) picker on the concat-3 backbone; that is a research project, not a free simplification. Driving the merge tree's pass A from `ml/`'s group head run broad is proven dead.

## Where it went
Nothing merged. The merge tree and its write-up were deleted with the other `ml-*` siblings on 2026-09-14 (only `ml-worker` survives).
