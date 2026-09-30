# Slight-bank LabelPresence heads stacked with a CO-bank GLP

**Date:** 2026-05-27 · **Verdict:** inconclusive (measured additive gain, never migrated; the experimental tree was deleted 2026-09-14)

## Hypothesis
Two independent fixes should add up: (a) retrain the three highest-Slight LP heads (Adenomas and adenocarcinomas, Adnexal and skin appendage, Osseous and chondromatous) with model-confused ("empirical") hard negatives and a lower threshold (0.30), and (b) route groups with a BCE-cycle-3 CO-bank GLP (`groupagg`, top-1).

## Setup
- Experimental clone of `ml/`, held-out test half (n=11,661 cases). The other 22 LPs used canonical weights and thresholds.
- Baseline: canonical 4-stage pipeline (2026-05-25).

## Result
| Config | n | Good | Slight | G+S | Off | FP | FN |
|---|---|---|---|---|---|---|---|
| Canonical | 8,818 | 45.7% | 14.4% | 60.1% | 15.0% | 2.3% | 22.6% |
| Slight-bank LPs alone | 9,268 | 44.3% | 16.5% | 60.8% | 15.5% | 2.4% | 21.3% |
| CO-bank GLP alone | 8,136 | 47.1% | 14.2% | 61.3% | 8.3% | 2.2% | 28.3% |
| Stacked (t=0.30) | 8,387 | 45.9% | 16.4% | 62.3% | 8.3% | 2.3% | 27.2% |

Re-sweeping the three pilot thresholds with the GLP in the loop went monotone down to t=0.01, where G+S read 72.0% but Good fell to 27.2%, Slight rose to 44.8% and rows grew 69%. That is threshold-driven multi-label expansion, not an improvement, so t=0.30 stayed.

## Why
- The GLP fixed group routing (Off 15% to 8.3%); the Slight-bank heads added within-group recall (Slight +2.2, FN -1.1 against the GLP alone). Effects were additive within about 0.3 pp.
- Caveats: 86% of the standalone Slight-bank gain came from emitting more predictions, not from the intended FN rescue (13.7%); Adenomas regressed -1.8 pp G+S and should be excluded from further Slight-bank work; the stacked gain over canonical (+2.2 pp) is partly row expansion. An earlier "+11.1 pp" figure in the source report compared GLP against GLP, not against canonical.

## Don't retry unless…
You migrate the pieces into the current tree (they exist nowhere in `ml/` now, so this is a reimplementation) and judge on Good and Slight separately, not combined G+S.

## Where it went
Not migrated. Checkpoints and launcher lived in the deleted experimental tree. Recorded so the +2.2 pp additive result is not lost. Related: [GLP experiments](2026-05-26-glp-listwise-and-per-group-thresholds.md).
