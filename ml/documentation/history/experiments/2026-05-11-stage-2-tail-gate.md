# Stage-2 tail gate

**Date:** 2026-05-11 · **Verdict:** adopted

## Hypothesis
After the argmax fallback, stage 2 could still emit several below-threshold groups per case, and tail groups far below the top group inflated CO because stage 3 still ran on them. Capping predictions per case and dropping tail groups that trail the top group by more than a gap should convert CO into FN at a good rate.

## Setup
- Baseline: per-LP thresholds, group threshold 0.85, no tail gate (eval-history row 40, n=11,138 rows).
- Sweep over cap K and gap on the test set (no retrain).

## Result
| Config | Good | Slight | G+S | CO | FP | FN | n |
|---|---|---|---|---|---|---|---|
| No gate | 36.1% | 19.6% | 55.7% | 21.4% | 7.5% | 15.4% | 11,138 |
| K=2, gap 0.08 | 37.3% | 19.3% | 56.6% | 16.8% | 7.8% | 18.7% | 10,334 |

Net: G+S +0.9 pp, CO -4.6 pp, FN +3.3 pp, rows -7%.

## Why
The curve is unimodal with a flat peak at gap 0.06-0.09 (about 56.6-56.7% G+S); 0.08 was picked on the plateau. K=3 or more is identical to K=2 once the gap is set. The gate trades absolute coverage for per-row precision: the no-gate baseline produces more correct rows in absolute terms (4,023 against 3,855 Good). The optimal gap depends on stage-2 calibration, so it must be re-picked after any group retrain.

## Don't retry unless…
Not applicable (adopted).

## Where it went
Kept as `tail_max_predictions` and `tail_max_group_prob_gap` in each generation's `thresholds.json`, applied in `report_mapping/inference/stages.py`, and now fitted jointly with the gate and group thresholds by `calibrate.py` (`TAIL_GRID`). Legacy values in production: K=2, gap 0.08. The standalone sweep script is gone. See [../../concepts/report-mapping.md](../../concepts/report-mapping.md).
