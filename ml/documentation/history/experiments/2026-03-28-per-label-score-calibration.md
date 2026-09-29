# Per-label score calibration (offsets)

**Date:** 2026-03-28 · **Verdict:** reverted

## Hypothesis
After mean-centering, labels still differ in score variance, so a label the model is unsure about loses the argmax to noisier labels. A per-label offset `b_l` (grid searched, labels with at least 10 cases) should fix that.

## Setup
- Baseline: the pre-calibration checkpoint in `training-log/training-log-finetune.md` scored 69.4% G+S (Good 22.1%, Slight 47.3%, Off 6.1%). The older ideas-rejected note called the baseline "Phase 18 best, 70.4%"; the two sources conflict, and the deltas below (-9.7 and -5.5 pp) only match 69.4%. Old evaluator.
- Three objectives tried for the offsets: exact match, group-level per label, and net gain across all annotated cases.

## Result
| Objective | Good | Slight | G+S | Off | Delta G+S |
|---|---|---|---|---|---|
| Pre-calibration | 22.1% | 47.3% | 69.4% | 6.1% | n/a |
| Exact-match (v1) | 25.5% | 34.2% | 59.7% | 11.9% | -9.7 pp |
| Group-level per label (v2) | n/a | n/a | harmful in-sample (93.7% to 89.2% on the annotation set), discarded | n/a | n/a |
| Net gain (v3) | 22.6% | 41.3% | 63.9% | 6.9% | -5.5 pp (about 5k fewer predictions) |

FP and FN are not tabulated in the source summary.

## Why
Offsets stole wins across groups and interacted badly with the `embedding_min_sim` threshold. Variance bias was not the binding constraint; the original diagnosis was a data ceiling.

## Don't retry unless…
The corpus has grown a lot and you recalibrate against a fresh production checkpoint. The later per-LP and gate thresholds ([per-LP thresholds](2026-05-10-per-lp-thresholds.md)) are the version of "calibrate scores" that did work, because they are fit per head on held-out data rather than per label.

## Where it went
`ml/training/binary/calibrate.py` and `--calibration-offsets` were removed with the old tree; the offsets file was empty (`{}`).
