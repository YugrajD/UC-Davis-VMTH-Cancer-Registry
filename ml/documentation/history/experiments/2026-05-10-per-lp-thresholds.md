# Per-LP threshold calibration

**Date:** 2026-05-10 · **Verdict:** adopted

## Hypothesis
One global LabelPresence threshold of 0.5 is wrong by construction: per-head precision spans 0.18-0.96 and 13 of 17 heads were overconfident (macro precision 0.537, recall 0.918). A per-group threshold fitted on held-out data should cut over-firing, the main source of "slightly off".

## Setup
- Baseline: global LP threshold 0.5 (Phase 30 revert baseline, test n=16,902 rows).
- Test cases split 50/50 by md5 of `case_id`; thresholds swept per head on the sweep half (0.05 to 0.95) and scored on the eval half.

## Result
| Measure | Before | After |
|---|---|---|
| Top-1 exact-term Good, eval half (2,683 cancer cases) | 46.7% | 55.4% (+8.7 pp) |
| Spurious labels | baseline | -53% |
| Macro F1 / micro F1 (eval half) | n/a | +0.082 / +0.19 |
| Full-test pipeline, Good | 25.7% | 36.1% (+10.4 pp) |
| Full-test pipeline, Slight | 33.8% | 19.6% |
| Full-test pipeline, G+S | 59.5% | 55.7% (-3.8 pp) |
| Full-test pipeline, FN | 9.2% | 15.4% (+6.2 pp) |

Pipeline rows: n=16,902 before, n=11,138 after. Thirteen of 17 heads landed at t >= 0.85; Mast cell (0.69) and Histiocytes (0.49) were the well-calibrated outliers. On the later concat-3 stack, 22 of 25 heads benefited and thresholds mostly landed at 0.85-0.95 (eval-half macro F1 0.6522 to 0.7342).

## Why
Stricter thresholds convert Slight into Good but drop borderline-correct predictions into FN, so combined G+S fell while Good rose sharply. It was adopted because precision on what is predicted matters more than coverage for coding. Note the two sources: the old accepted-ideas list quoted only the +8.7 pp Good figure; project notes also recorded the G+S drop.

Bottleneck snapshot that motivated it (17-group TF-IDF stack, test set 16,902 rows): Good 25.7%, Slight 33.8%, CO 23.4%, FP 8.0%, FN 9.2%. Stage 1 gate was saturated (P 0.91, R 0.95, F1 0.93, AUC 0.98); the group head had macro F1 0.78, top-1 82.7% and top-3 97.5%; per-group LP heads had validation F1 0.76-0.96 but production precision only 0.29-0.61 (Osseous 0.30, Gliomas 0.18, Uncommon 0.11). Adenomas and adenocarcinomas was the biggest drag (17.6% of rows, 18.1% Good). Stage 3 over-firing was the main source of Slight, so it was the first place to calibrate.

## Don't retry unless…
Not applicable (adopted). Thresholds must be refit after any head retrain.

## Where it went
Replaced by `calibrate.py` (`report_mapping/training/calibrate.py`), which fits one threshold per LP on a calibration partition (grid 0.05-0.95 step 0.05, F1 objective) into `checkpoints/label_presence/lp_thresholds.json`; the old sweep script is gone. Production `current/` (`gen-0-legacy`) still carries the hand-picked legacy values. See [../../concepts/report-mapping.md](../../concepts/report-mapping.md).
