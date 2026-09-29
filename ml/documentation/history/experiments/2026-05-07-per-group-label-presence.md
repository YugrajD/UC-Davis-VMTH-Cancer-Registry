# Per-group LabelPresenceClassifier (stage 3a, Phase 28)

**Date:** 2026-05-07 · **Verdict:** adopted

## Hypothesis
Cosine similarity or keyword rules pick the term inside the predicted group poorly (most errors were "slightly off": right group, wrong term). One binary MLP per group, scoring `[report_emb | label_emb]` against the other labels in the same group, should convert Slight into Good with a sharper boundary than one global classifier.

## Setup
- Baseline: Phase 27 three-stage pipeline (group threshold 0.85), old evaluator, test set.
- 25 per-group heads on the TF-IDF backbone, positives from annotation, 5 in-group negatives per positive, recall weight 0.5, label-presence threshold 0.5.

## Result
| Config | Good | Slight | G+S | CO | FP | FN | Rows |
|---|---|---|---|---|---|---|---|
| Phase 27 baseline | 12.9% | 42.4% | 55.3% | 21.6% | 5.0% | 18.2% | 9,127 |
| Phase 28, LP threshold 0.5 | 28.8% | 29.1% | 57.9% | 25.3% | 5.7% | 11.1% | 15,100 |
| Phase 28, LP threshold 0.9 | 38.3% | 17.6% | 55.9% | 22.9% | 5.4% | 15.8% | 10,923 |

Mast cell reached 90% Good (393 of 436); weakest were rare groups (Ductal/lobular 1%, Acinar cell 2%).

## Why
Stage 3a did what it was for: Good +15.9 pp, Slight -13.3 pp at threshold 0.5. CO and row count rose because low thresholds emit several labels per group, and every extra row is CO when the group is wrong. No threshold beat Phase 27 on both G+S and CO. After the phase-29 backbone cold start the operational baseline became 59.5% G+S (Good 25.7%, Slight 33.8%, CO 23.4%, FP 8.0%, FN 9.2%; test n=16,902 rows), which is a different setup from the 57.9% row.

## Don't retry unless…
Not applicable (adopted). Attempts to sharpen these heads with hard negatives failed twice: [QW1/QW2](2026-05-10-lp-hard-negatives-qw1-qw2.md) and [the retry](2026-05-13-lp-hard-neg-retry-and-top3-rerank.md).

## Where it went
Kept as stage 3 of report mapping (`report_mapping/model/heads.py`, `training/label_presence.py`), with `n_cols=3, col_pair_mode=True, col_combine="learned"` since [concat-3](2026-05-12-concat-3-text-representation.md). One `.pt` per group under `checkpoints/label_presence/` in each generation. Decision: [0003](../decisions/0003-four-stage-report-mapping.md).
