# Three-stage pipeline: case-presence gate, GroupClassifier, keyword correction (Phases 23-25)

**Date:** 2026-04-28 to 2026-05-02 · **Verdict:** adopted (became the four-stage pipeline once [per-group LabelPresence](2026-05-07-per-group-label-presence.md) was added)

## Hypothesis
The single all-label classifier ([Phases 1-22](2026-03-03-single-presence-classifier-co-bank.md)) emitted too many false positives and had a hard CO floor. Give each stage one job: a case-level gate drops non-cancer reports, a group classifier picks the ICD group with groups competing in one sigmoid loss, and keyword rules pick the term inside the group.

## Setup
- Data: 46,652 train cases (21,853 cancer / 24,799 non-cancer) with LLM-cascade annotation; test split 11,661 cases.
- Run 8 (2026-04-28): GroupClassifier on the round-2 backbone. Uncapped BCE `pos_weight` reached 3,587x and the model predicted every group for every case; `--max-class-weight 50` and weight decay 1e-3 fixed it (macro F1 0.1922).
- Run 10 (2026-04-29): add `CasePresenceClassifier` gate (768-dim, recall weight 0.7). Run 11 (2026-05-01, "Phase 25"): retrain the gate with recall weight 0.85.

## Result
| Config (test set) | Good | Slight | G+S | CO | FP | FN | Rows |
|---|---|---|---|---|---|---|---|
| Run 8, GroupCLF without gate | 9.1% | 24.9% | 34.0% | 35.8% | 27.9% | 2.2% | 12,330 |
| Run 10, gate rw 0.7 | 9.1% | 35.7% | 44.8% | 36.8% | 10.9% | 7.6% | 6,858 |
| Phase 24, gate rw 0.7, gate 0.5, group 0.90 | n/a | n/a | 49.1% | 22.1% | 3.7% | 25.0% | 6,748 |
| Phase 25, gate rw 0.85, gate 0.5, group 0.90 | n/a | n/a | 62.6% | 26.2% | 6.7% | 4.5% | 7,084 |
| Phase 25 re-scored with the per-label evaluator (2026-05-02) | n/a | n/a | 51.8% | 19.3% | 4.7% | 24.1% | 8,744 |

Sources conflict on the Phase 25 headline: the 62.6% figure comes from the old evaluator; the evaluator was changed on 2026-05-02 to count each uncovered gold label as its own FN row, and the same pipeline then scores 51.8%. Both are recorded here; neither is comparable to the 2026-05-13 numbers below.

## Why
- The gate removed most false positives (FP 27.9% to 10.9%) and is independently tunable.
- Recall weight matters: 0.7 made the gate too strict (FN 25%); 0.85 gave FN 4.5% in that setup. See the unresolved recall-weight note in the [2304-dim gate record](2026-05-12-case-presence-gate-2304.md).
- Uncapped class weights are unusable for rare groups; `max_class_weight=50` and `weight_decay=1e-3` remain in the recipe.

## Don't retry unless…
Not applicable (adopted).

## Where it went
Kept as stages 1, 2 and 4 of report mapping (`report_mapping/model/heads.py`, `inference/stages.py`). See [../../concepts/report-mapping.md](../../concepts/report-mapping.md) and decision [0003](../decisions/0003-four-stage-report-mapping.md).
