# Single all-label PresenceClassifier with CO-bank cycles (Phases 1-22)

**Date:** 2026-03-03 to 2026-03-28 · **Verdict:** superseded by [three-stage gate, group and keyword correction](2026-04-29-three-stage-gate-group-keyword.md)

## Hypothesis
One binary MLP scoring every (report, label) pair, retrained in cycles on a growing "completely off" (CO) negative bank, can replace raw cosine similarity between PetBERT embeddings.

## Setup
- Baseline: cosine similarity, frozen PetBERT, keyword-matched labels as ground truth (about 1.3k labelled cases at first, 5,788 by Phase 11).
- Classifier input `[report_emb | label_emb]`; each cycle mined new CO and false-positive pairs from its own predictions.
- Metrics below come from the old per-label, top-k evaluator against keyword ground truth. They are not comparable with any number after May 2026.

## Result
| Phase | Change | G+S | CO | FN |
|---|---|---|---|---|
| none | cosine only | 3.3% | 42.6% | n/a |
| 1 | first classifier | 7.5% | 44.7% | 6.9% |
| 3-5 | CO negatives, then a rolling bank (co=10) | 22.9% | 42.5% | 3.7% |
| 9 | recall weight 0.25 ended the cycle-to-cycle oscillation | 20.4% | 42.7% | 4.2% |
| 11 | keyword data grew to 5,788 confirmed cases | 33.1% | 31.8% | 1.3% |
| 12 | cold start on XPU, same data | 32.0% | 32.1% | 1.3% |

Good and Slight were not split out in the phase summary table; see `history/training-log/training-log-binary.md` for per-run columns.

## Why
- Raw cosine had two labels with mean similarity 0.85-0.91 to every report, so argmax picked them for nearly everything. Per-label mean subtraction and CO-bank negatives were the first steps that worked.
- The biggest single driver was more labelled data (1,273 cases gave about 20% G+S; 5,788 cases about 33%).
- A completely-off floor of about 30% remained: scoring labels pairwise gives no global view of which group a report belongs to. The GroupClassifier later addressed that.

## Origins (March planning notes)
- Ground truth was about 1,709 keyword-matched rows across about 1,273 of about 2,783 cases; every other case was treated as non-cancer. The verdict scheme (Good, Slightly off, Completely off, False positive, later False negative) was defined in this era and is what `evaluation/verdicts.py` scores today.
- Cosine baseline on 13,855 prediction rows: Good 0.1%, Slight 3.2%, CO 42.6%, FP 54.1%. False positives scored 0.84-0.94 cosine (mean 0.90), higher than many correct predictions, so no threshold could separate them; that finding motivated a learned presence head instead of a better cut-off. Full fine-tuning was rejected at the time as too risky for about 1.3k cases.
- Early plans also proposed multi-diagnosis output (up to 5 ranked predictions per case, `diagnosis_index` as rank); a package refactor in March renamed the scan packages and split training into `binary/`, `group/` and `finetune/`. Both are gone with the old tree.

## Don't retry unless…
Not applicable; the whole approach was replaced.

## Where it went
`--mode train-classifier`, `training/binary/{build_training_pairs,run_cycle,update_co_bank,train}.py` and `model/presence_classifier.py` were deleted in Phase 28 (May 2026). The tree they lived in is archived under `ml/output/archive/2026-09-27_legacy-tree/` (local, untracked). Related records: [label enrichment](2026-03-04-label-embedding-enrichment.md), [per-column embeddings](2026-03-05-per-column-embeddings-and-wider-hidden-layer.md).
