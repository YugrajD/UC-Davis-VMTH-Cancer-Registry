# 2304-dim case-presence gate

**Date:** 2026-05-12 · **Verdict:** adopted

## Hypothesis
Under concat-3 the natural stage-1 input is the 2304-dim per-case vector, the same view the downstream heads see, so the gate should be retrained on it rather than on the 768-dim mean.

## Setup
- Baseline: per-section backbone + concat-3, no gate (G+S 24.0%, FP 53.9%) in the three-stage prototype tree (experimental sibling tree, since deleted).
- Gate: `CasePresenceClassifier(emb_dim=2304)`, 46,652 train cases (21,411 cancer / 25,241 non-cancer), case-disjoint validation split, 20 epochs, best at epoch 14. Threshold swept 0.30-0.95.

## Result
| Config | Good | Slight | G+S | CO | FP | FN | n |
|---|---|---|---|---|---|---|---|
| No gate | 6.2% | 17.8% | 24.0% | 20.2% | 53.9% | 1.9% | 57,842 |
| Gate at 0.5 | 12.9% | 36.9% | 49.8% | 39.8% | 5.5% | 4.9% | 26,967 |

Threshold sweep (G+S only): 48.3% at 0.30, 49.8% at 0.5, 51.3% at 0.85, 51.5% at 0.95; 0.85 chosen by G+S/FN ratio (FN climbs slowly until 0.85, then quickly). A CO-bank cycle at 0.85 added +0.5 pp (51.3% to 51.8%), so one cycle was judged enough. Validation F1 0.942 (P 0.937, R 0.947). In the promoted four-stage stack the gate at 0.85 is part of the 62.1% result; it also raised FN about +2.1 pp against the older 0.5 gate.

A wider head (`hidden_dim` 2048 against 512) was tested on the same stack and was negative: G+S 51.8% to 51.1%, Good +0.2, Slight -0.9. Classifier capacity is not the bottleneck.

## Why
The gate removed about 29.7k false positives for about 99 lost Good rows. The ungated numbers are dominated by FP, so the 24.0% to 49.8% jump mostly reflects FP dropping from 53.9% to 5.5%.

## Unresolved: recall_weight (0.85 vs 0.7)
Three sources disagree and none settles it.
- The old `training-ideas/ideas-accepted.md` (deleted; kept in git tag `docs-pre-overhaul-2026-09-29`) says the gate "was trained with `recall_weight=0.85`" and that 0.7 gave FN about 25%. That matches the Phase 25 *768-dim* gate (validation score 0.939 at 0.85; see [three-stage record](2026-04-29-three-stage-gate-group-keyword.md)).
- The 2304-dim gate itself was trained at recall weight 0.7 (validation score 0.944), per the project notes and the promotion write-up.
- Code (`ml/report_mapping/training/recipe.py`, `GATE.recall_weight = 0.7`) cites the legacy training guide Step 5 command with `--case-presence-recall-weight 0.7`.

The likely reading is that 0.85 was the 768-dim gate and 0.7 the concat-3 gate, but nothing in the repo records which value trained the checkpoint now in `gen-0-legacy` (its manifest recipe is only labelled "legacy"). Status: unresolved; the code value 0.7 is what a retrain would use.

Also note: the May 2026 docs quote a gate operating point of 0.85; the legacy thresholds frozen into `gen-0-legacy` at cutover use 0.80.

## Don't retry unless…
Not applicable (adopted). More gate width or CO-bank cycles showed no further gain.

## Where it went
Kept as stage 1 (`report_mapping/training/case_presence.py`, recipe `GATE`). Its threshold is now refit by `calibrate.py` over `GATE_GRID`. Decision: [0003](../decisions/0003-four-stage-report-mapping.md). The stack this belongs to: [concat-3](2026-05-12-concat-3-text-representation.md).
