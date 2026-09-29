# 0005 Rewrite `ml/` cleanly, prove parity against a frozen baseline, then cut over

**Status:** accepted · **Date:** approved 2026-09-25, cut over 2026-09-27 UTC (commit `46708f6` is stamped 2026-09-26 19:42 local)

## Context
The old tree kept production hyperparameters only in documentation commands, seeded inconsistently, keyed its embedding cache on file modified time, drew the "Uncommon" label order from a Python `frozenset` (so about 280-330 of 68,800 rows changed with the hash seed), and carried many dead experiment flags. The project also needed the gold/silver/bronze design ([icd-mapping-strategy](../../concepts/icd-mapping-strategy.md)). The published baseline of 62.1% G+S did not reproduce from files on disk: the gate, group and some label-presence heads had been retrained and the annotation file rewritten since it was measured (3,153 of 67,908 aligned rows differed from the 2026-05-29 production file).

## Decision
1. Write the new tree as a clean slate in `ml/next/` under a plan with numbered work packages, then move it into `ml/` and delete the old tree, parity harness and one-time importers.
2. Regenerate a frozen reference with the old inference code on 2026-09-25 (`PYTHONHASHSEED=0`) and treat **61.76%** (not 62.1%) as the baseline: eval half of the legacy split, G+S 61.8%, exact 61.76% on 4,456 per-code rows (Good 45.8, Slight 16.0, CO 15.3, FP 2.6, FN 20.4); full legacy test 61.8% on 8,916 rows (Good 46.4, Slight 15.4, CO 15.0, FP 2.6, FN 20.6). The reference is somewhat optimistic: the tail gate and 0.80 gate were chosen on all of test.
3. Gate the cutover on parity levels L1-L3; L4 is report-only.

## Consequences
- Parity results before cutover: L1 pass (identical verdicts); L2a and L2b pass (58,313 cases re-embedded, minimum cosine 1.000000, 68,800 of 68,800 rows identical, 199 Uncommon-reordered); L3 pass, three seeds mean G+S 61.44% against 61.76% (-0.32 pp, tolerance +/-1.93; Good -0.45, Slight +0.12, CO +0.11, FP +0.83, FN -0.62 pp); L4 cold start (backbone 3 epochs on 56,109 section pairs, 27 min) 62.31% (+0.55 pp; Good +0.30, Slight +0.25, CO -0.50, FP +0.30, FN -0.35 pp), report only. Odontogenic and Transitional cell groups lost more than 5 pp in both L3 and L4 (small groups: 5 pp is about 3 codes).
- The 62.1% to 61.76% difference is a change of reference, not a regression: the old figure came from a superseded generation.
- Production `current/` is `gen-0-legacy` (legacy checkpoints imported once, legacy hand-picked thresholds), not a refit generation. A refit ran on the new split (WP14; fork onto `three-way-v1`, full refit, test G+S 62.13%) and is parked in the archive, waiting for real gold and a trigger.
- WP14 and WP15 gold steps ran on **mock gold** that lives only in a scratch folder and in test code. Nothing from them is a result. The archive folder named `MOCK-GOLD-candidate-do-not-promote` must never be promoted.
- The old tree's data was archived locally under `ml/output/archive/2026-09-27_legacy-tree/`; the two change requests written for the backend were implemented afterwards ([ml-worker](../plans/ml-worker-change-request.md), [audit list](../plans/audit-list-change-request.md)).
- Line endings are pinned to LF in every writer so Windows and Mac hashes match.

## Evidence
[plan and status](../plans/ml-rewrite-plan.md), [concat-3](../experiments/2026-05-12-concat-3-text-representation.md) (where 62.1% came from), [generations decision](0004-generations-candidate-current-archive.md). Current description: [../../concepts/report-mapping.md](../../concepts/report-mapping.md) and [../../concepts/evaluation.md](../../concepts/evaluation.md).
