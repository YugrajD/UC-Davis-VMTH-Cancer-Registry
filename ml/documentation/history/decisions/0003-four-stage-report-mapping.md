# 0003 Four stages: case-presence gate, group classifier, per-group label presence, keyword correction

**Status:** accepted · **Date:** 2026-04-29 (three stages), 2026-05-07 (per-group heads added)

## Context
One classifier scoring every (report, label) pair emitted false positives on non-cancer reports and had a hard completely-off floor of about 30%, because it never reasoned about which group a report belongs to. Attempts to fix it with KNN group gating, richer label embeddings or per-pair architectures failed.

## Decision
Give each stage one job:
1. A case-level gate rejects non-cancer reports.
2. A group classifier (one sigmoid over about 25 groups, competing in one loss) picks the ICD group, with an argmax fallback and a tail gate that caps and trims extra groups.
3. One label-presence head per group picks the term inside the group, with a per-head threshold.
4. Keyword correction narrows by ICD-O behavior digit and group-specific subtype rules (a pure rule filter, no training).

Thresholds are calibrated, not hand-set: `calibrate.py` fits them on a calibration partition.

## Consequences
- Errors are attributable to a stage (FP to the gate, CO to the group head, Slight to stage 3), which is how the later experiments were chosen.
- Stage interactions matter: a recall shift from a new backbone or hard-negative loss did not compose with the gate structure ([backbone hard-neg](../experiments/2026-03-27-backbone-hard-neg-margin.md)); changing group thresholds shifts CO/FN trade-offs; thresholds must be refit after any head retrain.
- Rare groups (fewer than 200 cases) merge into an "Uncommon" head; promoting them individually failed.

## Evidence
[three-stage pipeline](../experiments/2026-04-29-three-stage-gate-group-keyword.md), [per-group label presence](../experiments/2026-05-07-per-group-label-presence.md), [tail gate](../experiments/2026-05-11-stage-2-tail-gate.md), [per-LP thresholds](../experiments/2026-05-10-per-lp-thresholds.md), [2304-dim gate](../experiments/2026-05-12-case-presence-gate-2304.md), rejected alternatives ([KNN gating](../experiments/2026-03-23-knn-group-gating.md), [retrieve-and-rerank](../experiments/2026-05-19-recsys-retrieve-and-rerank.md), [27 groups](../experiments/2026-05-07-uncommon-threshold-and-27-groups.md)). Current description: [../../concepts/report-mapping.md](../../concepts/report-mapping.md).
