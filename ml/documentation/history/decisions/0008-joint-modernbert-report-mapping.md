# 0008 Report mapping is one jointly trained ModernBERT over the whole report

**Status:** accepted · **Date:** 2026-10-08 (Phase 2 test result; supersedes [0001](0001-concat-3-per-section-text-representation.md), [0002](0002-per-section-contrastive-backbone.md) and [0003](0003-four-stage-report-mapping.md))

## Context
The four-stage PetBERT pipeline (case-presence gate, group classifier, per-group label-presence heads, keyword correction) fed each head from frozen or adapted PetBERT embeddings of three merged sections, each cut to 512 tokens. Measured on a 3,000-report sample with the PetBERT tokenizer:
- The HISTOPATHOLOGICAL SUMMARY exceeds 512 tokens in 14.6% of reports (cut at inference) and 256 tokens in 46.5% (cut in backbone training).
- All long fields together: median 640 tokens, 99th percentile 2,176, maximum about 3,900.

Every stage read embeddings of those truncated sections, so the text the pipeline could see was set by the cut. The four-stage pipeline had been refined through the experiments indexed in [history/README.md](../README.md); its reference baseline for the comparison is `gen-20260927T025939Z`, called WP14 (Good 46.27%, Slight 15.86% on the `three-way-v1` test partition). The previous retrieve-and-rerank attempt ([2026-05-19](../experiments/2026-05-19-recsys-retrieve-and-rerank.md)) failed on retrieval recall (0.748 at K=20, 0.865 at K=50) and on a cross-case calibration cliff; its record said not to retry without a fix for both. The September 2026 experiment ([Phase 0](../experiments/2026-09-30-retrieve-rerank-v2-openjev.md), [Phases 1 and 2](../experiments/2026-10-08-joint-modernbert-gate-term.md)) supplied one.

## Decision
Report mapping is one model that reads the whole report once.

- **Input.** The 8 `report.csv` text columns in fixed order (CLINICAL ABSTRACT, GROSS DESCRIPTION, HISTOPATHOLOGICAL SUMMARY, COMMENT, FINAL COMMENT, ANCILLARY TESTS, ADDENDUM, ADDITIONAL INFORMATION), each non-empty one as `HEADING: text`, joined into one sequence of at most 4,096 ModernBERT tokens. Only 6 of the training reports (0.03%) and no calibration report exceed that length.
- **Model.** `thomas-sounack/BioClinical-ModernBERT-base`, mean-pooled, fine-tuned end to end with two fp32 heads trained jointly: a sigmoid head over the 455 diagnosis terms seen in train silver (scored on terms, not ICD codes, because Good is scored on the term) and a neoplasm gate head. The gate loss uses every labelled case; the term loss uses cancer cases only, with BCE.
- **Decision rule.** Gate probability below `gate_t` yields one `Non-Cancer` row (method `rejected_by_case_presence`). Otherwise the rank-1 term is emitted, plus the rank-2 term when its probability p2 is at least `tau` and p2/p1 is at least `rho`. The frozen values are `gate_t` 0.75, `tau` 0.10, `rho` 0.25.
- **Thresholds are fitted on the calibration partition only**, maximising G+S subject to a false-positive cap of 2.43% of scored rows (WP14's calibration FP share). The first freeze used a cap of 2.93% (WP14 plus 0.5 pp, gate_t 0.65); the user re-froze at 2.43% because the G+S curve is a plateau between gate_t 0.65 and 0.85 (69.23 to 69.09) while FP falls from 2.86% to 1.64%, so the tighter cap bought margin on the FP bar for a 0.05 pp G+S cost.
- **Bronze learns from silver.** Silver conventions, including the habit of choosing the "not otherwise specified" (NOS) term, are the target, not something to correct (user direction, 2026-09-30). A prediction that is "more specific than silver" counts as an error. Whether silver's NOS habit is clinically right is a question for the vet audit, not for model selection ([NOS demotion](../experiments/2026-05-29-nos-subtype-demotion.md) already showed rule-based correction harms).

### Rejected alternatives
- **Zero-shot LLM rerank (OpenJev, a community NLI-trained Qwen3.5 4B model run locally).** Failed its pre-registered ECE bar (raw 0.089, temperature-scaled 0.101, bar 0.05) and its verdicts were worse than WP14 on Good (30.0% against 48.1%). It picks the more specific sibling term where silver uses NOS, and it fixes 17 retriever errors while overriding 163 correct ones. The hosted TypeSafe Jev was excluded on privacy grounds (patient text must not leave the machine).
- **A separate neoplasm-gate model next to the retriever.** Worked (the standalone ModernBERT gate reached AUC 0.9906 against the old PetBERT gate's 0.9863), but doubled CPU cost. Joint training matches it within noise at half the cost.
- **A gate head trained on the frozen pooled vectors of a cancer-only retriever (shared gate).** Failed its adoption rule: AUC 0.9885, FP 3.75%, Good 40 fewer cases, FN 32 more. The retriever encoder was trained on cancer cases only, which is the likely reason (not tested separately).
- **Keeping the four-stage structure with a better backbone.** Not tested; the retriever with WP14's old gate already beat the old structure by +12.0 pp Good on calibration, so the extra stages were not needed to beat the old structure.

## Consequences
- Result, `three-way-v1` test partition, one-shot (4,028 rows against WP14's 4,299; share % of scored rows):

| | Good | Slight | CO | FP | FN | G+S |
|---|---|---|---|---|---|---|
| joint ModernBERT `rr-20261008T210937Z` | 58.47 | 10.60 | 10.92 | 2.14 | 17.87 | 69.07 |
| WP14 `gen-20260927T025939Z` | 46.27 | 15.86 | 12.84 | 2.35 | 22.68 | 62.13 |
| paired bootstrap, new - WP14 (95% CI) | +12.20 [+10.91, +13.60] | -5.26 [-6.39, -4.11] | -1.92 [-2.98, -0.91] | -0.21 [-0.82, +0.32] | -4.80 [-5.85, -3.83] | +6.94 [+5.53, +8.48] |

- Pre-registered bars all pass: Good lower bound +10.91 pp (bar at least -2 pp), G+S 69.07% (bar at least 62.13%), FP 2.14% (bar at most 3.35%, WP14 plus 1 pp). Test matches calibration (Good 58.47 against 59.99, FP 2.14 against 2.18), so there is no sign of threshold overfitting.
- **Slight falls 5.26 pp** while Good rises 12.20 pp, so G+S moves less than Good (+6.94 pp). Good and Slight must always be read separately. Discordant cases: 371 are Good only under the new model and 96 Good only under WP14. No per-group breakdown of the shift was made; a post-hoc per-group look at the test partition (diagnosis only, no tuning) was listed as an optional next step and has not been done.
- **Only 455 of the 844 taxonomy terms are predictable.** The head covers the terms seen in train silver; rarer terms cannot be emitted. Rare-term pairs are about 6% of gold. The accepted-but-unconfident case is what the review cut catches ([0009](0009-bronze-restructure-and-lifecycle.md)). Ideas to reach the rest are parked in [open-ideas.md](../open-ideas.md).
- **The test partition is spent for this design.** It was scored once; never tune on it. Later acceptance uses gold-eval and calibration.
- **Cost.** One encoder pass per report. CPU fp32 (8 threads, 50 calibration reports, mean 487 tokens): 0.284 s per report for the joint model against 0.571 s for two models. Training takes about 56 minutes on one RTX 5070 Ti (peak 4.15 GiB) with gradient checkpointing and token-budget batches.
- **Library pins move.** The experiment ran on transformers 5.18.0 and the generation is trained and served on it (the old tree pinned 4.46.3); see [0009](0009-bronze-restructure-and-lifecycle.md).
- The embedding cache and fingerprint of [0004](0004-generations-candidate-current-archive.md) have no counterpart: there are no separate embeddings to go stale.

## Evidence
[Phase 0 and the OpenJev pilot](../experiments/2026-09-30-retrieve-rerank-v2-openjev.md); [gate bake-off, seed check, shared and joint models, freeze, and the Phase 2 test](../experiments/2026-10-08-joint-modernbert-gate-term.md); [restructure plan](../plans/restructure-plan.md). Current description: [../../concepts/report-mapping.md](../../concepts/report-mapping.md).
