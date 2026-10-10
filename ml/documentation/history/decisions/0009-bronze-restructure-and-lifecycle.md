# 0009 Bronze restructure: one-model lifecycle, review rule and audit

**Status:** accepted · **Date:** 2026-10-09 · amends [0004](0004-generations-candidate-current-archive.md) (embedding fingerprint and cache parts)

This record is the `--reason` for the first architectural adoption (`promote.py --adopt --reason ml/documentation/history/decisions/0009-bronze-restructure-and-lifecycle.md`).

## Context
[0008](0008-joint-modernbert-report-mapping.md) replaces the PetBERT four-stage pipeline with one jointly trained ModernBERT. Everything built around the old model assumed several heads on shared embeddings: the embedding cache, the embedding fingerprint, three checkpoint families per generation, the review rule (confidence and top1-top2 margin constants tuned for the old scores), the gate-only Report-Mapping audit, and a worker that read a PetBERT bundle. The user's direction was a complete overhaul of the process, not a promotion of a new model into the old structure. The binding decisions and work packages are in the [restructure plan](../plans/restructure-plan.md).

## Decision

### Two kinds of improvement
- **Retraining** (same code, new weights): `promote.py` judges the candidate on **gold only**: the paired case-cluster bootstrap on gold-eval must show a Good lower bound of at least -2 pp and at least one retraining trigger must be met. There is no silver fallback; without gold-eval rows it refuses. `retrain_cycle.py --apply` is opt-in and promotes automatically only when that rule passes; without the flag the cycle recommends.
- **Architectural change** (new code): adopted manually with `promote.py --adopt --reason <path>`. The path must point at an existing file under `ml/documentation/history/`. The reason is written into the manifest, the whole of `current/` is archived to `ml/output/archive/YYYY-MM-DD_<description>/`, and the candidate is swapped in. `promote.py` refuses to promote a candidate whose `architecture` block differs from `current/`'s and points to adopt. This restructure is the first adoption, accepted on the Phase 2 result of 0008 plus the reproduction checks below, not on parity with the old pipeline (the [0005](0005-ml-rewrite-and-cutover.md) precedent).

### Generation
- A generation is one directory: `encoder/` (ModernBERT weights, tokenizer, config), `head.pt`, `gate_head.pt`, `terms.json`, `term_info.json` (term to group and code, snapshot of `ml/taxonomy/labels.csv`), `thresholds.json`, `manifest.json`.
- A **spec check** replaces the embedding fingerprint. Every loader, promote, export and S3 publish verifies file hashes, that the manifest's `architecture` block (which carries the text-spec version) equals the code's, that `weights_sha256` matches the weight files, that `n_terms` equals the size of `terms.json` (the head's width is checked again when the model loads), that calibration is `calibrated`, and that the calibration is bound to the generation's `weights_sha256`. The embedding cache is removed: there are no embeddings to cache.
- Every retrain starts from the base checkpoint with a fresh head over the train partition's terms; there is no warm start. Thresholds are refitted after every training run.
- **Recalibrating** (new thresholds, same weights) creates a new generation id (`generations.py recalibrate`); it replaces the old fork.
- The FP cap is a named config constant (`FP_CAP`, 0.0243), with `--max-fp-share` as the override. The uncommon-groups list is a scoring-rule file (24 groups, WP14's list, pinned in `config.UNCOMMON_GROUPS_TXT`), not part of a generation.
- **The first generation is the frozen weights of `rr-20261008T210937Z`, re-housed, not retrained.** They were trained on raw silver `silver-0-legacy` (vague cases included). Retrains use the corrected-annotations table, so the next retrain will differ in its training data as well as in its random state.

### Review rule (bronze-only cases)
A case is queued for review when the gate rejected it, or when it was accepted and its rank-1 confidence is below `review_cut`. `review_cut` is fitted per generation at calibration time, as the cut that queues 23.36% of accepted calibration cases (`REVIEW_TARGET_SHARE`, the volume the old rule queued: 592 of 2,534). For the first generation `review_cut` is 0.85, which queues 23.23% of 2,596 accepted calibration cases. The margin rule and `unidentified_cancer` are removed. Queue priority stays the gate probability. This is the safety net for the model's blind spot (only 455 of 844 terms are predictable): a case the gate accepts but where no term is confident is queued.

### Report-Mapping audit
The audit is a 5-fold cross-fit over the train partition (`training/crossfit.py`, `scripts/oof.py`): five models are trained, each scoring the fold it did not see, and the output (numbers only) records the gate probability and the term probabilities of every labelled term. The sampler contradicts silver at the case level (a cancer case with a low gate probability, a non-cancer case with a high one) and at the **term level** (a cancer case with a gate probability of at least 0.8 where every labelled term scores below 0.2). It is run once per silver lineage or retrain (estimated at 4 to 5 hours of GPU; not run at the time of writing). See [manual-audit.md](../../concepts/manual-audit.md).

### Other contracts
- **Has-text rule.** A case has text if any of the 8 report columns is non-empty (the PetBERT rule needed one of three sections). On the frozen predictions this adds 141 cases (109 train, 17 calibration, 15 test).
- **Upload text.** The worker scores the `Text` field of an upload as flat text without a heading. Measured on calibration against the full headed 8-column text, flat text costs: Good -0.10, Slight -0.01, CO +0.43, FP +0.50 [+0.20, +0.85], FN -0.82, G+S -0.11 pp (one heading: FP +0.64).
- **Scope.** `ml/` and `ml-worker/` are changed; the worker's changes are listed in `ml-worker/CHANGES.md`. The backend is untouched: it still sets `MODEL_PATH=<bundle>/petbert` and the worker uses only the parent directory. Required changes for other owners (backend, compose, CI, docs) are listed in `ml/documentation/restructure-change-request.md`.
- **Pins.** transformers 5.18.0 (tokenizers 0.23.2, huggingface_hub 1.33.0, safetensors 0.8.0), with torch 2.9.1 kept because it works with transformers 5.18 (R3).

## Acceptance
Phase 2 of 0008 is the model evidence. The rebuilt code had to reproduce it:

| Check | Result |
|---|---|
| R1: rebuilt GPU predict reproduces `rr-20261008T210937Z_predictions.csv` (65,009 rows, 58,172 cases) | pass: 0 differing cells under a per-partition replay; one-pass prediction identical on all shared cases; 141 extra cases from the 8-column rule |
| R2: in-memory `predict_frame` rows equal the CSV `run_predict` rows | pinned by a test |
| R3: CPU fp32 (torch 2.9.1+cpu, 8 threads) against GPU bf16 on calibration | pass: Good -0.03, Slight +0.05, CO +0.05, FP 0.00, FN -0.07 pp (bar 0.3 pp each); 21 of 5,828 cases change decision; CPU 0.72 s per report chunked, 0.41 s single call (p95 0.80) |
| R4: calibration silver-eval equals the freeze record | pass: Good 2,447, Slight 375, CO 456, FP 89, FN 712 of 4,079 rows; with the 17 extra calibration cases Good 60.06, Slight 9.21, CO 11.17, FP 2.23, FN 17.32 (4,081 rows) |
| R5: `python -m pytest ml -q` green | pending when this record was written; the adopt runs after it |
| R6: Opus sign-off | pending when this record was written; the adopt runs after it |

## Consequences
- A generation cannot load, promote, export or publish unless its spec check passes; an old-format bundle fails the new worker image and the reverse, so image and bundle switch together.
- Retraining cannot promote on silver alone; until gold-eval exists, `promote.py` refuses and a retrain stays a candidate.
- Architectural changes need a written history record before they can be adopted, so each one leaves an auditable reason in the manifest.
- The test partition is spent for this design; future accepts depend on gold.
- The worker is CPU-only (fp32); on the default 4-vCPU batch machine throughput is lower than on the 8-thread test box, and HTTP uploads are bounded by the worker timeout.

## Evidence
[0008](0008-joint-modernbert-report-mapping.md), the [experiment records](../experiments/2026-10-08-joint-modernbert-gate-term.md), and the [restructure plan](../plans/restructure-plan.md) (work-package notes carry the acceptance numbers). Current descriptions: [../../concepts/generations.md](../../concepts/generations.md), [../../concepts/coding.md](../../concepts/coding.md), [../../concepts/manual-audit.md](../../concepts/manual-audit.md), [../../concepts/report-mapping.md](../../concepts/report-mapping.md).
