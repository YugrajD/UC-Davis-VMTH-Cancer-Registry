# Retrieve-and-rerank v2, Phases 1 and 2: ModernBERT term head with a neoplasm gate, scored once on test

**Date:** 2026-10-08 (Phase 1 ran 2026-09-30 to 2026-10-03) · **Verdict:** adopted ([0008](../decisions/0008-joint-modernbert-report-mapping.md), [0009](../decisions/0009-bronze-restructure-and-lifecycle.md))

## Hypothesis
After [Phase 0](2026-09-30-retrieve-rerank-v2-openjev.md) the LLM rerank was not needed: the ModernBERT retriever's own top-1 plus a case-presence gate already beat WP14 on calibration. Phase 1 asks which gate, how many models and which thresholds; Phase 2 asks whether the frozen result beats WP14 on the held-out test partition under bars fixed in advance.

## Setup
- **Design.** Whole-report text (8 headed columns, at most 4,096 tokens), BioClinical ModernBERT-base, 455-term head. Decision rule: gate probability below `gate_t` gives one `Non-Cancer` row; otherwise the rank-1 term, plus rank-2 when p2 is at least `tau` and p2/p1 is at least `rho`. Thresholds are fitted on the calibration partition only, objective G+S with Good as the tie-break (WP14's, which searched a 0.50 to 0.90 gate grid; the Phase 1 sweeps below also went lower, down to 0.20).
- **Baseline: WP14** (`gen-20260927T025939Z`), refitted on `three-way-v1` (train only, calibration only), scored on silver. The production `gen-0-legacy` fitted its thresholds on test and is optimistic, so it is not the baseline.

| WP14 on `three-way-v1` test (4,299 rows) | Good | Slight | CO | FP | FN | G+S |
|---|---|---|---|---|---|---|
| share % | 46.27 | 15.86 | 12.84 | 2.35 | 22.68 | 62.13 |

  On calibration (4,322 rows, in-sample): Good 47.59, Slight 14.69, CO 12.59, FP 2.43, FN 22.70, G+S 62.29. The ruler (WP14's uncommon-groups list) reproduced these test numbers exactly before any new model touched test. `evaluation/verdicts.py` scores `unidentified_cancer` as CO, not FN (a known artefact).
- **Pre-registered bars** (copied here so they survive the deletion of the working notes):
  - Retrieval (Phase 0): go at pair recall of at least 0.95 for some N of at most 30 and group recall of at least 0.98; kill below 0.92 at N=49.
  - Reranker (Phase 0): ECE at most 0.05, best raw threshold within [0.2, 0.8], neoplasm AUC at least 0.95; kill if any fails.
  - **Final (Phase 2):** Good bootstrap lower bound at least -2 pp, G+S at least the baseline's, FP at most the baseline's plus 1 pp. Kill if still more than 5 pp behind after one variant.
  - Silver noise (about 25 to 30% in some groups, from the annotation audit) caps every number.
- **Bootstrap.** `evaluation/intervals.paired_bootstrap`: stratified case clusters, 1,000 resamples, seed 0; Phase 2 used a single stratum and write-once output.

## Result

### Gate bake-off (Phase 1; calibration eval half, share % with counts where recorded)
Gate model: a binary ModernBERT trained on all 46.5k train cases with label rows, mean pooling, 3 epochs, lr 5e-5, best epoch on a 5% held-out slice (held-out AUC by epoch 0.9898 / 0.9902 / 0.9894; epoch 1 kept; 55.8 minutes). Retriever seed 2 retrained for the seed check (31 minutes).

| system | gate_t | Good | Slight | CO | FP | FN | G+S | gate AUC |
|---|---|---|---|---|---|---|---|---|
| seed 42 + old WP14 gate | .70 | 58.77 (1,183) | 9.64 (194) | 10.58 (213) | 3.87 (78) | 17.14 (345) | 68.41 | .9863 |
| seed 42 + new gate | .65 | 59.24 (1,189) | 9.87 (198) | 10.91 (219) | 3.09 (62) | 16.89 (339) | 69.11 | .9906 |
| seed 2 + old gate | .70 | 58.98 | 9.71 | 10.65 | 3.98 | 16.67 | 68.69 | .9863 |
| seed 2 + new gate | .35 | 59.15 (1,209) | 9.88 | 11.35 | 4.26 (87) | 15.36 (314) | 69.03 | .9906 |
| WP14 | .80 | 48.36 (1,005) | 14.10 (293) | 12.27 (255) | 2.60 (54) | 22.67 (471) | 62.46 | |

- **Seed check passes.** Good is about 10.6 pp above WP14 with either seed. Top-1 is 1,046 of 1,294 cancer cases for both seeds, a coincidence (checked independently): they pick different terms on 28% of cases.
- **The new gate beats the old one**: higher AUC, lower FP and FN at matched retriever, and a flat curve (eval G+S 68.6 to 69.6 for gate_t 0.20 to 0.90); the old gate falls off below 0.40.
- **FP risk.** Every system's FP share is above WP14's 2.60, and seed 2 + new at gate 0.35 (FP 4.26) would fail the Phase 2 FP bar. The fit-half objective is flat, so the chosen gate_t is close to arbitrary and trades FP for FN. A cap on FP at freeze time was proposed. The user stopped here on 2026-10-01 before any freeze.

### One model instead of two
| calibration eval half | Good | Slight | CO | FP | FN | G+S | gate AUC | cancer / non-cancer passed | top-1 |
|---|---|---|---|---|---|---|---|---|---|
| seed 42 + new gate (2 models) | 59.24 (1,189) | 9.87 (198) | 10.91 (219) | 3.09 (62) | 16.89 (339) | 69.11 | .9906 | 94.2 / 2.8 | 80.8 |
| shared gate head on frozen retriever vectors | 59.91 (1,149) | 8.50 (163) | 8.50 (163) | 3.75 (72) | 19.34 (371) | 68.40 | .9885 | 93.7 / 3.9 | n/a |
| joint model (1 model) | 60.14 (1,204) | 8.99 (180) | 11.29 (226) | 3.55 (71) | 16.03 (321) | 69.13 | .9906 | 95.6 / 3.3 | 80.4 |

- **Shared gate (option 1) failed the pre-set adoption rule** (within about 0.5 pp of the two-model result on each verdict, AUC about 0.99): Good -40 cases, FP +10, FN +32 (counts), FP 3.75 would exceed WP14 plus 1 pp. The bake-off also picked a different second-term rule (tau .30, rho 0 against .10 / .25), explaining part of the Slight and CO shift. Its held-out AUC was optimistic because the retriever encoder had seen the holdout.
- **Joint model (option 2)** matches two models within noise: one ModernBERT with a 455-term head and a gate head, trained on all 46,572 train cases (44,243 fit, 2,329 held out of both losses; gate loss on all cases, term loss on cancer cases, 3 epochs, lr 5e-5, final epoch kept; 55.9 minutes, 4.15 GiB). Held-out gate AUC by epoch .9876 / .9898 / .9906 (the standalone gate: .9902 on the same slice); held-out term top-1 .742 / .760 / .766. Count deltas against two models: Good +15, Slight -18, CO +7, FP +9, FN -18; G+S and AUC identical. The differences are mostly the operating point (gate_t .60 against .65) and all lie within about 1 standard error (about 0.7 to 1.1 pp on about 2,000 rows).
- **CPU timing** (50 calibration reports, 8 threads, fp32, mean 487 tokens): two models 0.571 s per report (median 0.378, p95 1.63); one model 0.284 s (median 0.185, p95 0.80), a ratio of 2.0.

### Freeze with an FP cap
- First freeze `rr-20261003T192657Z`: cap = WP14's calibration FP 2.43% plus 0.5 pp = 2.93%; gate .65, tau .10, rho .25. Calibration in-sample: Good 59.99, Slight 9.24, CO 11.35, FP 2.86, FN 16.56, G+S 69.23 (4,124 rows). Gate .65 to .85 is a G+S plateau (69.23 to 69.09) while FP falls from 2.86 to 1.64; gate_t 0.50 and below is ineligible under the cap.
- The user chose to re-freeze with the cap at WP14's calibration FP, 2.43%: **`rr-20261008T210937Z`**, gate .75, tau .10, rho .25. Calibration in-sample: Good 59.99, Slight 9.19, CO 11.18, FP 2.18, FN 17.46, G+S 69.18 (4,079 rows). It gives up 0.05 pp of G+S for FP margin on the test bar (WP14 test FP 2.35 plus 1 = 3.35). The first freeze was never scored on test.

### Phase 2: one-shot test, `three-way-v1` test partition
Predictions cover 58,172 cases (65,009 rows; 5,818 test cases) with the same coverage as WP14. Share % of scored rows with counts:

| system | rows | Good | Slight | CO | FP | FN | G+S |
|---|---|---|---|---|---|---|---|
| `rr-20261008T210937Z` (joint, gate .75) | 4,028 | 58.47 (2,355) | 10.60 (427) | 10.92 (440) | 2.14 (86) | 17.87 (720) | 69.07 |
| WP14 | 4,299 | 46.27 (1,989) | 15.86 (682) | 12.84 (552) | 2.35 (101) | 22.68 (975) | 62.13 |

Paired bootstrap, new minus WP14, percentage points (95% CI):

| Good | Slight | CO | FP | FN | G+S |
|---|---|---|---|---|---|
| +12.20 [+10.91, +13.60] | -5.26 [-6.39, -4.11] | -1.92 [-2.98, -0.91] | -0.21 [-0.82, +0.32] | -4.80 [-5.85, -3.83] | +6.94 [+5.53, +8.48] |

Bars: Good lower bound +10.91 against at least -2 (pass); G+S 69.07 against at least 62.13 (pass); FP 2.14 against at most 3.35 (pass). Overall pass.

Discordant cases: Good under the new model but not under WP14, 371; Good under WP14 but not the new model, 96. Cancer cases where the new model says Non-Cancer and WP14 gives a term: 55; the reverse: 117. Test matches calibration (Good 58.5 against 60.0; FP 2.14 against 2.18), so there is no sign of threshold overfitting.

## Why
- **Good +12.2 pp.** Likely reasons (no ablation isolated them): one encoder reading the whole report (no 512-token section truncation) with one head over the exact terms, trained on silver's term choices including its NOS habit; the old pipeline had to get the group right, then the term inside the group, then survive keyword correction.
- **Slight -5.26 pp.** Slight is "right group, wrong term". The test shows fewer Slight rows and more Good rows, but no per-group or per-case breakdown of where the Slight cases went was made (a post-hoc per-group look at test, diagnosis only, is an open optional step).
- **Gate and term head together.** Joint training matched the standalone gate's AUC while the shared head on frozen retriever vectors did not; the retriever encoder had been trained on cancer cases only, which is the likely reason (not tested separately).
- **Limits.** Only the 455 terms seen in train silver can be predicted (of 844); rare-term pairs are about 6% of gold. Silver noise caps every number.

## Don't retry unless…
- The shared-gate idea: not unless the retriever's training includes non-cancer cases (which is just the joint model).
- Tuning the second-term rule or gate threshold on test: the test partition is spent for this design; use calibration and, later, gold.
- Rare-term ideas (label-text-initialised head, label-text scoring head, contrastive pre-adaptation): discussed on 2026-10-08 and parked by the user; the expected gain is at most about 1 to 2 pp Good. See [open-ideas.md](../open-ideas.md).

## Where it went
Adopted as the report-mapping model in the bronze restructure: [0008](../decisions/0008-joint-modernbert-report-mapping.md) is the design decision and [0009](../decisions/0009-bronze-restructure-and-lifecycle.md) the lifecycle and adoption record ([plan](../plans/restructure-plan.md)). The frozen weights were re-housed as the first generation, not retrained. The Phase 1 and Phase 2 scripts (`bronze_rr/`, `rr_phase1.py`) were deleted in the restructure; their output directories and this record carry the numbers.
