# Retrieve-and-rerank v2, Phase 0: ModernBERT retrieval and a zero-shot OpenJev rerank

**Date:** 2026-09-30 · **Verdict:** reranker dropped; the retriever was adopted into Phase 1 ([next record](2026-10-08-joint-modernbert-gate-term.md))

## Hypothesis
The 2026-05-19 retrieve-and-rerank attempt ([record](2026-05-19-recsys-retrieve-and-rerank.md)) failed on retrieval recall (0.748 at K=20, 0.865 at K=50) and a calibration cliff, and said not to retry without a fix for both. Two changes address those: (1) retrieval from a single fine-tuned encoder reading the whole report, and (2) a community zero-shot reranker that answers "is a neoplasm diagnosed?" and scores each candidate term, so it can return one term, several or none, and needs no cross-case calibration of its own training.

## Setup
- **Retriever.** The 8 `report.csv` text columns, each prefixed by its heading, in one sequence of at most 4,096 tokens. `thomas-sounack/BioClinical-ModernBERT-base` fine-tuned end to end, mean-pooled, with a linear head over the 455 terms of the train silver (terms, not ICD codes). Trained on the 21,411 train cancer cases for 3 epochs (about 31 minutes per loss, peak 4.1 GiB). Two losses compared: `bce` (full multi-hot) and `softmax` (per-positive softmax cross-entropy, other positives masked). Only 6 train reports (0.03%) exceeded 4,096 tokens; no calibration report did.
- **Reranker.** OpenJev `qwen3.5-4b-nli-v5` (community, MIT licence, Qwen3.5 NLI 4B), run locally on a pinned revision with the model card's `decide` template. The hosted TypeSafe Jev was excluded: patient text must not leave the machine.
- **Evaluation data.** Calibration cancer cases: 2,697 cases, 3,392 gold pairs (9 out of vocabulary, counted as misses). Pilot: 1,000 calibration cases (50/50 cancer and non-cancer), split by an md5 bit into a fit half (515) and an eval half (485, of which 229 cancer); reranker temperature and thresholds were fitted on the fit half only.
- **Baseline.** WP14 (`gen-20260927T025939Z`), trained on the `three-way-v1` train partition and calibrated on its calibration partition.
- **Pre-registered bars.** Retrieval: go if pair recall is at least 0.95 at some N of at most 30 and group recall at least 0.98; kill if pair recall is below 0.92 at N=49. Reranker: ECE at most 0.05, best raw threshold within [0.2, 0.8] (the old attempt peaked at 0.995 or above), neoplasm AUC at least 0.95 (the existing gate gets 0.98); kill if any fails. With about 500 eval cases ECE has an uncertainty of about 0.02.

## Result

### Retrieval (calibration cancer cases, recall)
| loss | pair@10 | pair@20 | pair@30 | pair@49 | group@10 | group@20 | group@30 | case@30 | multi-term case@30 | rare-term (under 5 train cases) pair@30 |
|---|---|---|---|---|---|---|---|---|---|---|
| bce | .934 | .957 | .970 | .981 | .977 | .988 | .992 | .964 | .865 | .542 |
| softmax | .933 | .955 | .969 | .981 | .967 | .986 | .992 | .965 | .875 | .583 |

The two losses tie within noise (standard error about 0.35 pp). Both pass the bars (pair recall at least 0.95 by N=20; group recall 0.992 at N=30; 0.981 at N=49, far above the 0.92 kill line). BCE was chosen as the headline winner at every N up to 20; the pilot used N=30, with N=20 scored from the same raw file as a sensitivity check. Weak spots: rare terms and multi-term cases.

### OpenJev pilot (eval half, 485 cases)
| bar | n=30 | n=20 | verdict |
|---|---|---|---|
| neoplasm AUC at least 0.95 | 0.978 | 0.978 | pass |
| best raw threshold within [0.2, 0.8] | 0.47 (fit half), 0.30 (eval) | 0.47 / 0.24 | pass (no cliff) |
| ECE at most 0.05 | raw 0.089, temperature-scaled 0.101 | raw 0.083, temperature-scaled 0.077 | **fail** (beyond the +-0.02 noise) |

The ECE failure is the pre-registered kill for the zero-shot reranker. Verdicts on the pilot eval half (share % of scored rows, counts in brackets; row totals differ, 463 against 368):

| system | Good | Slight | CO | FP | FN | G+S |
|---|---|---|---|---|---|---|
| OpenJev, N=30, retriever candidates | 30.0 (139) | 29.2 (135) | 12.5 (58) | 7.1 (33) | 21.2 (98) | 59.2 |
| OpenJev, N=20 | 31.3 (147) | 28.7 (135) | 12.8 (60) | 7.9 (37) | 19.4 (91) | 60.0 |
| WP14 | 48.1 (177) | 12.5 (46) | 13.6 (50) | 2.4 (9) | 23.4 (86) | 60.6 |

OpenJev's FN share is lower only because its row total is larger; its FN count is higher (98 against 86). WP14's thresholds had seen this eval half, so its row is mildly optimistic.

### Why OpenJev under-performed (read-only investigation, no GPU)
- Oracle (gold forced into the candidates) equalled the retriever (hit rate 98.7%), so retrieval was not the bottleneck. OpenJev's top-1 is the silver term only 47.6% of the time on eval-half cancer cases (the retriever: 80.8%).
- No usage bug: template, label index, normalisation and the neoplasm question all matched the model card; nothing was truncated; no position or batching effect.
- A purpose mismatch: it was trained on about 2 to 6 mutually exclusive options with a defining rubric each. We gave it 30 overlapping hierarchical terms whose rubric (the group name) is identical for siblings. The card makes no calibration claim and has no medical benchmark.
- The dominant error is the NOS convention: in 55% of wrong picks the gold term is NOS and the pick is not (the reverse: 10%). NOS share is 68% in gold, 71% in the retriever's top-1 and 46% in OpenJev's picks. Top confusions: Mast cell tumor NOS to Subcutaneous mast cell tumor (19) and to Patnaik grade II (15); Hemangiosarcoma NOS to well differentiated (12). Mast-cell top-1 accuracy: OpenJev 0.12, retriever 1.00. On non-NOS gold the two are close (0.61 against 0.67).
- Entailment cannot rank specificity: in 39% of wrong cases both gold and pick score above 0.5. About 30% of wrong cases are true misses (gold P(entailment) below 0.1).
- No complementary signal: OpenJev fixes 17 retriever errors and overrides 163 correct ones; retriever logit plus lambda times log P(entailment) never beats the retriever alone (0.804 to 0.819 against 0.827).

### Option C: retriever top-1 with the WP14 case-presence gate (user-approved check)
The cheap diagnostic put to the user after the failure: take the retriever's own top-1 and let the existing WP14 gate decide cancer or not, no LLM. Whole calibration eval half (2,851 cases, 1,294 cancer, natural mix), share % (counts):

| system | Good | Slight | CO | FP | FN | G+S |
|---|---|---|---|---|---|---|
| retriever top-1 + WP14 gate | 60.6 (1,129) | 8.3 (155) | 7.8 (146) | 2.4 (44) | 20.8 (388) | 69.0 |
| retriever top-1 + retriever-score gate (not deployable) | 52.2 (1,102) | 5.3 (112) | 5.1 (108) | 17.3 (365) | 20.0 (423) | 57.5 |
| WP14 | 48.6 (1,005) | 14.6 (302) | 11.9 (246) | 2.6 (54) | 22.3 (461) | 63.2 |

On the 485-case pilot eval half: retriever + WP14 gate 57.9 / 7.6 / 10.9 / 1.8 / 21.8; retriever + OpenJev neoplasm gate 56.4 / 7.4 / 11.2 / 7.1 / 17.8; WP14 48.1 / 12.5 / 13.6 / 2.4 / 23.4. The retriever-score gate is not deployable: it was trained on cancer cases only, so 21% of non-cancer cases pass. Caveats: silver on calibration, not test; the WP14 gate's thresholds saw this half (both rows share the gate, so the comparison is fair but optimistic); no bootstrap interval at this stage, though the Good count gap (+124) is far outside noise.

## Why
1. A single end-to-end encoder over the whole report retrieves far more than the earlier PetBERT-section retriever did (the difference is not isolated to the encoder or the text): pair recall 0.970 at N=30 and 0.981 at N=49, against 0.748 at K=20 and 0.865 at K=50 before. The tail of the old failure (rare terms 0.54 to 0.58 at N=30, multi-term cases 0.865 to 0.875) remains.
2. The retriever's top-1 follows silver's conventions, NOS included (NOS share 71% against 68% in gold), which is the likely reason its own top-1 beats the pipeline's group-then-term path by +12.0 pp Good on calibration (not isolated by an ablation); an LLM reranker that prefers the more specific term works against the target. User direction on 2026-09-30: bronze should learn from silver as much as possible.
3. OpenJev cannot be calibrated into an acceptable operating point (ECE 0.077 to 0.101 against a bar of 0.05).

## Don't retry unless…
You have a reranker that was trained for silver's conventions (for example fine-tuned on train silver, the plan's named fallback), or a human audit shows that silver's NOS choices are wrong and the labels change. A zero-shot LLM that prefers specific terms will keep losing Good to NOS. Some OpenJev "misses" may be clinically more precise; only vet review can settle that, and it belongs on the audit list, not in model selection.

## Where it went
The reranker was dropped; the retriever (BCE) and the pre-registered bars carried into Phase 1 ([Phases 1 and 2](2026-10-08-joint-modernbert-gate-term.md)), which added a case-presence gate and ended in decision [0008](../decisions/0008-joint-modernbert-report-mapping.md). The experiment code was deleted in the bronze restructure ([0009](../decisions/0009-bronze-restructure-and-lifecycle.md)); the git history keeps it.
