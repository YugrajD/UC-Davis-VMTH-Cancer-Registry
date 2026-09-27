# ML Directory — Overview

> **Describes the tree after the `ml/next` → `ml` cutover (WP13).** Until that commit, the new code
> lives in `ml/next/` and the old tree is still production; this banner is removed by the cutover
> commit. See [ml-rewrite-plan.md](ml-rewrite-plan.md) for status.

A machine learning system that codes veterinary cancer cases to standardized Vet-ICD-O-canine-1
labels (term, group, ICD code) from three sources of decreasing confidence — a specialist's
**manual audit** (gold), the clinic's **diagnosis** text (silver), and the pathology **report** text
(bronze) — and records how far each code can be trusted. See
[icd-mapping-strategy.md](icd-mapping-strategy.md) for the full strategy.

Production report-mapping inference (`scripts/predict.py`) loads a per-section contrastive PetBERT
backbone, embeds each report as a 2304-dim concat-3 vector, and runs a 4-stage classifier pipeline
(case-presence gate → group classifier → per-group label-presence head → keyword correction).

**Current reference baseline: G+S 61.76% on the eval half of the legacy split** (4,456 per-code
rows; Good 45.8, Slight 16.0, CO 15.3, FP 2.6, FN 20.4). The previously published **62.1%** does not
reproduce from the files on disk today and is superseded — see
[report-mapping.md](report-mapping.md#the-621-caveat) for the reconciliation.

---

## Directory map

```
ml/
  config.py            Every path, incl. ARCHIVE_ROOT (written only by generations/)
  io_utils.py           The one shared CSV reader/writer (BOM, latin-1 reports / utf-8 outputs)
  taxonomy/             labels.csv; taxonomy.py (load, label texts, code<->term<->group);
                        behavior.py; subtype.py
  manual_audit/         sheets.py (review CSVs + sidecars); tier3_audit.py (row-level Tier-3
                        sample/pilot/ingest -> audit store); eval_batch.py (case-level gold-eval
                        batch); gold.py (case-level gold ingest); cause_pass.py (misses -> cause store)
  diagnosis_mapping/    keyword_tiers.py (no_signal/tier1/tier2); llm_tier.py; llm_client.py (local
                        only); cleanup.py; silver.py (run -> versioned silver generation); stats.py
  report_mapping/
    sections.py          The single concat-3 section spec + report-text builder
    model/               backbone.py (load, mean-pool embed); heads.py (3 heads, state_dict-
                         compatible with legacy .pt); generation.py (load/save a generation dir)
    training/             recipe.py (production hyperparameters + seeds); labels.py; backbone.py;
                         case_presence.py; group.py; label_presence.py; calibrate.py; oof.py
    inference/            embedding_cache.py (content-hash keyed); stages.py; keyword_correction.py
                         (+ Lipoma rescue); predict.py
  coding/                rule.py (decisive/vague); adopt.py (gold > silver > bronze); corrected.py
                        (report-mapping training labels); queue.py (review queue)
  evaluation/            verdicts.py; intervals.py (Wilson, Kish n_eff, stratified case-cluster
                        bootstrap); silver_eval.py; gold_eval.py (four results); audit_rates.py
  generations/           manifest.py; splits.py; guards.py (leakage checks); promote.py; triggers.py
  handoff/               contracts.py; imports.py; exports.py; worker_format.py (shared with
                        ml-worker)
  scripts/               Thin entry points — see "Entry points" below
  tests/                 pytest, synthetic fixtures only (a tiny random-init BERT saved locally)
```

| Subdir | Purpose |
|---|---|
| `taxonomy/` | Vet-ICD-O-canine-1 table (845 terms, 52 groups) plus behavior/subtype keyword filters used by report-mapping's Stage 3b. |
| `manual_audit/` | Gold: the row-level Tier-3 diagnostic audit, the case-level gold-eval batch, the gold store, the cause pass. See [manual-audit.md](manual-audit.md). |
| `diagnosis_mapping/` | Silver: the 3-tier diagnosis cascade + ensemble cleanup, versioned as silver generations. See [diagnosis-mapping.md](diagnosis-mapping.md). |
| `report_mapping/` | Bronze: the 4-stage PetBERT pipeline, its training and its inference. See [report-mapping.md](report-mapping.md). |
| `coding/` | Applies gold > silver > bronze per case; builds the report mapping's training labels; builds the review queue. See [coding.md](coding.md). |
| `evaluation/` | Verdict scoring, confidence intervals, silver-eval and the four gold-eval results. See [evaluation.md](evaluation.md). |
| `generations/` | Manifests, splits, leakage guards, promotion and retraining triggers — shared by every versioned artefact. See [generations.md](generations.md). |
| `handoff/` | File contracts with the cloud (registry app / backend) and the worker bundle contract. See [handoff.md](handoff.md). |
| `scripts/` | The only Python entry points. |
| `tests/` | pytest; synthetic fixtures only — never reads `ml/data/` or `ml/output/`. |

---

## Where outputs go

All under `output/` (gitignored). Paths below are the `config.py` constants, relative to `ml/`.

| Path | Contents |
|---|---|
| `output/splits/<split_id>/{train,calibration,test}_cases.txt` + `manifest.json` | A split generation. `three-way-v1` is the default. |
| `output/silver/<silver_id>/annotation.csv` + `manifest.json` | A silver generation (the diagnosis cascade's output). |
| `output/diagnosis_mapping_stats/<silver_id>/` | Coverage-stats artifacts for a silver generation. |
| `output/manual_audit/audit_store.csv` | Row-level Tier-3 audit judgements. Never gold. |
| `output/manual_audit/gold_store.csv` | Case-level gold: one row per (case_id, code), or one `NO_CANCER` row. |
| `output/manual_audit/eval_batch_ledger.csv` | The gold-eval batch series ledger (`N_h`, `n_h`, `sample_weight`, ...). |
| `output/manual_audit/cause_store.csv` | Cause-pass answers on misses. |
| `output/manual_audit/tier3_audit/` | Tier-3 audit review/key/instructions/taxonomy sheets. |
| `output/manual_audit/eval_batch/` | Case-level eval-batch review sheets. |
| `output/coding/corrected_annotations.csv` | The report mapping's training labels (train partition only). |
| `output/coding/adopted_codes.csv` | The registry's adopted code per case. |
| `output/coding/review_queue.csv` | Cases the specialist needs to look at. |
| `output/report_mapping/current/`, `output/report_mapping/candidate/` | Report-mapping generations (production / being trained). Layout in [report-mapping.md](report-mapping.md). |
| `output/report_mapping/embedding_cache/<key>.npz` | Content-hash keyed PetBERT embedding cache. Never bundled into a generation. |
| `output/predictions/<generation_id>_predictions.csv` | Stamped report-mapping predictions. |
| `output/eval/silver_eval_history.csv` | One line per `evaluate.py silver` run. |
| `output/handoff/inbox/`, `output/handoff/outbox/` | Cloud file contracts (pending diagnoses, gold imports; silver/coding exports; worker bundle tarballs). |
| `output/archive/YYYY-MM-DD_<desc>/` | An archived (superseded) report-mapping generation. Written only by `generations/promote.py`; nothing loads from it. |

---

## Entry points (`scripts/`)

| Script | Subcommands | Does |
|---|---|---|
| `map_diagnoses.py` | `run \| import-legacy \| stats` | Diagnosis-mapping cascade; import the legacy silver generation; coverage stats. |
| `audit.py` | `tier3-sample \| tier3-pilot \| tier3-ingest \| eval-batch \| ingest-sheet \| ingest-gold \| cause-sheet \| ingest-cause` | Manual-audit sheets and stores. |
| `split.py` | `import-legacy \| create \| check` | Split generations; leakage guards. |
| `code_cases.py` | `adopt \| corrected \| queue` | Coding-rule outputs. |
| `train.py` | `--stage backbone\|case-presence\|group\|label-presence\|heads\|oof` | Train a report-mapping candidate. |
| `calibrate.py` | — | Fit every inference threshold on the calibration partition. |
| `predict.py` | — | Stamped report-mapping predictions (`--embed-only` to just build the cache). |
| `evaluate.py` | `silver \| gold \| audit-rates` | Verdicts, the four gold-eval results, Tier-3 audit rates. |
| `promote.py` | `[--apply]` | The promotion recommendation, or carrying it out. |
| `generations.py` | `import-gen0 \| import-cache \| status` | Legacy generation imports; current/candidate status + trigger check. |
| `handoff.py` | `import-pending \| import-gold \| export-silver \| export-coding \| export-bundle` | Cloud file contracts. |
| `retrain_cycle.py` | — | The strategy's local retraining lane, end to end (recommend-only). |

`split.py check` also runs automatically inside `train.py`, `calibrate.py`, `evaluate.py gold` and
`promote.py`, and refuses on any leakage-guard violation.

---

## Quick start

Use `ml/.venv/Scripts/python.exe` on Windows, `ml/.venv/bin/python` on macOS/Linux. Every
`scripts/*.py` adds `ml/` to `sys.path` automatically.

**Run tests:**
```bash
ml/.venv/Scripts/python.exe -m pytest ml -q -p no:cacheprovider
```

**Run inference** (scores every row in `ml/data/report.csv` against the production generation):
```bash
ml/.venv/Scripts/python.exe ml/scripts/predict.py --generation current --device cuda --local-only
```

**Retrain** (heads only, on an existing silver generation and split):
```bash
ml/.venv/Scripts/python.exe ml/scripts/retrain_cycle.py --silver silver-0-legacy --device cuda --local-only
```
See [training-guide.md](training-guide.md) for the full cycle broken into individual steps, exact
commands and runtimes.

**Run the diagnosis cascade** (needs LM Studio running locally):
```bash
ml/.venv/Scripts/python.exe ml/scripts/map_diagnoses.py run --id silver-1
```

---

## Where to read next

| File | When to read it |
|---|---|
| [icd-mapping-strategy.md](icd-mapping-strategy.md) | The project-level strategy: how gold, silver and bronze combine to code every case. Read this first. |
| [report-mapping.md](report-mapping.md) | You're invoking `predict.py`/`train.py`/`calibrate.py`, or want the 4-stage design and why it looks the way it does. |
| [diagnosis-mapping.md](diagnosis-mapping.md) | You're running or debugging `map_diagnoses.py`. |
| [manual-audit.md](manual-audit.md) | You're running the Tier-3 audit or an eval-batch draw, or ingesting gold. |
| [coding.md](coding.md) | You want the adoption rule, the corrected-annotations builder, or the review queue. |
| [evaluation.md](evaluation.md) | You're scoring predictions, reading a gold-eval report, or want the CI methodology. |
| [generations.md](generations.md) | You're creating a split, promoting a candidate, or want the retraining-trigger logic. |
| [handoff.md](handoff.md) | You're exchanging files with the backend, or building/reading a worker bundle. |
| [training-guide.md](training-guide.md) | You're retraining and need exact commands + expected runtimes. |
| [ml-rewrite-plan.md](ml-rewrite-plan.md) | You're working on the rewrite itself: contract, work packages, parity runbook, status. |
| [ml-worker-change-request.md](ml-worker-change-request.md) | You're deploying the worker or changing the backend's model-upload path. |
| [resume-on-new-machine.md](resume-on-new-machine.md) | You're setting this project up on a different computer. |
| [box-rclone-sync-proposal.md](box-rclone-sync-proposal.md) | Proposed (not implemented) Box + rclone layout for sharing data and weights. |
| [archive/](archive/) | Historical docs — the pre-rewrite tree, the annotation-redesign plan, phase logs. Do not consult for current behavior. |
