# ML Directory — Overview

A machine learning system that codes veterinary cancer cases to standardized Vet-ICD-O-canine-1
labels (term, group, ICD code) from three sources of decreasing confidence — a specialist's
**manual audit** (gold), the clinic's **diagnosis** text (silver), and the pathology **report** text
(bronze) — and records how far each code can be trusted. Start with
[concepts/icd-mapping-strategy.md](concepts/icd-mapping-strategy.md) for the full strategy.

Report mapping (`scripts/predict.py`) loads one jointly trained BioClinical ModernBERT, reads each
report as a single text (the eight `report.csv` text columns under their headings) in one pass, and
answers two questions from the same pooled vector: a **gate** (is there a cancer diagnosis) and a
**term head** (which Vet-ICD-O term). A fitted decision rule turns the two into the predicted
terms. Design: [concepts/report-mapping.md](concepts/report-mapping.md); why:
[history/decisions/0008](history/decisions/0008-joint-modernbert-report-mapping.md).

**Where things stand (2026-10-09)**
- Production `current/` is `gen-20261009T224439Z` (split `three-way-v1`, silver `silver-0-legacy`):
  the weights of the retrieve-and-rerank v2 experiment's joint model, frozen and re-housed in the
  new generation layout, not retrained, and adopted as the first generation through
  [ADR 0009](history/decisions/0009-bronze-restructure-and-lifecycle.md). Thresholds are frozen at
  gate 0.75, tau 0.10, rho 0.25 and the review cut at 0.85.
- One-shot test on the `three-way-v1` test partition (scored against silver `silver-0-legacy`, so
  agreement with silver rather than accuracy), against the generation it replaced (`gen-20260927T025939Z`,
  the PetBERT four-stage pipeline). All figures are percent of per-code rows:

  | | Good | Slight | CO | FP | FN | G+S |
  |---|---|---|---|---|---|---|
  | Joint ModernBERT | 58.47 | 10.60 | 10.92 | 2.14 | 17.87 | 69.07 |
  | Four-stage (replaced) | 46.27 | 15.86 | 12.84 | 2.35 | 22.68 | 62.13 |

  Paired bootstrap, new minus old: Good +12.20 [+10.91, +13.60], Slight -5.26, CO -1.92,
  FP -0.21 [-0.82, +0.32], FN -4.80, G+S +6.94 [+5.53, +8.48]. All three pre-registered bars passed.
  The test partition is **spent** for this design: never tune on it. The experiment record is
  [history/experiments/2026-10-08-joint-modernbert-gate-term.md](history/experiments/2026-10-08-joint-modernbert-gate-term.md).
- No real gold exists yet. Gold-eval, promotion and `retrain_cycle.py` wait on the specialist's
  first reviews; until then `retrain_cycle.py` stops before training, and a retrain would be judged
  on gold only.
- The cloud side must follow the restructure (new worker image and bundle, new review rule):
  [restructure-change-request.md](restructure-change-request.md) lists what other owners change.

Good / Slight / CO (completely off) / FP / FN are the evaluation verdicts, defined in [concepts/evaluation.md](concepts/evaluation.md).

```mermaid
flowchart LR
    rep["report.csv"] --> bronze["Report mapping<br>(bronze)"]
    dia["diagnoses.csv"] --> silver["Diagnosis mapping<br>(silver)"]
    gold["Gold store<br>(specialist reviews)"] --> coding
    bronze --> coding["Coding<br>gold > silver > bronze"]
    silver --> coding
    coding --> comb["combined_predictions"]
    comb --> app["Registry app<br>(code of record)"]
    coding -->|"review queue"| list["Audit list"]
    audits["Audit and eval<br>batches"] --> list
    list --> app
    app -->|"gold export"| gold
```

The registry app loads `combined_predictions` as each case's code of record and shows the audit
list as the Audit Worklist; the specialist's reviews come back as gold, and the next combined file
carries them. File formats: [reference/handoff-contracts.md](reference/handoff-contracts.md).

---

## How the docs are organised

| Folder | Holds | Read it when |
|---|---|---|
| [concepts/](concepts/) | How each part works and why | You want to understand a component |
| [how-to/](how-to/) | Exact commands for one task | You want to do something |
| [reference/](reference/) | Lookup tables: every script and flag, every path, every file contract | You need a precise fact |
| [history/](history/README.md) | What we tried, why it won or failed, finished plans, frozen training logs | You're about to try an idea, or wonder why something is the way it is. Never for current behaviour. |

Each fact has one home doc; other docs link to it.

| Doc | What it covers |
|---|---|
| [concepts/icd-mapping-strategy.md](concepts/icd-mapping-strategy.md) | The strategy: gold > silver > bronze, split roles, the improvement cycle, what is built and what is future. **Read first.** |
| [concepts/report-mapping.md](concepts/report-mapping.md) | Bronze: the input text, the joint model, the decision rule, calibration and the review cut, training, why. |
| [concepts/diagnosis-mapping.md](concepts/diagnosis-mapping.md) | Silver: the tiered diagnosis cascade, LLM cleanup, silver generations. |
| [concepts/manual-audit.md](concepts/manual-audit.md) | Gold: origins, the two audits, eval batches, the universal audit list, the cause pass. |
| [concepts/coding.md](concepts/coding.md) | Combining gold/silver/bronze per case; corrected annotations; the bronze review rule and the ML review queue. |
| [concepts/evaluation.md](concepts/evaluation.md) | Verdicts, silver-eval, the four gold-eval results, confidence intervals. |
| [concepts/generations.md](concepts/generations.md) | Generation layout, manifests, the spec check, splits, guards, retrain versus adopt, triggers, archive. |
| [how-to/train-and-promote.md](how-to/train-and-promote.md) | Retrain a candidate, calibrate, evaluate, promote; adopt an architectural change; recalibrate. |
| [how-to/run-inference.md](how-to/run-inference.md) | Produce report-mapping predictions. |
| [how-to/run-audit-cycle.md](how-to/run-audit-cycle.md) | Draw audit and eval batches, ingest gold, score it. |
| [how-to/sync-with-s3.md](how-to/sync-with-s3.md) | Move data, stores and model generations between machines. |
| [how-to/new-machine-setup.md](how-to/new-machine-setup.md) | Set the project up on a new computer. |
| [reference/scripts-and-flags.md](reference/scripts-and-flags.md) | Every `scripts/*.py`, subcommand and flag with its default. |
| [reference/paths.md](reference/paths.md) | Every `config.py` path and what lives there under `output/`. |
| [reference/handoff-contracts.md](reference/handoff-contracts.md) | Files exchanged with the backend and the ml-worker bundle. |
| [reference/classification-systems.md](reference/classification-systems.md) | Background on the ICD / Vet-ICD-O coding systems. |
| [restructure-change-request.md](restructure-change-request.md) | What the owners of the backend, compose file, CI, docs and frontend must change after the 2026-10-09 restructure. |
| [history/README.md](history/README.md) | How we got here: timeline and index of every experiment and decision. |
| [history/open-ideas.md](history/open-ideas.md) | Ideas not tried yet. |

---

## Directory map

```
ml/
  config.py             Every path (incl. ARCHIVE_ROOT, written only by generations/), the FP cap, the review target and the S3 constants
  io_utils.py           The one shared CSV reader/writer
  pytest.ini            Test settings (testpaths=tests)
  requirements.txt      Python dependencies (CUDA 12.8 torch index noted inside)
  taxonomy/             labels.csv (845 terms, 52 groups); taxonomy.py; behavior.py; subtype.py
  manual_audit/         sheets.py; terms.py; diagnosis_mapping_audit.py; report_mapping_audit.py;
                        eval_batch.py; audit_list.py; gold.py; cause_pass.py
  diagnosis_mapping/    keyword_tiers.py; llm_tier.py; llm_client.py (local LLM only); cleanup.py;
                        silver.py (versioned silver generations); stats.py
  report_mapping/
    text.py             The eight report columns -> one text; has_text
    batching.py         Length-bucketed steps and token-budget micro-batches
    model.py            ENCODER_ID; JointClassifier (term head + gate head); save/load
    generation.py       Generation layout, manifest, spec check (no torch)
    inference/          scoring.py; decide.py (thresholds -> rows); predict.py (load, predict_frame, run_predict)
    training/           recipe.py; labels.py; data.py; loop.py; train.py; calibrate.py; crossfit.py
  coding/               rule.py; combine.py; corrected.py; queue.py
  evaluation/           verdicts.py; intervals.py; silver_eval.py; gold_eval.py; audit_rates.py;
                        uncommon_groups.txt (the scoring ruler: 24 groups)
  generations/          manifest.py; splits.py; guards.py; triggers.py; promote.py
  handoff/              contracts.py; imports.py; exports.py; worker_format.py (shared with ml-worker)
  s3sync/               remote.py; sets.py; models.py; hooks.py; client.py; files.py; state.py
  scripts/              Thin entry points (below)
  tests/                pytest, synthetic fixtures only — never reads data/ or output/
  documentation/        These docs
  data/, output/        Private inputs and all outputs (gitignored) — see reference/paths.md
ml-worker/              The cloud inference service; imports report_mapping/ and handoff/worker_format.py; its CHANGES.md records every change
```

## Entry points (`scripts/`)

One line each; every flag is in [reference/scripts-and-flags.md](reference/scripts-and-flags.md).

| Script | Does |
|---|---|
| `map_diagnoses.py` | `run` the diagnosis cascade into a silver generation; `stats` on one. |
| `audit.py` | Draw audit and eval batches; ingest sheets, gold and cause answers. |
| `split.py` | `create` a split generation; `check` it against the leakage guards. |
| `code_cases.py` | `combine`, `corrected`, `queue`: the coding-rule outputs. |
| `train.py` | One step: labels + split to a candidate generation, calibration pending. |
| `calibrate.py` | Fit gate_t, tau, rho (under the FP cap) and the review cut on the calibration partition. |
| `predict.py` | Stamped report-mapping predictions from a generation. |
| `oof.py` | 5-fold cross-fit over the train partition, for the Report-Mapping audit. |
| `evaluate.py` | `silver`, `gold`, `audit-rates`. |
| `promote.py` | The promotion recommendation (gold rule); `--apply` carries it out; `--adopt --reason` takes an architectural change; `--publish` then pushes `current/` to S3. |
| `generations.py` | `status` of current/candidate/archive and the triggers; `recalibrate` a generation into a new one. |
| `handoff.py` | Import from and export to the backend (`--push` to sync what it wrote). |
| `sync.py` | S3 sync of data, stores and the `current/` generation (dry run unless `--apply`). |
| `retrain_cycle.py` | The local retraining lane end to end; recommends by default, `--apply` promotes when the gold rule passes. |

---

## Quick start

Use `ml/.venv/Scripts/python.exe` on Windows (`ml/.venv/bin/python` elsewhere). Setup:
[how-to/new-machine-setup.md](how-to/new-machine-setup.md).

**Run tests** (includes `tests/test_docs.py`, which checks these docs' links, script names and flags):
```bash
ml/.venv/Scripts/python.exe -m pytest ml -q -p no:cacheprovider
```

**Run inference** against the production generation:
```bash
ml/.venv/Scripts/python.exe ml/scripts/predict.py --generation current
```

**Run the diagnosis cascade** (not a smoke test: it creates a new silver generation, needs LM Studio running locally, and takes tens of minutes):
```bash
ml/.venv/Scripts/python.exe ml/scripts/map_diagnoses.py run --id silver-1
```

**Retrain or adopt:** see [how-to/train-and-promote.md](how-to/train-and-promote.md).
