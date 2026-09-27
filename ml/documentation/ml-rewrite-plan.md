# ML Rewrite Plan — implementing the ICD mapping strategy

**Status:** Approved 2026-09-25 (all recommended defaults accepted). Implements
[icd-mapping-strategy.md](icd-mapping-strategy.md) as a clean-slate rewrite of `ml/` and `ml-worker/`.
This is the working contract for the Part B agent team and for any later session picking the work up.

## Status and next steps (2026-09-26, Windows session)

Work is committed on `Revised-ICD-Mapping`. The suite passes with 617 tests on Windows
(`ml\.venv\Scripts\python.exe -m pytest ml/next -q -p no:cacheprovider`; the Windows venv was brought up to
`ml/requirements.txt`: numpy 2.x, pandas 2.2.3, scikit-learn 1.5.0, pytest).

**Parity (L1–L3 gate the cutover):**

| Level | Where | Result |
|---|---|---|
| L1 | Mac, re-run on Windows | PASS, identical verdicts on the eval half and the full test |
| L2a | Mac | PASS (68,800 rows; 199 Uncommon-reordered cases) |
| L2b | Windows, CUDA | PASS: 58,313 cases re-embedded, min cosine 1.000000; 68,800/68,800 rows identical (199 Uncommon-reordered); eval-half G+S 61.8%, +0.00 pp. Re-embed 9.2 min |
| L3 | Windows, CUDA | PASS: mean G+S 61.44% (seeds 60.4 / 61.6 / 62.3, sd 0.96) vs 61.76%, −0.32 pp (tolerance ±1.93). Good −0.45, Slight +0.12, CO +0.11, FP +0.83, FN −0.62 pp. 46,572 gate/group cases as expected. About 6.7 min per seed |
| L4 | Windows | not run yet (report only) |

- **L3 notes.** Groups losing > 5 pp (reported, not gating): Odontogenic tumors (n=51, 86.3 → 78.9) and
  Transitional cell papillomas and carcinomas (n=61, 75.4 → 69.3); at these sizes 5 pp is 3 codes. The FP rise
  is gate calibration: with the gate held at 0.80 (legacy mode), the retrained gates pass 42.9 / 44.5 / 43.9%
  of eval-half cases against legacy's 43.0%. WP14's refit of every threshold on the calibration half is where
  that is absorbed.
- **L2 exactness is Mac-bound.** On Windows (CPU or CUDA) the strict L2 check fails on 13 cases: 4-decimal
  probabilities differ in the last digit (1e-4) and tied rows reorder; no case's code set changes. L2a's 1e-5
  tolerance was set on the Mac's CPU; L2b's tolerances are the cross-machine check.

**Done:**
- **WP0–WP9, WP11:** as before (L1/L2a on the Mac; `evaluate.py silver` reproduces 61.76%; every case is
  adopted or queued).
- **Windows line endings.** Every `ml/next` text and CSV writer pins LF, so Windows artefacts and gold snapshot
  hashes match the Mac's. A CRLF working copy of `ml/ICD_labels/labels.csv` (checked out before
  `.gitattributes` existed) had to be re-checked out for the pack to verify.
- **Manifest lineage + unique IDs.** `train.py` records `parents.silver_id`, `gold_train_snapshot` and
  `gold_train_codes` (from the labels table it trained on) and mints `generation_id = gen-<UTC timestamp>`
  (e.g. `gen-20260927T003905Z`) on every training run; calibrate keeps it.
- **WP10.** `generations.py status`, `promote.trigger_status`, `retrain_cycle.py` (recommend-only; heads-only
  unless `--backbone`; optional gold ingest; stops without gold-eval or without a met trigger unless
  `--force`). Promotion sets manifest status `archived` / `current` and moves the embedding-cache entries the
  new `current/` cannot use into the archive (listed in its manifest).
- **WP11 smoke run.** A bundle of `current/` is 538.8 MB (63 members, top-level folder `current/`); sha256
  sidecar and re-extracted verification pass. Deleted afterwards.
- **WP12.** `ml-worker` reads one verified bundle (root = parent of the model path; other path variables must
  point inside it), predicts through `report_mapping.inference.predict.predict_frame`, and adds
  `source_version` to each prediction. Dataset A's Text fills every section; uploads are read as UTF-8; the
  embedding length follows the generation (512, the legacy worker used 256). Dockerfiles copy the
  post-cutover `ml/` layout. Worker parity: `test_worker_parity.py` runs `batch_predict.py` in-process
  against `run_predict`. Backend change request: [ml-worker-change-request.md](ml-worker-change-request.md).
  `.gitignore`'s `ml-*/` also matches `ml-worker/`, so new files there need `git add -f` until it is excepted.

**Next:**
1. L4 (cold start, report only) on Windows.
2. WP16 docs (post-cutover layout; `training-guide.md` Step 8 grid), then the WP13 cutover (L1–L3 pass, worker parity test green; needs Opus sign-off).
3. WP14, then WP15, on Windows.

## Decisions (binding)

- **Clean-slate rewrite** of `ml/` code plus `ml-worker/`. Existing code is a reference, not a constraint.
  `backend/`, `database/`, `frontend/` are out of scope; the cloud is reached only through `handoff/` file
  contracts and change requests to the backend developer.
- **Design as if gold exists.** The full gold path is first-class code; tests use synthetic fixtures.
- **Keep the report-mapping model design**: per-section contrastive PetBERT backbone, concat-3 (2304-d),
  case-presence gate → group (+ tail gate) → per-group label-presence → keyword correction (+ Lipoma rescue).
  `--model` accepts any HF checkpoint dir/name and defaults to the current generation's `petbert/`.
- **Build beside, delete at parity.** New code lives in `ml/next/`, mirroring the final tree. Parity is
  measured on the OLD split, OLD silver and OLD threshold procedure. Cutover is one commit that deletes the
  old tree and `git mv ml/next/* ml/`.
- **Parity depth:** L1–L3 gate the delete; L4 (full cold start) runs once, is reported, and does not block.
- **Three-way split** train / calibration / test, created *after* cutover as its own generation.
  Calibration = the md5(case_id)%2==0 "sweep" half of the legacy test split; test = the other ("eval") half.
  Train is unchanged (no retrain for the split generation). Every threshold is fitted on calibration only;
  gold-eval comes from test only. The temporal holdout stays as a drift check.
- **`ARCHIVE_ROOT`** lives in `config.py`, used only by `generations/` to write archives. Nothing loads
  models or data from it.
- **Train into `candidate/`.** Promotion archives `current/` to `ARCHIVE_ROOT/YYYY-MM-DD_<desc>/` and swaps
  the candidate in; a losing candidate is deleted.
- **Promotion rule:** primary metric = weighted per-code exact accuracy (`good` share), G+S secondary.
  Promote only if (a) a retraining trigger is met and (b) the lower 95% bound of the paired
  (challenger − incumbent) difference, from a stratified case-cluster bootstrap on the *current* gold-eval,
  is ≥ −2.0 pp. The incumbent is re-scored on the same cases every time. The tool recommends; `--apply`
  promotes.
- **Gold stores are split.** Row-level Tier-3 audit judgements live in the audit store and never count as
  gold. Gold is case-level: one row per (case, code), or one `NO_CANCER` row, with a mandatory `origin`
  (`eval_batch` / `review_queue` / `random_slice`) that decides gold-eval vs gold-train. The existing
  `tier3_audit_*` sheets must still ingest.
- **Declined LLM answers** (`tier3_llm` "No Match") are decisive non-cancer. No switch.
- The first evaluation batch excludes the 198 cases already seen in the Tier-3 audit.
- **Cloud bundle:** send the backend developer a change request (download `checkpoints/label_presence/`,
  `checkpoints/thresholds.json`, `manifest.json`; update `docker-compose.yml` defaults). The worker refuses to
  start if a file listed in its manifest is missing.
- **Dropped:** `features/` (demographics, recency), the case-based / common-labels / top-n evaluators, the
  similarity and visualization debug outputs, `compare_llm_models.py`, `llm_pipeline/audit.py`, the broken
  `tests/test_text_filters.py`.
- **Approved edits outside `ml/`:** an `ml` pytest job in `.github/workflows/ci.yml`; `CLAUDE.md` archive
  wording and entry-point names (at cutover).

## Machines

| Machine | Does |
|---|---|
| **Mac (this one)** | All code changes. pytest. Parity L1 (scorer). Parity L2 part 1 (inference with old checkpoints + old embedding cache). Silver replay of the keyword tiers. |
| **Windows + CUDA** | All training and heavy embedding: L2 part 2 (re-embed from `report.csv`), L3 (3-seed head retrains), L4 (cold start), the split generation's threshold refit if it needs re-embedding, and the corrected-annotations generation. Runs from the committed code with the commands in the runbook below. |

`ml/.venv` on the Mac uses Python 3.12 (pyenv) and the CPU/MPS torch wheel; Windows uses the CUDA 12.8 wheel.

## Implementation conventions (every Part B agent follows these)

- **Import root is `ml/next/`**, with absolute imports (`import config`, `from taxonomy.taxonomy import …`).
  Nothing hardcodes "next" except the one `# removed at cutover` line in `config.py`.
- **`ml/next/config.py` is the only source of paths.** Add a constant there when a WP needs a new artefact.
  Legacy inputs use the `LEGACY_` prefix and are removed after cutover. New outputs never reuse an old path.
- **Tests:** pytest, run with `ml/.venv/bin/python -m pytest ml/next -q`. Synthetic data comes from
  `tests/fixtures.py` (factory functions) and `tests/conftest.py` (fixtures). Extend those files and don't
  duplicate them. Tests that import the old `ml/` tree for comparison say `LEGACY COMPARISON — deleted at
  cutover` in their module docstring. Tests never read `ml/data/` or `ml/output/`; real-data checks
  (parity, replays) are scripts that print counts only.
- **One package per WP.** A WP writes only inside its own package(s), its own tests, and new constants in
  `config.py`. If it needs something from another package, it uses that package's public functions.
- **Privacy:** never print, quote or copy report or diagnosis text. Headers, IDs, codes and counts only.
- **Style:** KISS per `CLAUDE.md`. Carry old logic over faithfully where the plan says "carry". Keep
  existing comments that still apply.
- No commits and no pushes. The main thread suggests commit messages to the user.

## Findings that shape the rewrite (verified)

- Production hyperparameters live only in doc commands. Code defaults differ (group trainer defaults 50
  epochs / dropout 0.3; production is 300 / lr 5e-5 / dropout 0.1 / wd 1e-3 / max-class-weight 50). Gate:
  20 epochs, recall-weight 0.7, pos_weight 1.0. LP: 25 epochs, 5 neg/pos, recall-weight 0.5, dropout 0.3,
  wd 1e-4, n_cols=3, per-section learned combine, case-disjoint GroupShuffleSplit. Backbone: 3 epochs,
  batch 32, lr 2e-5, temperature 0.07.
- Backbone `max_length` is 256 in training (`train_contrastive.py:150`) and 512 at inference (`cli.py:25`).
  Keep both.
- Inference thresholds: gate 0.80 and group 0.85 (only in `run_production.py` `set_defaults`); tail gate
  K=2, gap 0.08 (`cli.py` default); LP thresholds from `lp_thresholds.json` (0.5 fallback + argmax fallback).
- Uncommon merge: groups with < 200 cases, "Neoplasms, NOS" always forced in. Lipoma rescue is
  unconditional. Keyword correction = behaviour digit + subtype rules for 7 groups. The concat-3 section
  spec is defined twice (`pipeline.py:54-58`, `build_contrastive_dataset.py:21-25`).
- Seeds: gate, LP and split seeded 42; group trainer seeds numpy only; contrastive trainer unseeded. Parity
  tolerance must be measured across seeds.
- The embedding cache is keyed on CSV mtime ±1 s. The rewrite keys it on content hash.
- **Parity reference (supersedes the published 62.1%).** The published G+S of 62.1% (4,414 rows) and 62.3%
  (8,835 rows) comes from the 2026-05-13 generation and does not reproduce from today's files. Since then, the
  case-presence, group and some LP heads were retrained, and `annotation.csv` was rewritten on 2026-08-06.
  - The reference was therefore regenerated on 2026-09-25 with the old inference code, reading the current
    legacy checkpoints and embedding cache (CPU, `PYTHONHASHSEED=0`). It was scored with the old evaluator.
  - **md5 eval half** (G+S versus silver on the md5 eval half of `test_cases.txt`): G+S 61.8%, exact 61.76%,
    on 4,456 per-code rows. Good 45.8, Slight 16.0, CO 15.3, FP 2.6, FN 20.4.
  - **Full test:** G+S 61.8%, exact 61.81%, on 8,916 rows. Good 46.4, Slight 15.4, CO 15.0, FP 2.6, FN 20.6.
  - LP thresholds were fitted on the sweep half. The tail gate and the 0.80 gate were chosen on all of test,
    so the figure is somewhat optimistic.
  - The 2026-05-29 production CSV does not match the checkpoints now on disk:
    - 3,153 of 67,908 aligned rows differ in term, group or code, across 3,357 cases.
    - 1,106 cases flip at the case-presence gate.
    - `case_presence_prob` differs on 60,033 rows, by up to 0.76.
    - The two files have 68,800 and 69,517 rows.
- **Legacy inference is not deterministic across processes.** `stages/__init__.py:88-90` builds the Uncommon
  head's label list by iterating a `frozenset`, so its order follows Python's string-hash seed. Between hash
  seeds, about 280–330 of 68,800 rows change term, group, code or confidence. The probabilities do not change.
  The reference pins `PYTHONHASHSEED=0`. For L2a to be exact, WP4 must reproduce that order, or L2 must treat
  those rows separately.
- `ml-worker` is broken: `app.py:118` / `batch_predict.py:79` pass `embedding_min_sim` to `ScanConfig`
  (no such field); neither wires the LP heads; worker pins `numpy==1.26.4` vs `ml`'s `numpy>=2.0`.
- `ml/tests/test_text_filters.py` imports a deleted module; CI runs nothing in `ml/`.
- Splits: 46,652 train / 11,661 test; temporal 56,905 / 1,408. annotation.csv: 188,774 rows, 58,208 cases.
  Vague cases (Uncertain or tier3_no_candidates): 4,199 (3,385 train / 814 test).

## Target tree

```
ml/
  config.py            every path, incl. ARCHIVE_ROOT (written only by generations/)
  io_utils.py          the one shared CSV reader/writer (BOM, latin-1 reports / utf-8 outputs)
  taxonomy/            labels.csv; taxonomy.py (load, label texts, code↔term↔group);
                       behavior.py; subtype.py
  manual_audit/        sheets.py (review CSVs + sidecars, blind by construction);
                       tier3_audit.py (row-level sample/pilot/ingest → audit store);
                       eval_batch.py (blind stratified case-level batch from test + weight ledger);
                       gold.py (case-level ingest, mandatory origin, replace-by-case, snapshot hash);
                       cause_pass.py (misses sheet → cause store)
  diagnosis_mapping/   keyword_tiers.py (no_signal/tier1/tier2); llm_tier.py; llm_client.py (local only);
                       cleanup.py; silver.py (run → versioned silver generation); stats.py
  report_mapping/
    sections.py        the single concat-3 section spec + report-text builder
    model/             backbone.py (load, mean-pool embed, label embeddings);
                       heads.py (3 heads, state_dict-compatible with legacy .pt);
                       generation.py (load/save a generation dir; verify manifest + embedding fingerprint)
    training/          recipe.py (production hyperparameters + seeds); labels.py (targets from any labels
                       table); backbone.py; case_presence.py; group.py; label_presence.py;
                       calibrate.py (all thresholds, calibration partition only; legacy mode for L3);
                       oof.py (k-fold heads-only out-of-fold predictions)
    inference/         embedding_cache.py (content-hash keyed); stages.py; keyword_correction.py
                       (+ Lipoma rescue); predict.py
  coding/              rule.py (decisive/vague); adopt.py (gold > silver > bronze + provenance);
                       corrected.py (corrected annotations); queue.py (review queue, bronze priority)
  evaluation/          verdicts.py; intervals.py (Wilson, Kish n_eff, stratified case-cluster bootstrap,
                       single + paired); silver_eval.py; gold_eval.py (four results, representativeness);
                       audit_rates.py
  generations/         manifest.py; splits.py (legacy import, three-way, temporal); guards.py (check-split +
                       leakage); promote.py; triggers.py
  handoff/             contracts.py; imports.py; exports.py; worker_format.py (shared with ml-worker)
  scripts/             thin entry points
  tests/               pytest, synthetic fixtures only (tiny random-init BERT saved locally)
```

## Artefacts (none holds free text; text is always looked up from gitignored data)

- **Split** `splits/<split_id>/{train,calibration,test}_cases.txt` + `manifest.json` (method, seed,
  stratification, fractions, parent, sha256, counts). Legacy split imported as `legacy-80-20` (no calibration
  file; manifest records the md5-half rule and that the tail gate was fitted on test). Temporal split has its
  own `split_id`.
- **Silver generation** `silver/<silver_id>/annotation.csv` = old columns + `silver_generation`; manifest
  holds cascade version (git SHA + constants hash), LLM model ids, cleanup on/off, input sha256,
  `decision_stage` counts. Current `annotation.csv` imported unchanged as `silver-0-legacy`.
- **Audit store** (row-level, never gold): one row per (case_id, diagnosis_number): `decision_stage,
  sample_stratum, sample_weight, batch`, cascade code/term/group/method, `match_strength` (renamed from
  `tier`), `verdict` ∈ correct/wrong/no_cancer/uncertain, corrected code, `reviewer, reviewed_at, notes`.
- **Gold store** (case-level): one row per (case_id, code) or one `NO_CANCER` row; `term, group, origin`
  (mandatory), `batch_or_export_id, upload_period, slice_rate, reviewer, reviewed_at, source_sha256`.
  Re-ingest replaces a case's rows. One origin per case; `NO_CANCER` never beside a code.
- **Eval-batch ledger**: `case_id, batch_id, split_id, silver_id, seed, stratum, N_h, n_h, sample_weight,
  excluded_ledgers, review_mode` plus the series target parameters. `sample_weight` is per-batch; pooled
  weights across a series come from `eval_batch.pooled_weights` (N_h of the first frame / Σn_h).
- **Cause store**: `case_id, gold_code, method (silver|bronze), source_version, input_supports (yes/no),
  reviewer, reviewed_at`.
- **Labels table** (what every trainer takes): `case_id, matched_term, matched_group, matched_code`; empty
  term = no cancer. Old `annotation.csv` qualifies unchanged. Corrected annotations add
  `label_source (gold|silver), silver_generation, gold_snapshot`; train partition only; vague-without-gold
  excluded.
- **Adopted codes**: `case_id, code, term, group, code_source (manual|diagnosis|report), source_version,
  source_confidence, review_status (auto_accepted|queued|confirmed)`.
- **Review queue**: `case_id, reason (vague_silver|low_conf_bronze|random_slice|no_evidence), priority,
  partition, silver_generation, bronze_generation`. `no_evidence` (WP9 fix 2, lowest priority) covers a
  case in split ∪ silver ∪ bronze with no gold, no diagnosis row and no bronze prediction at all — the
  coverage gap that otherwise adopted nothing and queued nothing.
- **Predictions**: old columns + `generation_id`.
- **Report-mapping generation** (`current/`, `candidate/`) — its layout *is* the cloud bundle layout:
  `petbert/`, `labels/labels.csv`, `checkpoints/{case_presence_classifier.pt, group_classifier_best.pt,
  label_presence/*.pt, lp_thresholds.json, thresholds.json, uncommon_groups.txt}`, `manifest.json`
  (generation_id `gen-<UTC timestamp>`; parents {silver_id, gold_train_snapshot, gold_train_codes, split_id}; resolved recipe + seeds; device and
  library versions; embedding fingerprint (backbone sha, section-spec version, max_length); calibration
  {partition, objective, values}; file sha256s; scores; status). The embedding cache sits beside it, never
  bundled.
- **Handoff**: inbox `pending_diagnoses_<export>.csv`, `gold_<export>.csv`; outbox
  `silver_codes_<silver_id>.csv`, `adopted_codes_<run>.csv`, `review_queue_<run>.csv`, bundle tarball; each
  with a sidecar manifest (schema_version, sha256).

## Entry points (`scripts/`)

| Script | Replaces | Does |
|---|---|---|
| `map_diagnoses.py run\|stats` | run_annotation, run_annotation_cleanup, run_data_analysis | Silver generation; coverage stats |
| `audit.py tier3-sample\|tier3-pilot\|tier3-ingest\|eval-batch\|ingest-gold\|cause-sheet\|ingest-cause` | run_gold_annotation | Specialist sheets and stores |
| `split.py create\|import-legacy\|check` | create_split, check_split | Split generations; `check` = all leakage guards |
| `code_cases.py adopt\|corrected\|queue` | — | Coding rule outputs |
| `train.py --stage backbone\|case-presence\|group\|label-presence\|heads\|oof` | run_training | Recipe defaults = production |
| `calibrate.py --generation --partition calibration` | sweep_lp_thresholds, sweep_tail_gate | All thresholds, in-process |
| `predict.py --generation current [--model …] [--embed-only]` | run_production | Stamped predictions |
| `evaluate.py silver\|gold\|audit-rates` | run_evaluation | Verdicts, four results, CIs |
| `promote.py [--apply]`, `generations.py status` | — | Paired rule, archive + swap; trigger status |
| `handoff.py import-pending\|import-gold\|export-silver\|export-coding\|export-bundle` | — | Cloud file contracts |
| `retrain_cycle.py` | — | The strategy's local lane in one go (recommend-only) |
| `parity.py` (temporary) | — | L1–L4 comparisons; deleted at cutover |

`split.py check` runs inside `train`, `calibrate`, `evaluate gold` and `promote` and refuses on any violation.

## Old → new

| Old | Verdict | New home |
|---|---|---|
| `ICD_labels/taxonomy.py`, `labels.csv`, `projection.py` | carry / merge | `taxonomy/taxonomy.py`, `taxonomy/labels.csv` |
| `ICD_labels/catalog.py` | split | label texts → `taxonomy/`; label embeddings → `report_mapping/model/backbone.py` |
| `ICD_labels/behavior_keywords.py`, `subtype_keywords.py` | carry | `taxonomy/behavior.py`, `taxonomy/subtype.py` |
| `llm_pipeline/pipeline.py` | rewrite | `diagnosis_mapping/keyword_tiers.py`, `llm_tier.py`, `silver.py` |
| `llm_pipeline/client.py`, `cleanup.py` | carry (fix reach-through) | `diagnosis_mapping/llm_client.py`, `cleanup.py` |
| `llm_pipeline/cli.py`, `run_annotation_cleanup.py` | rewrite | `scripts/map_diagnoses.py` |
| `llm_pipeline/audit.py`, `compare_llm_models.py`, `annotation/__init__.py` | drop | — |
| `gold/sample.py`, `pilot.py`, `csv_io.py` | carry | `manual_audit/tier3_audit.py`, `sheets.py` |
| `gold/ingest.py` | rewrite | `manual_audit/tier3_audit.py`; case-level gold new in `manual_audit/gold.py` |
| `gold/check_split.py`, `training/data/create_split.py` | rewrite | `generations/guards.py`, `generations/splits.py` |
| `gold/rates.py` | rewrite (+ CIs) | `evaluation/audit_rates.py` |
| `training/{binary,group,label_presence,contrastive}/*` | rewrite (seed all RNGs) | `report_mapping/training/*` |
| `production/petbert_pipeline/{pipeline,types,io}.py` | rewrite | `report_mapping/inference/predict.py` |
| `production/.../stages/*` | rewrite (behaviour-equivalent) | `report_mapping/inference/stages.py`, `keyword_correction.py` |
| `production/.../embedding.py`, `utils.py` | carry / split | `report_mapping/model/backbone.py`, `report_mapping/sections.py` |
| `production/.../embedding_cache.py` | rewrite (content hash) | `report_mapping/inference/embedding_cache.py` |
| `production/.../cli.py`, `__main__.py`, `README.md` | drop | `scripts/predict.py` |
| `model/*.py`, `constants.py` | carry (state_dict names kept) | `report_mapping/model/heads.py` |
| `evaluation/evaluate.py`, `common.py`, `log_evaluation.py` | carry core | `evaluation/verdicts.py`, `silver_eval.py` |
| `evaluate_case_based.py`, `evaluate_common_labels.py`, `evaluate_top_n_verdicts.py` | drop | — |
| `evaluate_case_presence.py`, `evaluate_groups.py`, `evaluate_label_presence.py` | rewrite | diagnostics in `report_mapping/training/calibrate.py` |
| `features/*` | drop | — |
| `analysis/annotation_stats.py` | carry | `diagnosis_mapping/stats.py` |
| `utils/csv_io.py`, `encoding.py` | carry | `io_utils.py`; `safe_filename` → `report_mapping/model/generation.py` |
| `tests/test_text_filters.py` | drop (broken) | — |
| `config.py`, `requirements.txt` | rewrite | new `config.py`; one pin set shared with the worker (+ pytest) |
| `ml-worker/app.py`, `batch_predict.py`, `Dockerfile*`, `requirements.txt` | rewrite (thin) | import `report_mapping.inference` + `handoff.worker_format` |

## Work packages

Model roles: Sonnet implements, a separate Sonnet verifies, Haiku runs, Opus evaluates — except where
marked, where Opus implements because errors would be silent.

| WP | What | Implement / Verify | Runs on |
|---|---|---|---|
| **WP0** Env + test scaffold | `ml/.venv` (py3.12), pytest config, fixture factory (tiny random-init BERT, random heads, synthetic reports/diagnoses/labels), `ml` job in CI | Sonnet / Sonnet | Mac |
| **WP1** Legacy reference + parity harness | Reproduce 62.1 with old code; freeze a private reference pack (md5 half lists, old predictions/verdicts, thresholds, checkpoint hashes); `parity.py` L1–L4. L1: identical verdicts. L2: identical codes per row, probs within 1e-5; re-embed: cosine ≥ 0.9999, ≥ 99.5% rows identical, eval-half G+S within ±0.2 pp. L3: 3 seeds, mean G+S within max(1.0 pp, 2·sd) of reference, each verdict share within ±1.5 pp, groups (≥ 50 codes) losing > 5 pp listed | **Opus** / Sonnet | Mac (L1, L2a); Windows (L2b, L3, L4) |
| **WP2** config, io_utils, taxonomy | All paths incl. `ARCHIVE_ROOT`; taxonomy ported incl. the labels.csv title-row quirk | Sonnet / Sonnet | Mac |
| **WP3** generations: manifest, splits, guards | Legacy split import byte-identical; three-way + temporal creation; guards (partition disjointness/coverage, gold-eval ⊆ test, eval-side queue gold never trains, labels ∩ (calibration ∪ test) = ∅, calibration reads calibration only, audit store never feeds corrected annotations) | **Opus** (splits, guards), Sonnet (manifest) / Sonnet | Mac |
| **WP4** sections, model, inference | One section spec; embed at 512; heads load legacy `.pt`; content-hash cache + one-time importer of the old cache; stages with exact current semantics; thresholds from the generation's `thresholds.json`; predictions carry `generation_id`; gen-0 = *copy* of old checkpoints into `current/` with backfilled manifest; loader refuses mismatched embedding fingerprint | Sonnet / Sonnet | Mac (L2a); Windows (L2b) |
| **WP5** training + calibrate | `recipe.py` pins production hyperparameters; all RNGs seeded; trainers take any labels table and write `candidate/`; `calibrate` (per-LP F1, then gate/group/tail; per-code G+S objective; small fixed grid; legacy mode = LP on md5 sweep half, rest fixed); `oof.py` | Sonnet (trainers), **Opus** (calibrate) / Sonnet | Mac (tests); Windows (L3, L4) |
| **WP6** evaluation | `verdicts` exact port; `intervals`; `silver_eval`; `gold_eval` (four results, weighted, CIs, representativeness); `audit_rates` | **Opus** / Sonnet | Mac |
| **WP7** manual_audit | Tier-3 sample/pilot/ingest (old sheets ingest unchanged); `eval_batch`; `gold`; `cause_pass` | Sonnet / **Opus** | Mac |
| **WP8** diagnosis_mapping | Port cascade, local LLM tier, cleanup; `silver.py`; import `silver-0-legacy`; replay no_signal/tier1/tier2 vs old annotation.csv | Sonnet / Sonnet | Mac |
| **WP9** coding | Vagueness table; adoption rule; corrected annotations; review queue | Sonnet / **Opus** | Mac |
| **WP10** promote, triggers, retrain_cycle | Promotion rule above; archive + swap; triggers (new silver lineage, random-slice CI non-overlap, ≥ 200 new gold-train codes) | **Opus** / Sonnet | Mac |
| **WP11** handoff | Schemas + versions; imports (origin mandatory, later export replaces); exports; bundle tarball + sha256 | Sonnet / Sonnet | Mac |
| **WP12** ml-worker | Keep env-var contract; resolve bundle root; verify manifest; wire LP heads + `thresholds.json`; fix the `TypeError`; import shared code; response schema + `source_version`; aligned pins; Dockerfiles copy `taxonomy/`, `report_mapping/`, `handoff/`; backend change request written | Sonnet / Sonnet | Mac |
| **WP13** Parity gate + cutover | Preconditions: L1–L3 pass, worker parity, green suite, Opus sign-off. One commit deletes the old tree and moves `ml/next/*` into `ml/`; old checkpoints archived; gen-0 becomes `current/`; CLAUDE.md edits | Sonnet / Sonnet, Opus approves | Mac |
| **WP14** Three-way split generation | Create split; refit all thresholds on calibration; silver-eval on test; `promote` (recommend) | Haiku runs / Opus evaluates | Windows |
| **WP15** Corrected-annotations generation | `code_cases corrected` → retrain → calibrate → `evaluate gold` vs silver-only incumbent; OOF disagreement analysis | Haiku runs / Opus evaluates | Windows |
| **WP16** Docs | README rewrite; per-package docs; 62.1% caveat; archive `annotation-redesign-plan.md` after moving its locked decisions; strategy-doc updates | Sonnet / Sonnet | Mac |

## Order

```
Wave 0:  WP0 ‖ WP2
Wave 1:  WP1 (reference + harness) ‖ WP3 ‖ WP6a (verdicts, intervals → L1) ‖ WP7 ‖ WP8
Wave 2:  WP4 (→ L2a) ‖ WP9 ‖ WP6b (silver_eval, gold_eval)
Wave 3:  WP5 ‖ WP10 ‖ WP11
Wave 4:  WP12
— Windows: L2b, L3, L4 —
Wave 5:  WP13 cutover (after L1–L3 pass)
— Windows: WP14, then WP15 —
WP16 docs alongside each WP
```

## Notes carried between work packages

- **WP3 → WP5.** Trainers filter silver down to the train partition in memory. They must run
  `guards.check_labels_train_only` and `guards.check_eval_queue_gold_not_trained` on the *filtered* frame. Don't
  pass them the whole-universe silver file, which fails by design.
- **WP3 → WP5.** Legacy splits have no calibration partition. Calibrate's legacy mode (LP on the md5 sweep
  half) uses `three-way-v1`: its train is byte-identical to legacy, and its calibration is exactly the sweep half.
- **WP3 → WP9.** `check_corrected_sources` can only confirm that gold-labelled cases have a gold-store row. The
  corrected-annotations builder must itself ensure that audit-store codes never enter silver rows.
- **WP3.** Split directories are write-once: `import-legacy`/`create` refuse to overwrite. `load_split` verifies
  the manifest first. The legacy temporal cutoff year isn't recorded anywhere (`cutoff_year: null`).
- **Real split counts.** `legacy-80-20` 46,652 / 11,661. `legacy-temporal` 56,905 / 1,408. `three-way-v1`
  46,652 / 5,828 calibration / 5,833 test.

## Parity runbook

`ml/next/scripts/parity.py` compares files only, and every level exits 1 on FAIL. It prints counts,
percentages, case IDs and group names, never text. Each level first verifies the reference pack at
`config.PARITY_REFERENCE_DIR` (`ml/output/parity_reference/`) against its manifest, and checks that the
legacy `annotation.csv` and `test_cases.txt` are unchanged since the freeze. Scoring always uses the eval half
(md5 rule) and the frozen `uncommon_groups.txt`. G+S is the exact row share, not round(good) + round(slight).

**Status: frozen 2026-09-25, re-frozen 2026-09-26.** The re-freeze added the sha256 of the old
`ICD_labels/labels.csv` and keys every legacy hash by its path relative to `ml/`. All 12 pack files and all
results are byte-identical to the first freeze. Every level re-verifies these files: `annotation.csv`,
`test_cases.txt`, `report.csv`, `uncommon_groups.txt`, `lp_thresholds.json` and `labels.csv`. It also refuses
a pack that contains unlisted files. The pack's `legacy_predictions.csv` was regenerated by the old
`run_production.py`, driven read-only by `parity/legacy_predict.py`:
- It uses production defaults, the current legacy checkpoints and the legacy embedding cache, on CPU with
  `PYTHONHASHSEED=0`.
- Only the `labels.csv` mtime check is bypassed. Writing the cache is refused, and the text-bearing writers
  are disabled.
- The 2026-05-29 production CSV is kept as `legacy_predictions_2026-05-29.csv` for the record.

Reference scores: eval half 61.8% on 4,456 rows; full test 61.8% on 8,916 rows (see Findings). The published
62.1% / 4,414 is superseded.

**Mac** (`ml/.venv/bin/python`, from the repo root):

```bash
# Once: build the pack using the old evaluator (it refuses if the pack is already frozen)
ml/.venv/bin/python ml/next/scripts/parity.py freeze

# L1 — the new scorer vs the legacy verdict tables (eval half and full test)
ml/.venv/bin/python ml/next/scripts/parity.py l1

# L2a — gen-0 inference (old checkpoints + imported old embedding cache) vs the legacy predictions
ml/.venv/bin/python ml/next/scripts/predict.py --generation current                    # (available after WP4)
ml/.venv/bin/python ml/next/scripts/parity.py l2 --predictions <gen-0 predictions.csv>
```

**Windows + CUDA** (PowerShell from the repo root). Code arrives through git (push on the Mac, pull on Windows);
the reference pack, `report.csv` and the embedding cache arrive through Syncthing. `ml/output/report_mapping/candidate`
must not exist before L3, and L3 must run before L4.

```powershell
# L2b: re-embed report.csv into ml\l2b_cache (outside Syncthing, so the shared cache is never read or overwritten)
ml\.venv\Scripts\python.exe ml\next\scripts\predict.py --generation current --device cuda --embed-only --cache-dir ml\l2b_cache
$emb = (Get-ChildItem ml\l2b_cache\*.npz | Select-Object -First 1).FullName
New-Item -ItemType Directory -Force -Path ml\output\predictions\parity | Out-Null
ml\.venv\Scripts\python.exe ml\next\scripts\predict.py --generation current --device cuda --cache-dir ml\l2b_cache --out ml\output\predictions\parity\l2b_predictions.csv
ml\.venv\Scripts\python.exe ml\next\scripts\parity.py l2 --reembedded --predictions ml\output\predictions\parity\l2b_predictions.csv --embeddings $emb

# L3: heads retrained with 3 seeds, legacy calibration (LP thresholds only), one predictions CSV per seed.
# Heads-only training reuses current's petbert/, so it hits the shared cache and never re-embeds.
foreach ($seed in 1,2,3) {
    ml\.venv\Scripts\python.exe ml\next\scripts\train.py --stage heads --labels silver-0-legacy --split three-way-v1 --seed $seed --device cuda --out candidate --local-only
    ml\.venv\Scripts\python.exe ml\next\scripts\calibrate.py --generation candidate --labels silver-0-legacy --split three-way-v1 --legacy
    ml\.venv\Scripts\python.exe ml\next\scripts\predict.py --generation candidate --device cuda --out ml\output\predictions\parity\l3_seed$seed.csv
}
ml\.venv\Scripts\python.exe ml\next\scripts\parity.py l3 --predictions ml\output\predictions\parity\l3_seed1.csv ml\output\predictions\parity\l3_seed2.csv ml\output\predictions\parity\l3_seed3.csv

# L4: cold start (backbone + heads), report only. The new backbone writes a new ~1 GB entry to the shared cache.
Remove-Item -Recurse -Force ml\output\report_mapping\candidate
ml\.venv\Scripts\python.exe ml\next\scripts\train.py --stage backbone --labels silver-0-legacy --split three-way-v1 --device cuda --out candidate --local-only
ml\.venv\Scripts\python.exe ml\next\scripts\train.py --stage heads --labels silver-0-legacy --split three-way-v1 --device cuda --out candidate --local-only
ml\.venv\Scripts\python.exe ml\next\scripts\calibrate.py --generation candidate --labels silver-0-legacy --split three-way-v1 --legacy
ml\.venv\Scripts\python.exe ml\next\scripts\predict.py --generation candidate --device cuda --out ml\output\predictions\parity\l4_predictions.csv
ml\.venv\Scripts\python.exe ml\next\scripts\parity.py l4 --predictions ml\output\predictions\parity\l4_predictions.csv
```

L2 passes when term, group and code are identical per row and the probabilities are within 1e-5. Rows are
keyed by (case_id, diagnosis_index), and a duplicate key fails.

L2b passes when every case has cosine ≥ 0.9999, ≥ 99.5% of rows are identical, and eval-half G+S is within
±0.2 pp. It compares rows on term, group and code only, because probabilities may drift after re-embedding.
Duplicate keys fail here too.

**Uncommon-reordered cases, in both L2 and L2b:** legacy picks among the passing labels of the merged
Uncommon pool in hash order, while the rewrite ranks them by confidence. The model scores are the same; only
the order of a case's rows changes.
- A case qualifies when every one of its differing rows is in a group from the frozen `uncommon_groups.txt`,
  on both sides.
- A qualifying case is compared as a multiset of its rows instead of row by row:
  - In L2, each row is (term, group, code, case_presence_prob, group_prob, confidence), with probabilities
    within 1e-5.
  - In L2b, each row is (term, group, code).
- A case whose multisets match is counted as "uncommon-reordered" and does not fail. In L2b its rows count as
  identical.
- Any other difference fails.
- Validated on real data: a legacy re-run with `PYTHONHASHSEED=1` differs from the seed-0 reference in 279
  rows across 98 cases. All 98 were uncommon-reordered, so L2 passes.
L3 passes when mean G+S is within max(1.0 pp, 2·sd) and each verdict share is within ±1.5 pp. Groups with
≥ 50 eval-half codes that lost more than 5 pp are listed but do not fail. The per-group count uses each row's
expected group(s).

Never run the old `run_production.py` directly on the Mac; `freeze` drives it through `legacy_predict.py`
instead. `labels.csv` has a newer mtime than the one stored in the legacy cache, even though its content is
unchanged. A direct run would therefore miss the cache, re-embed everything and overwrite
`ml/output/training/embedding_cache.npz`.
- **Encoding.** `io_utils.read_csv(path, encoding="latin-1")`: raw inputs (`report.csv`, `diagnoses.csv`) are
  latin-1, and everything the rewrite writes (stores, silver, labels tables, predictions) is utf-8. Read it with
  `encoding="utf-8"`.
- **Config defaults.** Never write `config.X` as a default argument value, because it binds at import time.
  Default to `None` and resolve `config.X` inside the function.

## Decisions added during Part B (2026-09-26)

- **Parity reference.** Regenerated on this Mac with the OLD inference code on the current checkpoints and cache
  (CPU, `PYTHONHASHSEED=0`), then frozen. The 05-29 predictions CSV no longer matched the checkpoints on disk.
- **Uncommon ranking fix.** Legacy picked the first LP survivor in the merged Uncommon pool in hash order. The
  new code ranks survivors by confidence, breaking ties by term. Model scores and all verdicts are unchanged. L2
  compares Uncommon-group rows per case without regard to order.
- **Eval-batch allocation is code-targeted.** About 30 gold codes per major group (≥1% of codes), a fixed
  60–120 no-cancer cases, and 1–2 per rare group. About 865 cases in 2–3 batches of about 400.
  "Specialized gonadal neoplasms" can't reach 30 codes from the test partition alone.
- **Gold is collected as taxonomy terms** (`term_1..term_5`); code and group are derived from the term. Codes
  alone are ambiguous for 186 of 534 codes.
- **Eval batches are reviewed in the registry app, which shows predictions (not blind).** Accepted by the user.
  The ledger records the review mode, and gold-eval reports state that the batch was non-blind, so accuracy may
  be optimistic because of anchoring.
- **WP7 → WP6b.** Gold-eval reports join the ledger to show `review_mode` (every batch is `app_non_blind`),
  and weight cases with `eval_batch.pooled_weights`, never by summing per-batch `sample_weight`.
- **WP7 → WP9.** The review queue must skip cases that already have gold. `gold_train()` rows always carry a
  resolved term, because ambiguous code-only rows are refused.
- **Eval-batch schedule.** Batch 1 at `--fraction 0.5` (~417 cases, about ±5.8 pp). Batch 2 at `--fraction 1.0`
  (~409 cases). Series total: ±4.1 pp, with 23 of 24 major groups at 30 or more codes.
- **LP threshold grid.** The production LP thresholds came from a 0.05 grid. That grid reproduces 25 of 25; the
  documented 0.01 grid reproduces only 12. `calibrate.LP_GRID` uses 0.05. WP16 must correct
  `training-guide.md` Step 8.
- **Calibration on three-way-v1 (gen-0, for information only).** Fitting every threshold picks gate 0.80,
  group 0.90, and tail K=2 with gap 0.05. That gives 62.29% G+S on calibration, against 61.86% at the legacy
  point. Measured on the same partition it was fitted on.
- **Gate/group training set.** Gate, group and OOF training use only train cases that have label rows. Vague
  cases dropped from the corrected annotations are no longer silent negatives. Under `silver-0-legacy` this drops
  the 80 no-diagnosis train cases (46,652 → 46,572), a known small difference for L3.
- **Cleanup pairs in the coding rule.** A cleanup-rewritten "No Match" is decisive non-cancer; a cleanup
  "Uncertain" is vague.
- **Coverage invariant.** Every case is either adopted or queued. Cases with no evidence at all (11 today: no
  diagnosis, empty report, no bronze row) are queued as `no_evidence` at the lowest priority.
- **Queue order.** Test-partition items come after all train and calibration items. Reviewing a test case
  removes it from the gold-eval frame, and the vague cases would otherwise leave gold-eval first.
- **Bronze gate is per row**, as in the backend: any row rejected by case presence or below 0.23 confidence
  queues the case. The margin check stays on rank 1.
- **LLM prompts stay byte-identical to legacy**, em-dashes and the odd hyphens in the cleanup prompt included.
  `test_llm_tier.py` pins them.
- **Representativeness counts all silver codes in test.** By that count "Specialized gonadal neoplasms" reaches
  41 codes; the eval-batch allocation targets strata by own-group share, so criterion 2 may still report a
  shortfall for gonadal and paragangliomas. The report states it; nothing is changed to hide it.
