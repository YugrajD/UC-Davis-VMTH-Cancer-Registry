# Bronze Restructure Plan — one jointly trained ModernBERT replaces the PetBERT four-stage pipeline

**Status:** Approved 2026-10-09 ("execute"). Branch `bronze-retrieve-rerank`; the pre-restructure state is tagged
`pre-restructure-2026-10-09` (local tag) and `main` keeps the old pipeline until merge. This is the working contract
for the Part B agent team. Agents append to **Notes carried between work packages** at the bottom; nothing else in
this file changes without the user.

## Why

The retrieve-and-rerank v2 experiment (2026-09-30 to 2026-10-08) ended with a jointly trained BioClinical
ModernBERT (term head + neoplasm gate head over the whole report) that beat the WP14 four-stage generation on the
one-shot `three-way-v1` test against pre-registered bars:

| | Good | Slight | CO | FP | FN | G+S |
|---|---|---|---|---|---|---|
| new (`rr-20261008T210937Z`) | 58.47 | 10.60 | 10.92 | 2.14 | 17.87 | 69.07 |
| WP14 (`gen-20260927T025939Z`) | 46.27 | 15.86 | 12.84 | 2.35 | 22.68 | 62.13 |

Paired bootstrap (new − WP14): Good +12.20 [+10.91, +13.60], Slight −5.26, CO −1.92, FP −0.21 [−0.82, +0.32],
FN −4.80, G+S +6.94 [+5.53, +8.48]. Bars: Good lower bound ≥ −2 pp, G+S ≥ WP14, FP ≤ WP14 + 1 pp — all pass.
The test partition is now **spent** for this design: never tune on it.

This is an **architectural change**, not a promotion: the process inside `ml/` is restructured around the new model.

## Decisions (binding)

1. **Two kinds of model improvement.**
   - *Retraining* (same code, new weights) goes through `promote.py` and is judged on **gold only** (paired
     case-cluster bootstrap on gold-eval, Good lower bound ≥ −2 pp, at least one trigger). No silver fallback:
     without gold-eval rows it refuses. `retrain_cycle.py --apply` (opt-in) promotes automatically when the rule
     passes; without the flag the cycle only recommends.
   - *Architectural change* (new code) is adopted manually: `promote.py` adopt mode requires `--reason <path>`
     pointing at an existing file under `ml/documentation/history/`; it records the path in the manifest, archives
     `current/` whole to `ARCHIVE_ROOT/YYYY-MM-DD_<description>/` and swaps. `promote.py` refuses to *promote* a
     candidate whose `architecture` block differs from current's and points to adopt instead.
   - This restructure is the first architectural adopt; its reason is ADR 0009.
2. **Package name stays `ml/report_mapping/`**, rebuilt from scratch. Deleted: the PetBERT four-stage code, the
   embedding cache, the embedding fingerprint, `ml/bronze_rr/`, `ml/experiments/`, `ml/scripts/rr_phase1.py`,
   `ml/requirements-rr.txt` (merged into `ml/requirements.txt`). Git history and the tag keep them.
3. **First generation = the frozen `rr-20261008T210937Z` weights, re-housed, not retrained.** They were trained on
   raw silver `silver-0-legacy` (vague cases included); retrains use the corrected-annotations table. Recorded in
   ADR 0009.
4. **Every retrain starts from the base checkpoint** (`thomas-sounack/BioClinical-ModernBERT-base`) with a fresh
   head over the train-partition terms. Thresholds are re-fitted after every training run. Warm start is not built.
5. **Report text.** The 8 `report.csv` columns in order (CLINICAL ABSTRACT, GROSS DESCRIPTION, HISTOPATHOLOGICAL
   SUMMARY, COMMENT, FINAL COMMENT, ANCILLARY TESTS, ADDENDUM, ADDITIONAL INFORMATION), each non-empty one as
   `"HEADING: text"` after `clean_text`, joined by `\n`; a missing column counts as empty. Must be byte-identical
   to `bronze_rr.report_text.build_report_text` for present columns. **A case has text iff any of the 8 columns
   is non-empty** (replaces the PetBERT three-section rule; the case-count delta is reported on calibration).
6. **Decision rule (unchanged).** gate < `gate_t` → one `Non-Cancer` row, method `rejected_by_case_presence`,
   confidence `0.00`; else rank-1 term (+ rank-2 iff p2 ≥ `tau` and p2/p1 ≥ `rho`), method **`term_head`**
   (renamed from `retriever_top1`). Predictions CSV keeps its 10 columns in order; `group_prob` stays blank.
   Thresholds fitted on the calibration partition with the FP cap `config.FP_CAP = 0.0243` (WP14's calibration
   FP share; `--max-fp-share` overrides). Frozen values: gate_t 0.75, tau 0.10, rho 0.25.
7. **Review rule (bronze-only cases).** Queue every gate-rejected case, and every accepted case whose rank-1
   confidence is below `review_cut`. The margin rule is dropped; `unidentified_cancer` is removed everywhere.
   `review_cut` is fitted per generation at calibration time and stored in `thresholds.json`: the cut that queues
   `config.REVIEW_TARGET_SHARE` of accepted calibration cases (0.2336 = WP14's calibration volume, 592 of 2,534
   accepted cases queued by the old 0.23/0.15 rule). Queue priority stays the gate probability. A case where the
   gate accepts but no term is confident (the likely out-of-vocabulary case: 455 of 844 terms are predictable)
   is caught by this cut.
8. **Report-Mapping audit = 5-fold cross-fit** on the train partition (`training/crossfit.py`, `scripts/oof.py`),
   numbers-only output in `config.OOF_DIR`; the sampler contradicts on the gate probability as before and gains
   term-level probabilities. Run once per silver lineage or retrain (~4–5 h GPU).
9. **Uncommon-groups list is a scoring-rule (ruler) file**, not a generation file: WP14's 24 groups pinned at
   `config.UNCOMMON_GROUPS_TXT` (committed; group names only). `silver_eval` keeps the generation-id check.
10. **Serving.** `ml-worker/` is changed and every change recorded in `ml-worker/CHANGES.md`. The backend keeps
    its contract: it sets `MODEL_PATH=<bundle>/petbert`; the worker only uses the parent directory. Nothing outside
    `ml/` and `ml-worker/` is edited; required changes for other owners go to
    `ml/documentation/restructure-change-request.md` (ID, owner, file, change, why, blocking, status).
11. **Pins.** `ml/requirements.txt` and `ml-worker/requirements.txt` move to transformers 5.18.0 / tokenizers 0.23.2
    / huggingface_hub 1.33.0 / safetensors 0.8.0, keeping torch 2.9.1 if parity holds (CI hard-codes torch 2.9.1 and
    lives outside `ml/`). If torch 2.9.1 fails, the CI change goes into the change request.
12. **Upload text mapping** (`handoff/worker_format.upload_to_reports`) is chosen by measurement on calibration
    (single `Text` under one heading vs flat), recorded in the notes below.

## Target tree

```
ml/report_mapping/
  __init__.py
  text.py            pandas only: REPORT_COLUMNS, TEXT_SPEC_VERSION, clean_text, build_report_text(s), has_text
  batching.py        length-bucketed steps, token-budget micro-batches, encode_reports
  model.py           ENCODER_ID, new_encoder, load_tokenizer, JointClassifier (term + gate heads, fp32), save/load
  generation.py      stdlib only: layout names, ARCHITECTURE, manifest build/verify, verify_spec, resolve_generation_dir,
                     new_generation_id, read_thresholds, GenerationError
  inference/
    scoring.py       logits_for (bf16 autocast on CUDA only), RankedScores, sigmoid, TOP_K
    decide.py        Thresholds, COLUMNS, case_rows, decide_all
    predict.py       load_generation -> LoadedGeneration, predict_frame (in memory; the worker), run_predict (report.csv -> CSV)
  training/
    recipe.py        Recipe (frozen values), PRODUCTION_SEED = 42, seed_all
    labels.py        load_labels_table, lineage, select_train, training_case_ids, cancer_case_ids, gate_targets
    data.py          TrainingData from a labels table + split: vocab, multi-hot targets, gate targets, holdout
    loop.py          fit (joint loss, AdamW, warmup, clip, bf16 autocast, grad checkpointing)
    train.py         train() -> candidate generation with calibration status "pending"
    calibrate.py     pattern counts, fit_thresholds (FP cap), fit_review_cut, calibrate() -> thresholds.json + manifest block
    crossfit.py      k-fold cross-fit over the train partition -> numbers-only OOF CSV
```

Import-light rule: `text`, `batching`, `model`, `generation` and `inference/*` never import `diagnosis_mapping`,
`evaluation`, `generations.guards`, `scipy` or `sklearn`. A test asserts this via `sys.modules`.

## Generation layout and manifest

```
<generation>/
  encoder/            HF ModernBERT weights (model.safetensors) + tokenizer files + config.json
  head.pt             term head (fp32)
  gate_head.pt        gate head (fp32)
  terms.json          term vocabulary, head order
  term_info.json      term -> [group, code] (snapshot of ml/taxonomy/labels.csv, first occurrence)
  thresholds.json     gate_t, tau, rho, review_cut, max_fp_share, review_target_share
  manifest.json
```

Manifest: `kind` (`report_mapping_generation`), `schema_version` 2, `generation_id` (`gen-<UTC>`), `status`,
`architecture` {name `joint-modernbert-v1`, text_spec_version, pooling `mean`, heads, decision_rule
`gate-top1-rank2-v1`}, `model` {encoder_id, weights_sha256 (over encoder/model.safetensors, head.pt, gate_head.pt,
terms.json), n_terms, max_length 4096}, `recipe`, `parents` {split_id, labels_source, silver_id,
gold_train_snapshot, gold_train_codes}, `training` {seed, device, library_versions, n_train, n_holdout, holdout
metrics}, `calibration` {status pending|calibrated, partition, split_id, labels, n_cases, objective, grid, values,
max_fp_share, weights_sha256, in-sample verdicts}, `adoption` {reason, adopted_at} (adopt only), plus the
`generations.manifest` fields (`created_at`, `git_sha`, `files`).

Spec check (`verify_spec`, run by every loader, promote, export and S3 publish): file hashes match; the
text spec equals the code's; `len(terms) == head.out_features` (checked at model load); calibration is
`calibrated` and bound to the same `weights_sha256`. Recalibration (new thresholds, same weights) creates a new
generation id (`generations.py recalibrate`, replaces `fork`).

## Entry points (thin)

| Script | Job |
|---|---|
| `train.py` | One step: labels + split → candidate generation (pending). `--labels --split --seed --device --model --out --limit --max-steps` |
| `calibrate.py` | Fit gate_t/tau/rho (FP cap) and review_cut on calibration. `--generation --labels --split --max-fp-share --device` |
| `predict.py` | `--generation --device --out` → `output/predictions/<generation_id>_predictions.csv` |
| `evaluate.py` | Unchanged CLI; ruler reads `config.UNCOMMON_GROUPS_TXT` |
| `promote.py` | Recommend / `--apply` (gold rule, same architecture) / adopt with `--reason` |
| `retrain_cycle.py` | As today with one training step; `--apply` opt-in |
| `generations.py` | `status`, `recalibrate` |
| `oof.py` | 5-fold cross-fit → `config.OOF_DIR` |
| `audit.py`, `code_cases.py`, `handoff.py`, `split.py`, `sync.py`, `map_diagnoses.py` | Unchanged CLIs, rewired internals |

## Acceptance checks (gate the adopt)

| Check | Bar |
|---|---|
| R1 | Rebuilt GPU `run_predict` on the re-housed generation reproduces `ml/output/predictions/rr-20261008T210937Z_predictions.csv` for every case in it (all columns; `method` mapped `retriever_top1`→`term_head`; `generation_id` differs). Extra cases from the 8-column empty rule are counted and reported. |
| R2 | `predict_frame` rows == `run_predict` rows on the same cases. |
| R3 | CPU fp32 (production pins, torch 2.9.1) vs GPU bf16 on the whole calibration partition: decision flips reported; Good, Slight, CO, FP, FN each within 0.3 pp. |
| R4 | Calibration silver-eval of the re-housed generation equals the freeze record (`thresholds.json` in-sample verdicts). |
| R5 | `python -m pytest ml -q` green, including `test_docs`. |
| R6 | Opus sign-off. |

## Work packages

| WP | Scope | Implement / Verify | Runs |
|---|---|---|---|
| 1 | This plan + tag | lead | — |
| 2 | `report_mapping/` rebuild, `config.py`, requirements merge, deletions, package tests | Sonnet / Sonnet | smoke only |
| 3 | `generations/` (promote, adopt, spec check), `s3sync/models.py`, `handoff/exports.py`, `evaluation/` ruler, scripts train/calibrate/predict/evaluate/promote/retrain_cycle/generations, their tests and fixtures | Sonnet / Sonnet | smoke only |
| 4 | Runs batch 1: re-house weights into `candidate/`, review_cut fit, R1, R4, upload-mapping measurement, R3, start cross-fit | — | Haiku |
| 5 | `coding/` (combine, queue), `manual_audit/report_mapping_audit.py`, `scripts/oof.py`, `scripts/audit.py`, tests | Sonnet / Sonnet | smoke only |
| 6 | `ml-worker/`, `handoff/worker_format.py`, `ml-worker/CHANGES.md`, worker tests, R2 test | Sonnet / Sonnet | smoke only |
| 7 | Docs: ADR 0008/0009, experiment records, living docs, change request, CLAUDE.md draft for the user | Sonnet / Sonnet | — |
| 8 | Runs batch 2: adopt with `--reason` ADR 0009, R5, cross-fit output check | — | Haiku |
| 9 | Evaluate R1–R5, sign-off; cleanup proposal for `rr_phase*` scratch (deletion needs the user's yes) | Opus | — |

## Conventions

- Work only in the `-rr` worktree; never touch the main tree. Never print report or diagnosis text — IDs, codes,
  counts only. Never commit `ml/data/`, `*.npz` or CSVs with report/diagnosis text.
- Paths only from `config.py`; never `config.X` as a default argument value (binds at import).
- Implementation agents run smoke tests only (tiny models, `--limit`); heavy runs belong to the Run agent.
- Agents do not commit; the lead commits per WP with `git commit --only -- <files>` and a one-line message.
- Python: `ml/.venv/Scripts/python.exe` (import torch before scipy/sklearn on Windows).

## Notes carried between work packages

(append below)

### 2026-10-09 WP2 (report_mapping rebuild): public interfaces for WP3/5/6

Import-light modules: `text`, `batching`, `model`, `generation`, `inference/*` (a subprocess test blocks scipy,
sklearn, diagnosis_mapping, evaluation, generations.guards; transformers itself imports sklearn when it is
installed, so the test blocks them the way the serving image lacks them). `generation` imports no torch.
`config`: `EMBEDDING_CACHE_DIR` removed; `FP_CAP`, `REVIEW_TARGET_SHARE`, `UNCOMMON_GROUPS_TXT` added
(`ml/evaluation/uncommon_groups.txt`, byte copy of WP14's list); `OOF_DIR` now holds cross-fit CSVs.

```
report_mapping.text
  TEXT_SPEC_VERSION = 1; REPORT_COLUMNS
  clean_text(value) -> str
  build_report_text(row: Series) -> str            # missing column = empty
  build_report_texts(reports: DataFrame) -> list[str]   # positional, one per row (repeated case_ids are kept)
  has_text(row: Series) -> bool
  read_reports() -> DataFrame                      # config.REPORT_CSV, latin-1, case_id cleaned
report_mapping.generation                          # stdlib + generations.manifest + config + text
  ARCHITECTURE (dict), KIND, SCHEMA_VERSION, WEIGHT_FILES, THRESHOLD_KEYS, layout names (ENCODER_DIR, HEAD_FILE, ...)
  class GenerationError(Exception)
  new_generation_id(now=None) -> "gen-<UTC>"
  resolve_generation_dir(name: str | Path) -> Path     # "current" | "candidate" | path
  weights_sha256(directory) -> str; weights_digest(file_hashes: dict) -> str
  read_terms(dir) -> list[str]; read_term_info(dir) -> dict[str, (group, code)]; read_thresholds(dir) -> dict
  write_pending_manifest(directory, *, encoder_id, max_length, recipe: dict, parents: dict, training: dict) -> dict
      # directory already holds encoder/, head.pt, gate_head.pt, terms.json, term_info.json; calibration {"status": "pending"}
  write_calibration(directory, thresholds: dict, block: dict) -> dict   # thresholds.json + calibrated block bound to weights; keeps created_at/git_sha
  verify_spec(directory, *, require_calibrated=True) -> manifest    # raises GenerationError
report_mapping.model
  ENCODER_ID; new_encoder(model_id=ENCODER_ID); load_tokenizer(model_id=ENCODER_ID)
  class JointClassifier(encoder, terms): .head .gate_head .terms; .save(directory, tokenizer); .init_biases(term_freq, gate_rate)
  load_joint(directory) -> (model, tokenizer)      # ValueError if terms.json and the term head disagree
report_mapping.inference.scoring
  TOP_K = 49; DEFAULT_TOKEN_BUDGET = 32768
  logits_for(model, tokenizer, texts, max_length, device, token_budget=DEFAULT_TOKEN_BUDGET) -> (term_logits, gate_logits, n_over)
  rank(case_ids, terms, term_logits, gate_logits) -> RankedScores(terms, case_ids, term_idx, probs, gate_probs)
report_mapping.inference.decide
  COLUMNS, NEVER, TERM_METHOD = "term_head", REJECTED_METHOD; Thresholds(gate_t, tau, rho)
  case_rows(...); decide_all(scores: RankedScores, th, term_info, generation_id) -> DataFrame   # no has_text arg: score only cases with text
report_mapping.inference.predict
  load_generation(directory, device) -> LoadedGeneration(model, tokenizer, thresholds, terms, term_info, max_length, generation_id)
  predict_frame(reports: DataFrame, loaded, device) -> DataFrame[decide.COLUMNS]   # case_id + any subset of the 8 columns
  run_predict(generation_dir, device, out=None) -> Path
report_mapping.training
  recipe: Recipe (frozen dataclass), PRODUCTION_SEED, seed_all(seed)
  labels: load_labels_table, lineage, select_train, training_case_ids, cancer_case_ids, gate_targets
  data: load_training_data(labels_table: DataFrame, split_id, limit=0) -> TrainingData; split_holdout(targets, fraction, seed); assign_folds(targets, k, seed)
  train: train(labels: str, split_id, out_dir, recipe, device, model_id=ENCODER_ID, limit=0, max_steps=0) -> manifest
         fit_model(texts, term_targets, gate_targets, vocab, recipe, device, model_id, max_steps=0, holdout=None) -> (model, tokenizer, summary)
  calibrate: calibrate(generation_dir, labels: str, split_id, max_fp_share, review_target_share, device) -> calibration block
         fit_thresholds, fit_review_cut(rank1_confidences, target_share), build_pattern_counts, PatternCounts
  crossfit: crossfit(labels: str, split_id, k, recipe, device, out_csv, limit=0, max_steps=0, model_id=ENCODER_ID) -> Path
```

- `thresholds.json` = `gate_t, tau, rho, review_cut, max_fp_share, review_target_share`. `review_cut` is fitted on the
  2-decimal rank-1 confidences of the accepted calibration cases (what the predictions CSV carries), so the queue
  rule `confidence < review_cut` on that CSV reproduces the fitted share.
- `calibrate` refuses a generation whose calibration is not `pending` (a recalibration is a new generation, so
  `generations.py recalibrate` must copy, reset `calibration` to pending, delete `thresholds.json`, new id, rewrite
  the manifest, then call `calibrate`). A re-housed frozen generation is built with `model.JointClassifier`-layout
  files + `term_info.json` + `generation.write_pending_manifest(...)`.
- Calibration scores only cases with text; labelled calibration cases without text count as false negatives in every
  threshold pattern (same as the old unscored-case rule). `calibration.n_cases_with_text` is recorded.
- Cross-fit CSV columns: `case_id, fold, gate_target, gate_prob, term, term_prob` (one row per labelled term of a
  cancer case, one empty-term row for a non-cancer case; `term_prob` empty when the term is outside the fold model's
  vocabulary). The audit (WP5) must read this instead of the old gate-only OOF file.
- `tests/fixtures.py`: `build_tiny_modernbert`, `build_tiny_generation(root, encoder_dir, calibrated=True)`;
  conftest fixture `tiny_modernbert_dir`. The old bundle builders (`build_report_mapping_bundle`, head builders) are
  still there for the not-yet-rewired tests; delete them in WP3. `build_tiny_bert` now builds its tokenizer with
  `vocab=` (transformers 5 ignored `vocab_file`, so every word was `[UNK]`).

**2026-10-09 lead, after WP2 verify (PASS WITH FIXES):**
- Re-housing (WP4) must NOT refit: build the generation with `write_pending_manifest`, then `generation.write_calibration` with the frozen gate_t 0.75 / tau 0.10 / rho 0.25, max_length 4096, `review_cut` from `fit_review_cut` on 2-decimal rank-1 confidences of accepted calibration cases (simulated: cut 0.85, share 0.2325 vs target 0.2336), and a calibration block from `build_pattern_counts`.
- 8-column has-text rule: all 58,313 report rows have text vs 58,172 under the old rule; 141 extra cases (109 train / 17 calibration / 15 test).
- R1 exact check: the harness replays the old per-partition scoring (train, calibration, test of `three-way-v1`, report order within each, one `logits_for` call per partition) and compares by case_id merge. Separately report cell flips of the library's one-pass `run_predict` vs the frozen CSV (bf16 batch-composition noise; verify measured ≤ 7.6e-6 term-logit drift for partition-vs-one-pass on synthetic text).
- R4: compare calibration verdicts to the freeze record after masking the 15 extra labelled calibration cases; report the delta with them included.
