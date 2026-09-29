# Scripts and flags

Every entry point in `ml/scripts/`: what it does, its subcommands and every flag with its default.
Each script is a thin wrapper; the work lives in the package named in the first line of its section.
To see a flag list on the machine, run the script (or a subcommand) with the standard help option.
Commands are written for Windows (`ml/.venv/Scripts/python.exe`); on macOS and Linux use
`ml/.venv/bin/python`. Task-by-task commands are in [train-and-promote.md](../how-to/train-and-promote.md),
[run-inference.md](../how-to/run-inference.md), [run-audit-cycle.md](../how-to/run-audit-cycle.md)
and [sync-with-s3.md](../how-to/sync-with-s3.md).

| Script | Job |
|---|---|
| `audit.py` | Draw audit batches, ingest gold, cause pass |
| `calibrate.py` | Fit a generation's thresholds |
| `code_cases.py` | Combined predictions, corrected annotations, review queue |
| `evaluate.py` | Silver-eval, gold-eval, audit rates |
| `generations.py` | Show generations and triggers, fork a generation |
| `handoff.py` | Cloud file imports and exports |
| `map_diagnoses.py` | Diagnosis cascade into a silver generation, coverage stats |
| `predict.py` | Predictions from a generation |
| `promote.py` | Promotion recommendation and swap |
| `retrain_cycle.py` | The local retraining lane in one call |
| `split.py` | Create a three-way split, run the leakage guards |
| `sync.py` | Sync file sets and the model with S3 |
| `train.py` | Train a candidate generation |

Conventions shared by the scripts:

- `--split` and `--split-id` name a split under `output/splits/`. Where a default is listed it is
  `three-way-v1` (`config.DEFAULT_SPLIT_ID`).
- `--device` accepts `auto`, `cpu`, `cuda`, `mps` and `xpu` on `train.py`, `predict.py` and
  `retrain_cycle.py` (default `auto`: cuda, then xpu, then mps, then cpu). `calibrate.py` takes any
  device string and defaults to `cpu`. Nothing else has a device flag.
- `--labels` and `--silver` take a `silver_id` (a directory under `output/silver/`); `--labels`
  also takes a labels-table CSV path.
- A generation is `current`, `candidate` or a literal directory.

## `audit.py`

Manual audit: `ml/manual_audit/`. Concepts: [manual-audit.md](../concepts/manual-audit.md). Subcommands:

**`dm-sample`**: draw a Diagnosis-Mapping audit batch (key CSV plus case list).

| Flag | Default | Meaning |
|---|---|---|
| `--silver-id` | required | silver generation to sample from |
| `--split-id` | `three-way-v1` | rows come from its calibration and test cases |
| `--batch` | required | batch number (int) |
| `--n-rows` | 200 | rows to draw |
| `--seed` | 42 | sampling seed |

**`rm-sample`**: draw a Report-Mapping audit batch (ledger CSV plus case list).

| Flag | Default | Meaning |
|---|---|---|
| `--oof-csv` | required | gate out-of-fold scores from `train.py --stage oof` |
| `--split-id` | `three-way-v1` | the split the scores were trained on |
| `--batch` | required | batch number (int) |
| `--n-contradicted` | 100 | cases whose gate score contradicts the label |
| `--n-random` | 100 | random baseline cases |
| `--seed` | 42 | sampling seed |

**`eval-batch`**: draw one batch of the case-level eval-batch series (gold-eval).

| Flag | Default | Meaning |
|---|---|---|
| `--batch-id` | required | batch name |
| `--silver-id` | required | silver generation that defines the strata |
| `--fraction` | required | share of each stratum's remaining target to draw now |
| `--split-id` | `three-way-v1` | cases come from its test partition |
| `--seed` | 42 | sampling seed |
| `--target-codes-per-group` | 30 | code target per big group |
| `--big-group-share` | 0.01 | share of codes that makes a stratum "big" |
| `--rare-stratum-n` | 2 | fixed draw for a rare stratum |
| `--no-cancer-target` | 100 | fixed draw for the no-cancer stratum |

**`ingest-sheet`**: ingest a filled eval-batch sheet into the gold store.

| Flag | Default | Meaning |
|---|---|---|
| `--sheet` | required | filled sheet path |
| `--batch-id` | required | batch the sheet belongs to |
| `--reviewer` | required | reviewer name |
| `--split-id` | the split recorded for the batch | override |
| `--labels-csv` | `config.LABELS_CSV` | taxonomy used to resolve terms |

**`ingest-gold`**: ingest case-level gold rows directly.

| Flag | Default | Meaning |
|---|---|---|
| `--rows-csv` | required | columns `case_id`, `origin`, and `term` and/or `code` |
| `--origin` | none | origin for every row when the CSV has no `origin` column (must not conflict with one) |
| `--reviewer` | required | reviewer name |
| `--labels-csv` | `config.LABELS_CSV` | taxonomy used to resolve terms |
| `--batch-or-export-id` | empty | recorded with the rows |
| `--upload-period` | empty | recorded with the rows |
| `--slice-rate` | empty | required for `random_slice` rows, in (0, 1] |

**`cause-sheet`**: build a misses sheet from a verdict table.

| Flag | Default | Meaning |
|---|---|---|
| `--verdicts-csv` | required | columns `case_id, gold_code, method, source_version` |
| `--out-csv` | required | sheet to write |

**`ingest-cause`**: ingest a filled cause-pass sheet.

| Flag | Default | Meaning |
|---|---|---|
| `--filled-csv` | required | filled sheet |
| `--reviewer` | required | reviewer name |

## `calibrate.py`

Fits every threshold on the split's calibration partition from cached embeddings; never embeds.
Package: `ml/report_mapping/training/calibrate.py`. Design: [report-mapping.md](../concepts/report-mapping.md).

| Flag | Default | Meaning |
|---|---|---|
| `--generation` | required | generation to calibrate |
| `--labels` | required | a silver id or labels CSV with rows on the calibration partition (never the corrected annotations, which are train-only) |
| `--split` | required | split holding the calibration partition |
| `--partition` | `calibration` | partition to fit on; only `calibration` is accepted |
| `--device` | `cpu` | device for the heads |

## `code_cases.py`

Coding-rule outputs: `ml/coding/`. Concepts: [coding.md](../concepts/coding.md). Subcommands:

**`combine`**: write the combined-predictions table.

| Flag | Default | Meaning |
|---|---|---|
| `--silver` | required | silver generation |
| `--split` | required | split id |
| `--predictions` | required | bronze predictions CSV |
| `--generation-id` | the file's own `generation_id` | override when the file holds several |
| `--out` | `config.COMBINED_PREDICTIONS_CSV` | output path |

**`corrected`**: write the corrected-annotations table (train partition only).

| Flag | Default | Meaning |
|---|---|---|
| `--silver` | required | silver generation |
| `--split` | required | split id |
| `--out` | `config.CORRECTED_ANNOTATIONS_CSV` | output path |

**`queue`**: write the ML review queue.

| Flag | Default | Meaning |
|---|---|---|
| `--silver` | required | silver generation |
| `--split` | required | split id |
| `--predictions` | required | bronze predictions CSV |
| `--generation-id` | the file's own `generation_id` | override when the file holds several |
| `--out` | `config.REVIEW_QUEUE_CSV` | output path |

## `evaluate.py`

Scoring: `ml/evaluation/`. Prints counts and percentages only. Concepts: [evaluation.md](../concepts/evaluation.md). Subcommands:

**`silver`**: bronze predictions against a labels table on one partition; appends a line to `config.SILVER_EVAL_HISTORY_CSV`.

| Flag | Default | Meaning |
|---|---|---|
| `--predictions` | required | predictions CSV |
| `--labels` | required | a silver id or labels CSV |
| `--split` | required | split id |
| `--partition` | required | `test` or `calibration` |
| `--half` | none | keep only the `eval` or `sweep` md5 half of the partition |
| `--generation` | `current` | generation that made the predictions |
| `--history` | `config.SILVER_EVAL_HISTORY_CSV` | history CSV to append to |

**`gold`**: the four gold-eval results with intervals and the representativeness check. Needs gold-eval rows.

| Flag | Default | Meaning |
|---|---|---|
| `--predictions` | required | predictions CSV |
| `--silver` | required | silver generation id |
| `--split` | required | split id |
| `--generation` | `current` | generation that made the predictions |
| `--n-boot` | 1000 | bootstrap replicates |
| `--seed` | 0 | bootstrap seed |
| `--misses-out` | none | write the misses table (input to `audit.py cause-sheet`) |

**`audit-rates`**: weighted Diagnosis-Mapping audit rates per stratum. No flags.

## `generations.py`

Generation management: `ml/generations/`. Concepts: [generations.md](../concepts/generations.md). Subcommands:

**`status`**: show `current/` and `candidate/` and whether a retraining trigger is met.

| Flag | Default | Meaning |
|---|---|---|
| `--silver` | required | silver a challenger would train on |
| `--split` | `three-way-v1` | split whose test side holds gold-eval |
| `--incumbent-predictions` | `output/predictions/<current id>_predictions.csv` | `current/`'s predictions; read only when gold-eval exists |

**`fork`**: copy a generation as a new, uncalibrated one on another split, then run `calibrate.py`.

| Flag | Default | Meaning |
|---|---|---|
| `--split` | required | split to calibrate on; its train must match the source's |
| `--from` | `current` | generation to copy |
| `--to` | `candidate` | destination |

## `handoff.py`

Cloud file contracts: `ml/handoff/`. Formats: [handoff-contracts.md](handoff-contracts.md).
Every subcommand except `export-bundle` accepts `--push`, which pushes the S3 sets it wrote
(`import-pending`, `export-silver` and `export-coding`: `handoff`; `import-gold` and
`export-audit-list`: `manual_audit`, then `handoff`) once it succeeds. Subcommands:

**`import-pending`**: land and merge a pending-diagnoses export.

| Flag | Default | Meaning |
|---|---|---|
| `--csv` | required | exported file |
| `--export-id` | required | export id |
| `--push` | off | then push the S3 sets |

**`import-gold`**: land a gold export and ingest it into the gold store.

| Flag | Default | Meaning |
|---|---|---|
| `--csv` | required | exported file |
| `--export-id` | required | export id |
| `--reviewer` | required | reviewer name |
| `--upload-period` | empty | for random-slice rows |
| `--slice-rate` | empty | for random-slice rows |
| `--push` | off | then push the S3 sets |

**`export-silver`**: write `silver_codes_<silver_id>.csv` to the outbox.

| Flag | Default | Meaning |
|---|---|---|
| `--silver-id` | required | silver generation |
| `--push` | off | then push the S3 sets |

**`export-coding`**: write `combined_predictions_<run>.csv` and `review_queue_<run>.csv` to the outbox.

| Flag | Default | Meaning |
|---|---|---|
| `--run-id` | required | run name in the file names |
| `--push` | off | then push the S3 sets |

**`export-bundle`**: tar a generation and a sha256 file for ml-worker. No `--push`; bundles are not synced.

| Flag | Default | Meaning |
|---|---|---|
| `--generation` | `current` | generation to bundle |

**`export-audit-list`**: write `audit_list_<id>.txt`, every case awaiting review, to the outbox.

| Flag | Default | Meaning |
|---|---|---|
| `--list-id` | required | list name; must be new |
| `--no-review-queue` | off | leave the ML review queue off the list |
| `--push` | off | then push the S3 sets |

## `map_diagnoses.py`

Diagnosis cascade: `ml/diagnosis_mapping/`. Concepts: [diagnosis-mapping.md](../concepts/diagnosis-mapping.md). Subcommands:

**`run`**: run the cascade (and, by default, the cleanup) into a new immutable silver generation.

| Flag | Default | Meaning |
|---|---|---|
| `--id` | required | new silver id |
| `--diagnoses-csv` | `config.DIAGNOSES_CSV` | input diagnoses |
| `--labels-csv` | `config.LABELS_CSV` | taxonomy |
| `--no-llm` | off | skip Tier 3; every eligible row is recorded as a declined match. The result is refused by consumers unless explicitly allowed; do not adopt it |
| `--model` | `LLM_MODEL` from `.env` | Tier-3 model name |
| `--llm-timeout` | 60 | seconds per Tier-3 call |
| `--skip-cleanup` | off | skip the verification cleanup pass |
| `--cleanup-models` | `google/gemma-4-31b,qwen/qwen3.6-27b` | comma-separated verifier models |
| `--cleanup-tiebreaker` | none | third model used when the pair disagrees |
| `--cleanup-timeout` | 60 | seconds per cleanup call |

**`stats`**: coverage statistics for a silver generation.

| Flag | Default | Meaning |
|---|---|---|
| `--silver` | required | silver id |
| `--out-dir` | `config.DIAGNOSIS_MAPPING_STATS_DIR/<silver id>` | output directory |
| `--no-plots` | off | skip PNG plots |
| `--allow-no-llm` | off | allow analysing a `--no-llm` generation |

## `predict.py`

Stamped predictions from a generation: `ml/report_mapping/inference/`. Guide: [run-inference.md](../how-to/run-inference.md).

| Flag | Default | Meaning |
|---|---|---|
| `--generation` | required | generation to predict with |
| `--model` | the generation's own `petbert/` | Hugging Face checkpoint directory or name |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, `mps`, `xpu` |
| `--embed-only` | off | fill the embedding cache and stop |
| `--out` | `output/predictions/<generation_id>_predictions.csv` | predictions CSV path |
| `--local-only` | off | disable Hugging Face downloads |
| `--cache-dir` | `config.EMBEDDING_CACHE_DIR` | embedding cache directory (an empty directory forces a re-embed) |

## `promote.py`

Promotion: `ml/generations/promote.py`. Rule and archive: [generations.md](../concepts/generations.md).
Without `--apply` it only prints the recommendation. It refuses without gold-eval rows.

| Flag | Default | Meaning |
|---|---|---|
| `--candidate-predictions` | required | predictions CSV made by `candidate/` |
| `--incumbent-predictions` | required | predictions CSV made by `current/` |
| `--split` | `three-way-v1` | split whose test side holds gold-eval |
| `--n-boot` | 1000 | bootstrap replicates |
| `--seed` | 0 | bootstrap seed |
| `--apply` | off | promote a winner (archiving `current/` first) or delete a losing candidate |
| `--description` | the incumbent's generation id | archive folder suffix |
| `--publish` | off | after a promotion, publish the new `current/` to S3; requires `--apply` |

## `retrain_cycle.py`

The local lane of the full cycle, recommend-only, each step as its own process. Steps and stop
conditions: [generations.md](../concepts/generations.md). Guide:
[train-and-promote.md](../how-to/train-and-promote.md). Refuses if `candidate/` exists. Currently
prints STOP because no gold exists.

| Flag | Default | Meaning |
|---|---|---|
| `--silver` | required | silver the challenger trains on |
| `--split` | `three-way-v1` | split id |
| `--device` | `auto` | device for training and prediction; calibration uses `cpu` when this is `auto` |
| `--local-only` | off | pass `--local-only` to training |
| `--backbone` | off | retrain the backbone before the heads |
| `--force` | off | train even when no retraining trigger is met |
| `--gold-csv` | none | a `gold_<export>.csv` to ingest first; needs `--export-id` and `--reviewer` |
| `--export-id` | none | export id for `--gold-csv` |
| `--reviewer` | none | reviewer for `--gold-csv` |
| `--upload-period` | empty | passed to the gold import |
| `--slice-rate` | empty | passed to the gold import |
| `--n-boot` | 1000 | bootstrap replicates for the promotion recommendation |

## `split.py`

Splits and leakage guards: `ml/generations/`. Splits are immutable. Subcommands:

**`create`**: create a three-way split from a two-way parent.

| Flag | Default | Meaning |
|---|---|---|
| `--parent` | required | parent split id |
| `--id` | required | new split id |

**`check`**: run every applicable leakage guard. It cannot exercise the calibration-inputs guard, which runs inside `calibrate.py`.

| Flag | Default | Meaning |
|---|---|---|
| `--split` | required | split id |
| `--labels-csv` | none | a train-only labels table to check as well |

## `sync.py`

S3 sync: `ml/s3sync/`. Guide: [sync-with-s3.md](../how-to/sync-with-s3.md). Every command that
writes is a dry run until `--apply`. Subcommands:

- **`status [SET]`**: compare local with the remote, no changes.
- **`push [SET]`** and **`pull [SET]`**: sync one file set or all of them (`SET` defaults to `all`).
- **`publish-model`** and **`pull-model`**: publish `current/` as an immutable generation, or fetch,
  verify and adopt the remote one.

| Flag | Default | Meaning |
|---|---|---|
| `--prefix` | none | top-level flag, before the subcommand: narrow to a sub-prefix of `config.S3_PREFIX`, for scratch runs |
| `--apply` | off | on `push`, `pull`, `publish-model` and `pull-model`: execute instead of printing the plan |

## `train.py`

Trains a candidate generation: `ml/report_mapping/training/`. Guide:
[train-and-promote.md](../how-to/train-and-promote.md). `--stage backbone` defaults `--model` to
`SAVSNET/PetBERT`; every other stage, and `predict.py`, default to the generation's own `petbert/`.
Leakage guards run only for the `heads` and `backbone` stages.

| Flag | Default | Meaning |
|---|---|---|
| `--stage` | required | `backbone`, `case-presence`, `group`, `label-presence`, `heads` (all three heads) or `oof` |
| `--labels` | required | a silver id or a labels CSV path |
| `--split` | `three-way-v1` | split id |
| `--seed` | 42 | training seed |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, `mps`, `xpu` |
| `--out` | `candidate` | `current`, `candidate` or a literal directory |
| `--local-only` | off | disable Hugging Face downloads |
| `--model` | see above | backbone directory or name to start from |
| `--oof-stage` | `case-presence` | with `--stage oof`: which head to run out-of-fold predictions for (`case-presence` or `group`) |
| `--k` | 5 | with `--stage oof`: number of folds |

_Last verified against code: 2026-09-29_
