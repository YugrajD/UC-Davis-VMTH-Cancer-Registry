# Paths and config constants

Every path and S3 constant in `ml/config.py`, what it holds, and how `ml/output/` is laid out. For
anyone who needs to find a file or change where something is written. Code never hardcodes a path:
every module reads it from `config.py`, so this page and that file move together. All paths below are
relative to the repository root. Nothing under `ml/data/` or `ml/output/` is ever committed (private
veterinary records).

## Roots and inputs

| Constant | Path | Holds |
|---|---|---|
| `PACKAGE_ROOT`, `ML_ROOT` | `ml/` | The package root; in-tree assets resolve from here |
| `DATA_DIR` | `ml/data/` | Raw patient input. Synced (set `data`) |
| `OUTPUT_DIR` | `ml/output/` | Every generated artifact |
| `REPORT_CSV` | `ml/data/report.csv` | Pathology report text, one row per case (read as latin-1) |
| `DIAGNOSES_CSV` | `ml/data/diagnoses.csv` | Diagnosis lines, one row per (case, diagnosis number) |
| `LABELS_CSV` | `ml/taxonomy/labels.csv` | The Vet-ICD-O-canine-1 taxonomy: 845 terms, 52 groups, 534 codes. In git |
| `DEFAULT_SPLIT_ID` | `three-way-v1` | Split id every command defaults to (not a path) |

## Silver and diagnosis mapping

| Constant | Path | Holds |
|---|---|---|
| `SILVER_DIR` | `ml/output/silver/` | One immutable directory per silver generation: `<silver_id>/annotation.csv` plus a manifest. Synced (set `silver`) |
| `DIAGNOSIS_MAPPING_STATS_DIR` | `ml/output/diagnosis_mapping_stats/` | Coverage statistics and plots, one subdirectory per silver id. Reproducible. Synced |

## Manual audit

| Constant | Path | Holds |
|---|---|---|
| `MANUAL_AUDIT_DIR` | `ml/output/manual_audit/` | Root of every audit store. Synced (set `manual_audit`) |
| `AUDIT_STORE_CSV` | `manual_audit/audit_store.csv` | Row-level judgements from the early Diagnosis-Mapping pilot; never gold; nothing writes to it now |
| `GOLD_STORE_CSV` | `manual_audit/gold_store.csv` | Case-level gold, one row per (case, code), each with an origin. Absent until the first gold is ingested |
| `EVAL_BATCH_LEDGER_CSV` | `manual_audit/eval_batch_ledger.csv` | Every eval batch drawn: strata, weights, review mode |
| `CAUSE_STORE_CSV` | `manual_audit/cause_store.csv` | Cause-pass answers. Absent until the first is ingested |
| `DIAGNOSIS_MAPPING_AUDIT_DIR` | `manual_audit/diagnosis_mapping_audit/` | Per batch: `diagnosis_mapping_audit_batch<N>_key.csv` (holds diagnosis text; local) and `..._batch<N>.txt` (case ids) |
| `DIAGNOSIS_MAPPING_AUDIT_BATCH1_TXT` | `diagnosis_mapping_audit/diagnosis_mapping_audit_batch1.txt` | Batch 1's case list; the eval batch excludes these cases |
| `REPORT_MAPPING_AUDIT_DIR` | `manual_audit/report_mapping_audit/` | Per batch: `report_mapping_audit_batch<N>.csv` (ledger) and `.txt` (case ids) |
| `AUDIT_LIST_LEDGER_CSV` | `manual_audit/audit_list_ledger.csv` | Every case put on an audit list and the gold origin it must return under |
| `EVAL_BATCH_DIR` | `manual_audit/eval_batch/` | Eval-batch review sheets and taxonomy sheets |

## Coding

| Constant | Path | Holds |
|---|---|---|
| `CODING_DIR` | `ml/output/coding/` | Coding-rule outputs. Synced (set `coding`) |
| `CORRECTED_ANNOTATIONS_CSV` | `coding/corrected_annotations.csv` | Report-mapping training labels (train partition only) |
| `COMBINED_PREDICTIONS_CSV` | `coding/combined_predictions.csv` | The best code set per case (gold, silver, bronze) |
| `REVIEW_QUEUE_CSV` | `coding/review_queue.csv` | ML-side review queue |

## Splits, generations, embeddings

| Constant | Path | Holds |
|---|---|---|
| `SPLITS_DIR` | `ml/output/splits/` | `<split_id>/{train,calibration,test}_cases.txt` and a manifest. Immutable. Synced (set `splits`) |
| `REPORT_MAPPING_DIR` | `ml/output/report_mapping/` | Root of the report-mapping generations |
| `REPORT_MAPPING_CURRENT_DIR` | `report_mapping/current/` | The production generation. Published with `publish-model` |
| `REPORT_MAPPING_CANDIDATE_DIR` | `report_mapping/candidate/` | Unpromoted training output; deleted if it loses. Not synced |
| `EMBEDDING_CACHE_DIR` | `report_mapping/embedding_cache/` | `<key>.npz` content-hash embedding cache; never bundled, not synced |
| `OOF_DIR` | `report_mapping/oof/` | Out-of-fold gate scores (`train.py --stage oof`) read by the Report-Mapping audit. Synced (set `oof`) |
| `ARCHIVE_ROOT` | `ml/output/archive/` | Replaced generations as `YYYY-MM-DD_<description>/`. Written only by `ml/generations/`, never loaded from. Not synced |

A generation directory (`current/` or `candidate/`) holds `petbert/` (the backbone), `labels/labels.csv`,
`checkpoints/` (`case_presence_classifier.pt`, `group_classifier_best.pt`, `label_presence/<group>.pt`,
`label_presence/lp_thresholds.json`, `thresholds.json`, `uncommon_groups.txt`,
`calibration_diagnostics.json`) and `manifest.json`. The manifest and fingerprint rules are in
[generations.md](../concepts/generations.md).

## Predictions and evaluation

| Constant | Path | Holds |
|---|---|---|
| `PREDICTIONS_DIR` | `ml/output/predictions/` | `<generation_id>_predictions.csv`. Synced (set `predictions`) |
| `EVAL_DIR` | `ml/output/eval/` | Evaluation outputs. Synced (set `eval`) |
| `SILVER_EVAL_HISTORY_CSV` | `eval/silver_eval_history.csv` | One line per silver-eval run |

## Handoff

| Constant | Path | Holds |
|---|---|---|
| `HANDOFF_DIR` | `ml/output/handoff/` | Cloud exchange root. Synced (set `handoff`), minus the bundles directory |
| `HANDOFF_INBOX_DIR` | `handoff/inbox/` | Raw copies (with sidecars) of every file the cloud sent |
| `HANDOFF_OUTBOX_DIR` | `handoff/outbox/` | Files ML sends: silver codes, combined predictions, review queue, audit lists |
| `HANDOFF_PENDING_DIAGNOSES_CSV` | `handoff/inbox/pending_diagnoses.csv` | Cumulative table merged from every pending-diagnoses import. Holds diagnosis text |
| `HANDOFF_BUNDLES_DIR` | `handoff/outbox/bundles/` | Worker bundle tarballs. Excluded from sync |

Formats and sidecars are in [handoff-contracts.md](handoff-contracts.md).

## S3 sync

| Constant | Value | Meaning |
|---|---|---|
| `S3_BUCKET` | `ucd-canine-registy-storage-231161110555-us-west-2-an` | "registy" is the real bucket name |
| `S3_REGION` | `us-west-2` | |
| `S3_PREFIX` | `ml-Revised-ICD-Mapping/` | Every key this tool writes sits under it |
| `S3_PROTECTED_PREFIXES` | `database/`, `ml/` | Other people's files; the key builder refuses them |
| `S3_SYNC_SETS` | `data`, `manual_audit`, `silver`, `splits`, `coding`, `predictions`, `eval`, `oof`, `diagnosis_mapping_stats`, `handoff` | Set name mapped to the local directory above |
| `S3_SYNC_EXCLUDED_DIRS` | `handoff/outbox/bundles/` | Never part of a set: a bundle duplicates the model generation |
| `S3_SYNC_STATE_JSON` | `ml/output/s3sync_state.json` | Per-machine record of what was last synced. Never copied between machines |
| `S3_SYNC_BACKUP_DIR` | `ml/output/s3sync_backup/` | Safety copies `pull` makes before replacing a file. Not synced |

Sync sets (`S3_SYNC_SETS`) and the local directory each one mirrors:

| Set | Directory |
|---|---|
| `data` | `ml/data/` |
| `manual_audit` | `ml/output/manual_audit/` |
| `silver` | `ml/output/silver/` |
| `splits` | `ml/output/splits/` |
| `coding` | `ml/output/coding/` |
| `predictions` | `ml/output/predictions/` |
| `eval` | `ml/output/eval/` |
| `oof` | `ml/output/report_mapping/oof/` |
| `diagnosis_mapping_stats` | `ml/output/diagnosis_mapping_stats/` |
| `handoff` | `ml/output/handoff/` (minus `outbox/bundles/`) |

The model generation is published separately from the sets. Layout and commands:
[sync-with-s3.md](../how-to/sync-with-s3.md).

## `ml/output/` at a glance

```
ml/output/
  silver/<silver_id>/            annotation.csv + manifest
  diagnosis_mapping_stats/<silver_id>/
  manual_audit/                  stores, ledgers, audit batches, eval_batch/
  coding/                        corrected_annotations, combined_predictions, review_queue
  splits/<split_id>/             train, calibration and test case lists
  report_mapping/
    current/                     production generation
    candidate/                   only while a retrain is pending
    embedding_cache/             <key>.npz
    oof/                         out-of-fold gate scores
  predictions/                   <generation_id>_predictions.csv
  eval/                          silver_eval_history.csv, evaluation outputs
  handoff/
    inbox/                       files from the cloud
    outbox/                      files to the cloud; bundles/ inside it
  archive/                       YYYY-MM-DD_<description>/ replaced generations
  s3sync_state.json              per machine
  s3sync_backup/                 per machine
```

`ml/diagnosis_mapping/.env` (untracked) is the one config file outside `config.py`: `LLM_HOST`,
`API_PORT` and `LLM_MODEL` for the Tier-3 LLM server.

_Last verified against code: 2026-09-29_
