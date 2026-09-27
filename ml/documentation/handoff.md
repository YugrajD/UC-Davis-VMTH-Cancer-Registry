# Handoff

The file contracts between ML (this codebase) and the cloud (the registry app / backend, and
ml-worker). Per [icd-mapping-strategy.md](icd-mapping-strategy.md): "the cloud stores, serves and
reviews; local machines compute... data crosses manually, through exports and imports the backend
developer builds." Package: `ml/handoff/` (`contracts.py`, `imports.py`, `exports.py`,
`worker_format.py`). Entry point: `scripts/handoff.py`.

## Schemas and sidecars (`contracts.py`)

Every exchanged file gets a sidecar `<file>.manifest.json` (`{kind, schema_version, sha256,
written_at}`), so a schema change on either side is visible rather than silently misinterpreted.
Bump a `*_SCHEMA_VERSION` constant whenever that file's columns change in a way the other side must
know about. Column lists are read from their owning module's own constants
(`diagnosis_mapping.silver`, `coding.combine`, `coding.queue`) rather than duplicated.

| Kind | Direction | Columns | Contains case text? |
|---|---|---|---|
| `pending_diagnoses` | inbox (cloud → ML) | `case_id, diagnosis_number, diagnosis` | Yes — inbox-only, never committed |
| `gold` (v2) | inbox (cloud → ML) | `case_id, term/code, ...`; `origin` optional for a listed case | No |
| `silver_codes` | outbox (ML → cloud) | annotation columns minus `diagnosis`, plus `silver_generation` | No |
| `combined_codes` | outbox (ML → cloud) | `coding.combine.COMBINED_CODES_COLUMNS` | No |
| `review_queue` | outbox (ML → cloud) | `coding.queue.REVIEW_QUEUE_COLUMNS` | No |
| `audit_list` | outbox (ML → cloud) | a `.txt`, one case_id per line, in review order | No |
| `report_mapping_bundle` | outbox (ML → cloud) | a whole generation directory, tarred | No |

## Imports (`imports.py`)

```
ml/.venv/Scripts/python.exe ml/scripts/handoff.py import-pending --csv PATH --export-id ID
ml/.venv/Scripts/python.exe ml/scripts/handoff.py import-gold     --csv PATH --export-id ID --reviewer "Dr. Smith"
```

- **`import_pending_diagnoses`** — every import is keyed by a mandatory `export_id`; a later
  export's rows for a case_id replace that case's earlier rows in the cumulative landing table
  (`config.HANDOFF_PENDING_DIAGNOSES_CSV`) — the same "later export replaces the earlier one" rule
  the gold store applies per case. This module owns that merge.
- **`import_gold`** — lands a raw copy for the audit trail, then delegates entirely to
  `manual_audit.gold.ingest_gold` for validation and storage. A blank or absent `origin` is filled
  from the audit-list ledger for a case that was on an audit list (refused for any other case).
  Never duplicates gold-store logic.

Every raw import is also copied into `config.HANDOFF_INBOX_DIR` under its export-stamped filename
with a sidecar — an audit trail of exactly what the cloud sent and when, independent of what the
merge/ingest step did with it. Read as utf-8 (handoff files are this rewrite's own outputs, not the
raw latin-1 legacy corpus).

## Exports (`exports.py`)

```
ml/.venv/Scripts/python.exe ml/scripts/handoff.py export-silver --silver-id silver-1
ml/.venv/Scripts/python.exe ml/scripts/handoff.py export-coding --run-id 2026-10-01
ml/.venv/Scripts/python.exe ml/scripts/handoff.py export-bundle --generation current
ml/.venv/Scripts/python.exe ml/scripts/handoff.py export-audit-list --list-id 2026-10-01
```

- **`export_silver`** — `silver_codes_<silver_id>.csv`. No diagnosis text (the cloud already holds
  it — it sent it as `pending_diagnoses`).
- **`export_coding`** — `combined_codes_<run>.csv` + `review_queue_<run>.csv`, the already-computed
  `coding.combine`/`coding.queue` outputs, re-stamped with a schema version. Both are already
  text-free.
- **`export_bundle`** — tars a report-mapping generation directory + a `.sha256` sidecar, for
  ml-worker. The source generation is verified (manifest file-hashes + embedding fingerprint, via
  `worker_format.verify_worker_bundle`) *before* bundling, so a stale or tampered generation is
  refused rather than shipped. `verify_bundle` re-runs the same check after extracting the tarball,
  for the receiving side to confirm it travelled intact.
- **`export_audit_list`** — `audit_list_<list_id>.txt`: every case awaiting specialist review
  (eval batch, Diagnosis-Mapping audit, Report-Mapping audit, review queue; each once; cases with
  gold left off), for the dashboard. Only case IDs cross; the gold origin each case must come back
  under stays in `config.AUDIT_LIST_LEDGER_CSV`, and `import_gold` fills it in. See
  [manual-audit.md](manual-audit.md), "Universal audit list".

## The worker bundle contract (`worker_format.py`)

A bundle is just a report-mapping generation directory — `report_mapping.model.generation`'s own
layout *is* the cloud bundle layout ml-worker downloads. This module names the subset of it
ml-worker needs, and the one check both sides run before trusting a bundle
(`verify_worker_bundle` — manifest file-hashes + embedding fingerprint, reusing
`generations.manifest.verify_manifest` and `report_mapping.model.generation.verify_fingerprint`).

`resolve_bundle_root(env, model_var)` — the bundle root is the parent of the worker's model-path env
var (`PETBERT_MODEL_PATH` for the HTTP worker `app.py`, `MODEL_PATH` for GCP Batch
`batch_predict.py`), which must be `<root>/petbert`. Every other path variable, if set, must point
at its own place inside that same bundle:

| Variable | Must be |
|---|---|
| `LABELS_CSV_PATH` | `<root>/labels/labels.csv` |
| `GROUP_CLASSIFIER_PATH` | `<root>/checkpoints/group_classifier_best.pt` |
| `CASE_PRESENCE_CLASSIFIER_PATH` | `<root>/checkpoints/case_presence_classifier.pt` |
| `LP_THRESHOLDS_JSON_PATH` | `<root>/checkpoints/label_presence/lp_thresholds.json` |
| `UNCOMMON_GROUPS_PATH` | `<root>/checkpoints/uncommon_groups.txt` |

A deployment mixing files from two generations refuses to start rather than predicting with them.

`upload_to_reports(upload)` — an upload (Dataset A: `anon_id` + one `Text` column) is expanded into
report-shaped columns: `Text` fills every section's first source column (so section 1, FINAL
COMMENT + COMMENT, is `Text` alone), matching the legacy worker's behaviour.

`response_rows(rows, text_by_id)` — one dict per case, in the legacy `"1) a 2) b"` numbered-string
format `ingestion_service.parse_predictions` reads, now also carrying `source_version` (the
`generation_id` that produced the prediction — the traceability icd-mapping-strategy.md asks for).

See [ml-worker-change-request.md](ml-worker-change-request.md) for what the backend developer needs
to change to deploy this (GCS upload layout, `gcp_batch_service.py`, `docker-compose.yml`).
