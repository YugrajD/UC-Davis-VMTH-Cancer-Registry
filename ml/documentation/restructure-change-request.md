# Change request: the report-mapping restructure (2026-10-09)

**From:** ML. **To:** the owners of `backend/`, `docker-compose.yml`, `.github/`, `docs/`,
`doc-site/`, `README.md`, `database/` and `frontend/`. **When:** deploy together with the ML
restructure: the new ml-worker image and the new model bundle go out as a pair (an old image refuses
a new bundle and the reverse). Nothing outside `ml/` and `ml-worker/` was edited by ML; this file
lists what the other owners should change. Why the restructure happened:
[ADR 0008](history/decisions/0008-joint-modernbert-report-mapping.md) and
[ADR 0009](history/decisions/0009-bronze-restructure-and-lifecycle.md). The worker side, file by
file, with the image changes and the deploy and rollback order, is
[ml-worker/CHANGES.md](../../ml-worker/CHANGES.md).

## Summary

The PetBERT four-stage report model is replaced by one jointly trained ModernBERT (a term head and a
cancer gate over the whole report). ML's own files and the backend's contract are kept, so **the
backend keeps working without any change**. What does change is the model bundle (a different folder
layout), the ml-worker image (new library pins), the meaning of two response values (`method` and
`confidence`) and the review rule for report-only cases. The items below are cleanups, one deploy
step and a few places where a number or a name now means something else.

## What stays the same

- **The files ML sends and receives.** `combined_predictions_<run>.csv` (schema v2),
  `review_queue_<run>.csv` (v1), `audit_list_<id>.txt` (v1), `silver_codes_<silver_id>.csv` and the
  gold export keep their columns, sidecars and import rules
  ([handoff-contracts.md](reference/handoff-contracts.md)). The backend's importers need no change.
- **The model-path contract.** The backend still sets `MODEL_PATH=<bundle>/petbert` (batch and ECS)
  or `PETBERT_MODEL_PATH` (the HTTP worker). The worker uses only the parent directory as the
  bundle root, so the leaf `petbert` is a name kept for the backend; the directory does not exist
  and the path need not exist. `docker-compose.yml`'s default
  `/ml/output/report_mapping/current/petbert` stays valid.
- **The worker's endpoints and response keys.** `GET /health` (`status`, `source_version`),
  `POST /predict` (multipart `file`, CSV or xlsx, 50 MB limit), `predictions.json` from the batch
  job, and the keys `anon_id`, `original_text`, `predicted_term`, `predicted_group`,
  `predicted_code`, `confidence`, `method`, `source_version`, with the `1) a 2) b` numbering for a case
  with two terms. The Dataset A upload format (`anon_id` plus `Text`) is unchanged.
- **Environment variable names and the S3 layout.** `JOB_ID`, `INPUT_CSV_PATH`, `OUTPUT_DIR`,
  `MODEL_PATH`, `PETBERT_MODEL_PATH`, and the ECS entrypoint's `S3_BUCKET`, `MODEL_FOLDER`,
  `AWS_REGION`, `AWS_S3_ENDPOINT_URL`; bundles are still downloaded whole from `models/{MODEL_FOLDER}/`.
  `TORCH_NUM_THREADS` is new and optional.
- **The ingestion parser.** `ingestion_service.parse_predictions` reads the unchanged keys and the
  numbered format; it only skips `method == "empty"`. The `data_source = 'petbert'` database value is
  a label, not a contract.
- **Tests that already pass.** `backend/tests/test_gcp_batch_service.py:194-195` asserts
  `MODEL_PATH == .../production/petbert` and still holds.

## What changes (values, not shapes)

| Before | Now |
|---|---|
| `method` was `embedding`, `low_confidence`, `label_presence`, `unidentified_cancer`, `lipoma_rescue` and others | `method` is `term_head` or `rejected_by_case_presence` |
| `confidence` was a cosine similarity or a stage score | the term's sigmoid probability to two decimals; `0.00` when the gate rejected the case. Not comparable with old values |
| Bundle folder: `petbert/`, `labels/`, `checkpoints/`, `manifest.json` | `encoder/`, `head.pt`, `gate_head.pt`, `terms.json`, `term_info.json`, `thresholds.json`, `manifest.json` |
| Worker verified the bundle against an embedding fingerprint | The worker runs the spec check ([generations.md](concepts/generations.md#the-spec-check)); an old bundle fails it |
| Text fed to the model: three sections | The whole text (upload: the flat `Text`), up to 4096 tokens |
| Image pins transformers 4.46.3 | transformers 5.18.0, tokenizers 0.23.2, huggingface_hub 1.33.0, safetensors 0.8.0 (torch 2.9.1 unchanged) |
| Review rule for report-only cases: confidence below 0.23, or top-1/top-2 margin below 0.15 | gate-rejected, or rank-1 confidence below the generation's `review_cut` (0.85 for the first generation), set at calibration ([coding.md](concepts/coding.md#the-bronze-review-rule)) |

## Changes requested

Blocking means the item must be done for the new image and bundle to work or to be deployed
correctly. Everything else can follow. Status is `open` until the owner records otherwise.

| ID | Owner | Where | Required change | Why | Blocking | Status |
|---|---|---|---|---|---|---|
| RC-01 | backend / deploy | `.env.example:77`, `backend/service.yaml:135`, the ECS task definition, the GCP Batch config | Build and push the three ml-worker images from this branch under new tags (nothing has built them), upload the new bundle to a **new** folder (`models/<folder>/`, contents of the extracted generation, so `models/<folder>/manifest.json` exists), and point `GCP_BATCH_IMAGE_URI` / the task definition and the selected folder at the new pair together. Do not overwrite the old folder. Steps: [ml-worker/CHANGES.md](../../ml-worker/CHANGES.md) | An old image refuses a new bundle and a new image refuses an old one; a mismatched pair fails at start-up | yes | open |
| RC-02 | docker-compose.yml | `docker-compose.yml:120-132` | Rebuild the `ml-worker` service (`docker compose build ml-worker`): the requirements changed | The old local image has transformers 4.46.3 and cannot load the new generation | yes (local dev) | open |
| RC-03 | docker-compose.yml | `docker-compose.yml:127-131` | Update the comment: it cites `ml/handoff/worker_format.py` and `WORKER_ENV_VAR_TO_BUNDLE_PATH`, which no longer exist (the five other path variables are not read). Optionally remove `HF_HOME` (line 127; the worker loads from the bundle only) and add `TORCH_NUM_THREADS` (see RC-12) | The comment describes the old contract | no | open |
| RC-04 | backend | `backend/app/services/gcp_batch_service.py:138-139, 156-160`; `backend/app/services/ml_task_service.py:42`; `backend/app/services/ingestion_service.py:597-600` | Docstrings and comments only. The bundle layout is `encoder/ head.pt ...` (not `petbert/, labels/, checkpoints/`) and is checked by the spec check, not "against manifest.json". `ingestion_service.py` cites `ml/documentation/audit-list-change-request.md`, now `ml/documentation/history/plans/audit-list-change-request.md`. Lines `:152` (`model_path = .../petbert`) and `:193` (`MODEL_PATH`) stay correct | Comments say things that are no longer true | no | open |
| RC-05 | backend | `backend/app/services/gcp_batch_service.py:97-119` (`_LEGACY_MODEL_DIRS`, `list_model_folders`), `backend/app/services/s3_service.py:20, 146-156` | Previous-generation bundle folders (PetBERT layout) are still listed as model folders and fail on the new image. Mark or hide them (or delete them once the old generation is not needed), or tell the admins not to select them. No code change is needed for the new bundle | A job that selects an old folder with the new image fails at start-up instead of predicting | no | open |
| RC-06 | backend | `backend/app/config.py:75-80`; `backend/local_ingest_bulk.py:95-96, 121` | `REVIEW_AUTO_ACCEPT_CONFIDENCE` 0.23 and `REVIEW_AUTO_ACCEPT_MARGIN` 0.15 (and the comment at lines 75-77) describe the old review rule; `local_ingest_bulk.py` uses both and tests `method == "unidentified_cancer"`, which is never produced now. The new rule is `confidence < review_cut` per generation (in `thresholds.json`), with no margin. Retire the constants, or read `review_cut` from the bundle's `thresholds.json` if the bulk loader must still flag cases. The HTTP and batch ingestion paths do not use these constants (the backend's ingest-time gate was retired in 83262de) | The bulk loader would flag cases by a rule that no longer matches the model | no | open |
| RC-07 | database | `database/scripts/reflag_review_thresholds.sql:10-23` (and `:71-73`) | Do not run it on new-generation predictions: it re-applies the 0.23 / 0.15 rule, tests `prediction_method = 'low_confidence'` and the `top2_margin` column. Rewrite it around `review_cut` or retire it | It would reflag the new predictions by the old rule | no | open |
| RC-08 | repo-root scripts | `analyze_review_thresholds.py` in the repo-root `scripts` folder (docstring `:2`, description `:403`, default input `:432`), `backend/tests/test_review_threshold_analysis.py` | The script analyses PetBERT review thresholds from labelled predictions. ML now fits the review cut at calibration; retire the script and its test, or re-point them at the new scores | Its conclusions apply to the old scores only | no | open |
| RC-09 | backend | `backend/app/services/gcp_batch_service.py:48-62`, `backend/app/services/s3_service.py:84-99`, `backend/app/services/job_processor.py:79-85, 209, 375-378, 555-556` | `petbert_summary.json` / `prediction_method_counts`: the worker never wrote `scan_output/petbert_summary.json` (that was already so before this change). Any counts would now be keyed on `term_head` and `rejected_by_case_presence`. Confirm the missing file is handled; no change is required | Verification only | no | open |
| RC-10 | backend / frontend | `backend/app/services/ingestion_service.py:680-683`, `frontend/src/components/AdminQueue/AdminQueue.tsx:98, 130` | The high / medium / low confidence bands (at least 0.8, 0.5 to 0.8, below 0.5) and the average confidence are computed on `confidence`, which is now a term probability, not a cosine similarity. Decide whether the bands and the labels still say what the admin queue means by them | The numbers shift with the new scale | no | open |
| RC-11 | .github | `.github/dependabot.yml:3-19` | The comment says transformers 5.x, tokenizers 0.21 and huggingface_hub 1.0 are not to be taken, and the `ignore` rules block them. `ml/requirements.txt` and `ml-worker/requirements.txt` now pin transformers 5.18.0, tokenizers 0.23.2 and huggingface_hub 1.33.0, so the rules contradict the pins and would hide security updates for the versions in use. Drop the three ignores and rewrite the comment (a backbone or library bump is still an ML task: it changes every generation's scores) | The config and the pins disagree | no | open |
| RC-12 | backend / deploy | `backend/app/config.py:42` and `.env.example:78` (`GCP_BATCH_MACHINE_TYPE=n1-standard-4`), `backend/app/services/job_processor.py:21` (`ML_WORKER_TIMEOUT = 600.0`), the ml-worker environment | Capacity, not measured on the deployment machines. The worker uses `TORCH_NUM_THREADS` threads (default `min(8, CPU count)`; `os.cpu_count()` ignores container CPU quotas, so set it explicitly in a limited container). The default batch machine gives 4 threads, slower than the measured 8-thread figures (0.72 s per report chunked, 0.41 s single call). The HTTP path is bounded by `ML_WORKER_TIMEOUT`: an upload of roughly 850 to 1,500 reports or more times out on `/predict`; large uploads belong on the batch path. Set `TORCH_NUM_THREADS` to the machine's vCPUs, and consider a larger machine type or a longer timeout | Throughput and timeouts differ from the old worker | no | open |
| RC-13 | .github | `.github/workflows/ci.yml:97-101` | Nothing required: torch 2.9.1 CPU, `ml/requirements.txt` and `pytest ml` are unchanged and sufficient. Optional: also install `fastapi`, `httpx` and `python-multipart` so `ml/tests/test_worker_app.py` and the app case of `ml/tests/test_worker_image.py` run in CI (they skip without them; they passed 3/3 with those installed outside the venv) | Without them the HTTP worker's routes are not exercised in CI | no | open |
| RC-14 | .env.example, backend | `.env.example:71, 88`; `backend/service.yaml:135`; `docs/GCP_BATCH_SETUP.md:70-74, 118` | Prose says "PetBERT inference". The image name `petbert-batch` is only a tag and needs no change | Naming | no | open |
| RC-15 | docs | `docs/GCP_BATCH_SETUP.md:43-50, 132-134` | Lines 43-50 upload `petbert/` to `models/petbert` and `ml/labels` to `models/labels` (the old flat layout): the new layout is the generation directory uploaded to `models/<folder>/`. Lines 132-134 name the methods `low_confidence`, `unidentified_cancer`, `embedding`, `label_presence`: now `term_head` and `rejected_by_case_presence`. Following the old upload steps yields a bundle the worker refuses at start-up | Wrong instructions for the deploy | no (the deploy steps in `ml-worker/CHANGES.md` are correct) | open |
| RC-16 | docs | `docs/DATA_PIPELINE.md:49, 101-104`; `docs/user_guide.md:410`; `doc-site/src/user-guide.md:410` | `method` is `embedding` / `low_confidence` / `empty` and the review auto-flagging by confidence, margin and `low_confidence`: replace with the two new methods; the backend no longer gates review at ingest | Method names and the gate are out of date | no | open |
| RC-17 | docs | `docs/handoff/HANDOFF.md:77-81, 133-135, 338, 401-408`; `doc-site/src/handoff.md` (same lines); `docs/handoff/future_plans.md:202, 211`; `doc-site/src/future-plans.md:202, 211` | `petbert_pipeline/`, `labels/` embeddings, cosine similarity, `embedding_min_sim`, `low_confidence`, the `ml-worker/` tree with `Dockerfile.batch`, and "PetBERT embeddings tuned against a label catalog": describe the joint ModernBERT model and the bundle layout, link `ml/documentation/` | The pipeline described no longer exists | no | open |
| RC-18 | docs | `docs/current-architecture.md:19, 23, 64`; `docs/data-processing.md:66-67, 129` | "ml/ volume (PetBERT models + training code)", "PetBERT inference", "PetBERT categorization"; `data-processing.md` also links a `ml/documentation/petbert-pipeline.md` that does not exist (a dangling link from before this change) | Naming and a dead link | no | open |
| RC-19 | README / doc-site | `README.md:17, 208`; `doc-site/src/introduction.md:22`; `docs/user_guide.md` and `doc-site/src/user-guide.md` lines 17, 329, 340, 351, 372, 407, 411-412, 419, 518, 546, 556 | "PetBERT (110M-param BERT pretrained on veterinary EHR data)", "PetBERT ML classification service", the glossary entry: the model is now a ModernBERT fine-tuned on this project's labelled reports. Decide whether the user-facing name stays "PetBERT" for continuity or changes | The model name is out of date | no | open |
| RC-20 | frontend | `frontend/src/components/shared/pipelineStages.ts:4, 9` | The UI label `'Running PetBERT inference'` | Naming | no | open |

## Deploy order

1. ML adopts the first generation on the ML side (`promote.py --adopt --reason ...`), which archives
   the previous `current/` whole. Done by ML before the rest.
2. RC-01: export the bundle, upload it as a new folder, build and push the three images under new
   tags.
3. RC-01 and RC-02: point the deployment at the new image tag and the new folder together; rebuild
   the local `ml-worker` service.
4. Check `/health` (or the first job's `source_version`) reports `gen-20261009T224439Z`, and run one
   small upload.
5. The rest of the table, in any order.

Rollback: re-deploy the previous image tag and select the previous bundle folder. The old generation
is also archived whole by the adoption, and the old worker can be rebuilt from the git tag
`pre-restructure-2026-10-09`.

## Reference (ML side)

Commands, run from the repository root with `ml/.venv/Scripts/python.exe`:

```bash
# Tar the production generation (spec check first) and write its .sha256 sidecar
ml/.venv/Scripts/python.exe ml/scripts/handoff.py export-bundle --generation current

# Which generation is current, and what it was adopted for
ml/.venv/Scripts/python.exe ml/scripts/generations.py status

# The worker-side tests (the HTTP-route tests skip without fastapi, httpx and python-multipart)
ml/.venv/Scripts/python.exe -m pytest ml/tests/test_worker_format.py ml/tests/test_worker_parity.py ml/tests/test_worker_app.py ml/tests/test_worker_image.py -q
```

`export-bundle` writes `ml/output/handoff/outbox/bundles/bundle_<generation_id>.tar.gz` and its
`.sha256`. The tarball wraps the generation in one top-level folder (`current/`): extract it and
upload that folder's **contents**. `sync.py publish-model` is not this layout: it writes
`generations/<id>/` plus `CURRENT.json` for the ML machines' own sync
([sync-with-s3.md](how-to/sync-with-s3.md)).

Where each fact lives:

| Question | Doc |
|---|---|
| What the model is and how it decides | [concepts/report-mapping.md](concepts/report-mapping.md) |
| The bundle layout, manifest and spec check | [concepts/generations.md](concepts/generations.md) |
| The worker contract (bundle root, input, response, capacity) | [reference/handoff-contracts.md](reference/handoff-contracts.md#the-worker-bundle-contract-worker_formatpy) |
| Every worker file that changed, deploy and rollback | [ml-worker/CHANGES.md](../../ml-worker/CHANGES.md) |
| The review rule and `review_cut` | [concepts/coding.md](concepts/coding.md#the-bronze-review-rule) |
| Why | [ADR 0008](history/decisions/0008-joint-modernbert-report-mapping.md), [ADR 0009](history/decisions/0009-bronze-restructure-and-lifecycle.md), the [restructure plan](history/plans/restructure-plan.md) |

Earlier change requests, implemented and kept as history:
[audit list](history/plans/audit-list-change-request.md),
[ml-worker bundle](history/plans/ml-worker-change-request.md).
