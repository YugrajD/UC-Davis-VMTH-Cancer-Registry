# ml-worker changes

`ml-worker/` is the inference service the backend calls to turn uploaded report text into report-mapping predictions: `app.py` is the local HTTP worker (`POST /predict`, `GET /health`), `batch_predict.py` runs the same prediction once as a job (GCP Batch), and `s3_batch_entrypoint.py` wraps it for ECS Fargate (S3 in, S3 out). All three load one report-mapping generation directory (the "bundle") and import the model code from `ml/`; they contain no model logic of their own.

**Date:** 2026-10-09. **Branch:** `report-mapping-modernbert` (renamed from `bronze-retrieve-rerank` on 2026-10-13).
**Backend change required:** see `ml/documentation/restructure-change-request.md`.
Context: one jointly trained BioClinical-ModernBERT (a term head and a neoplasm gate head over the whole report) replaces the PetBERT four-stage pipeline ([restructure-plan.md](../ml/documentation/history/plans/restructure-plan.md), decisions 10-12). The backend contract is kept: it still sets `MODEL_PATH=<bundle>/petbert`, and the worker still only uses the parent directory.

## What changed

| File | Change | Reason |
|---|---|---|
| `app.py` | Loads the generation once with `predict.load_generation(bundle_root, device)`; `/predict` scores each row's `Text` with `predict.predict_texts`; sets torch threads from `TORCH_NUM_THREADS`; title and docstring renamed; the unused `DEFAULT_*_THRESHOLD` constants are gone | The PetBERT sections / embedding-cache / classifier path no longer exists |
| `batch_predict.py` | Same loading and scoring as `app.py`; docstring says what is verified now | Same |
| `s3_batch_entrypoint.py` | Docstring and one comment only (bundle = generation directory, spec check instead of fingerprint). Code unchanged | Statements about `petbert/`, `labels/`, `checkpoints/` and the fingerprint were false |
| `Dockerfile` | Installs CPU torch 2.9.1 from the CPU index before the other requirements; comment updated | The image must not pull the CUDA wheels |
| `Dockerfile.batch`, `Dockerfile.ecs` | Same CPU torch install; the `COPY` set is now `ml/config.py`, `ml/io_utils.py`, `ml/report_mapping/` (whole package), `ml/generations/__init__.py` + `manifest.py`, `ml/handoff/__init__.py` + `worker_format.py`. `ml/taxonomy/`, the rest of `ml/generations/` and the rest of `ml/handoff/` are no longer copied | The new inference path reads term -> (group, code) from the bundle's own `term_info.json`, so it never imports the taxonomy; `tests/test_worker_image.py` runs the worker on exactly this file set |
| `requirements.txt` | transformers 5.18.0, tokenizers 0.23.2, huggingface_hub 1.33.0, safetensors 0.8.0 (were 4.46.3 / 0.20.3 / 0.36.2 / 0.7.0); torch stays 2.9.1; other pins unchanged | In step with `ml/requirements.txt`: the generation was trained and is served on these versions |

Files outside `ml-worker/` the worker imports and that changed: `ml/handoff/worker_format.py` (rewritten: bundle root, flat upload text, thread count, response rows), `ml/report_mapping/inference/predict.py` (new public `predict_texts`; `predict_frame` delegates to it).

## The bundle

A bundle is a generation directory, the same one `current/` holds (`handoff.py export-bundle` tars it, `sync.py publish-model` publishes it):

```
<bundle>/
  encoder/          HF ModernBERT weights (model.safetensors), tokenizer files, config.json   (replaces petbert/)
  head.pt           term head (fp32)
  gate_head.pt      gate head (fp32)
  terms.json        term vocabulary in head order
  term_info.json    term -> [group, code]
  thresholds.json   gate_t, tau, rho, review_cut, max_fp_share, review_target_share
  manifest.json
```

At start-up `load_generation` runs `report_mapping.generation.verify_spec` and the worker refuses to start (or, for the batch job, fails) unless: every file matches the hash in `manifest.json` and nothing listed is missing; the manifest is a `report_mapping_generation` of schema 2; its `architecture` block equals the code's (name `joint-modernbert-v1`, text spec version, pooling, heads, decision rule); `weights_sha256` matches the weight files; `n_terms` matches `terms.json` (the head width is checked again when the model loads); and the thresholds are `calibrated` for those same weights. This spec check replaces the embedding fingerprint. A bundle from the PetBERT era fails it, and a new bundle fails the old image: image and bundle must be switched together.

Where the bundle is: unchanged. The model-path variable (`PETBERT_MODEL_PATH` for `app.py`, `MODEL_PATH` for `batch_predict.py` and the ECS entrypoint) must be `<bundle>/petbert`, and the bundle root is its parent directory. The `petbert` leaf is only a name kept for the backend; the directory does not exist in the new layout and the path need not exist. If both variables are set they must name the same bundle. The five other path variables of the old worker (`LABELS_CSV_PATH`, `GROUP_CLASSIFIER_PATH`, `CASE_PRESENCE_CLASSIFIER_PATH`, `LP_THRESHOLDS_JSON_PATH`, `UNCOMMON_GROUPS_PATH`) are not read any more. The old worker refused to start when one of them pointed outside the bundle; the new one ignores them wherever they point, because there is nothing left for them to mix.

## Input mapping

Dataset A is `anon_id` plus one `Text` column. Each row's `Text` (stripped, nothing else) is the report text: flat, with no heading and no section columns. The training text is the report's columns each under its heading; headings were tried on uploads and measured on the calibration partition against the full headed report text (restructure plan, WP4 note):

| Upload text | Good | Slight | CO | FP | FN | G+S |
|---|---|---|---|---|---|---|
| Flat (shipped), pp vs headed | -0.10 | -0.01 | +0.43 | +0.50 [+0.20, +0.85] | -0.82 | -0.11 |
| One heading, pp vs headed | | | | +0.64 | | |

The cost of flat text is about half a point of false positives, nearly offset by fewer false negatives (G+S -0.11 pp). A row with empty `Text` gets no prediction rows and so is absent from `predictions`. Extra upload columns are ignored. `max_length` is 4096 tokens (the bundle's `model.max_length`; the PetBERT worker used 512).

## Response schema

`POST /predict` returns `{"predictions": [...], "input_rows": <prediction rows>}`; `batch_predict.py` writes the list as `predictions.json`; `GET /health` returns `{"status": "ok", "source_version": <generation_id>}`. Each prediction has the same keys as before: `anon_id`, `original_text`, `predicted_term`, `predicted_group`, `predicted_code`, `confidence`, `method`, `source_version` (the generation id). A case with a second term gets `"1) a 2) b"` strings in each field, byte-compatible with the previous worker.

What the values mean now:

- `method` is `term_head` (the gate accepted the case; the rank-1 term, plus rank 2 when its probability is at least `tau` and at least `rho` times rank 1) or `rejected_by_case_presence` (the gate probability is below `gate_t`: one `Non-Cancer` row with group `Non-Cancer`, empty code). `embedding`, `low_confidence`, `unidentified_cancer`, `label_presence` and the other PetBERT methods are not produced any more.
- `confidence` is the term's sigmoid probability to two decimals; a rejected row has `0.00`. It is not a cosine similarity and not comparable with old values.
- `predicted_group` and `predicted_code` come from the bundle's `term_info.json`.
- `group_prob` and the gate probability are in the offline predictions CSV only; the worker response never carried them.

## Runtime

Pins: torch 2.9.1 (CPU wheel), transformers 5.18.0, tokenizers 0.23.2, huggingface_hub 1.33.0, safetensors 0.8.0, Python 3.12. The images are CPU, fp32 (the library autocasts to bf16 only on CUDA). Torch threads: `TORCH_NUM_THREADS` if set (a positive integer, else the worker refuses to start), otherwise `min(8, CPU count)`; 8 is what the timings below used.

CPU timing on calibration reports (torch 2.9.1+cpu, 8 threads): 0.72 s per report in chunked calls (about 12 minutes per 1,000 reports), 0.41 s per report in a single call (single-call p95 0.80 s; chunk-level p95 1.00 s).

Capacity notes (not measured on the deployment machines):
- The worker uses `TORCH_NUM_THREADS` threads, default `min(8, os.cpu_count())`. `os.cpu_count()` ignores container CPU quotas, so set `TORCH_NUM_THREADS` explicitly in a CPU-limited container. The backend's default batch machine (`n1-standard-4`) gives 4 threads, so expect slower than the 8-thread figures above.
- The local HTTP path is bounded by the backend's `ML_WORKER_TIMEOUT` (600 s): at 0.4-0.7 s per report, an upload of roughly 850-1,500 reports or more times out on `/predict`. Large uploads belong on the batch path.

CPU against GPU on the whole calibration partition (CPU fp32 production pins versus GPU bf16): Good -0.03, Slight +0.05, CO +0.05, FP 0.00, FN -0.07 pp; 21 of 5,828 cases change decision. torch 2.9.1 works with transformers 5.18.0, so CI (which hard-codes torch 2.9.1) needs no change for torch.

## Not changed

- Endpoints, `/predict` request format (multipart `file`, `.csv` or `.xlsx`, 50 MB limit, UTF-8 as the backend writes it) and error codes.
- Env var names: `PETBERT_MODEL_PATH`, `MODEL_PATH`, `JOB_ID`, `INPUT_CSV_PATH`, `OUTPUT_DIR`, and the ECS entrypoint's `S3_BUCKET`, `JOB_ID`, `MODEL_FOLDER`, `AWS_REGION`, `AWS_S3_ENDPOINT_URL`. New and optional: `TORCH_NUM_THREADS`.
- `s3_batch_entrypoint.py` arguments, S3 layout (`models/{MODEL_FOLDER}/` downloaded whole), `MODEL_PATH=<model_root>/petbert`, and the outputs it uploads.
- Docker image tag names in the build-command comments.

## Deploy order and rollback

1. Adopt on the ML side (`promote.py --adopt --reason ...`, plan WP8), which archives the previous `current/` whole under `ml/output/archive/<date>_<description>/`.
2. Export the new `current/` (`handoff.py export-bundle`), extract the tarball (it wraps the generation in a `current/` directory), and upload the generation's contents to the bucket as a new bundle folder (GCS or S3 `models/<folder>/`, so that `models/<folder>/manifest.json` exists); do not overwrite the folder holding the previous bundle. `sync.py publish-model` is not this layout: it writes `generations/<id>/` plus `CURRENT.json` for the ML machines' own sync.
3. Build and push the three images from this branch under new tags (nothing here builds them).
4. Point the deployment at the new image tag and the new folder together (`GCP_BATCH_IMAGE_URI`, the ECS task definition, the local compose service); apply the backend items in the change request.
5. Check `/health` (or the first job's `source_version`) reports the new generation id, and run one small upload.

An old image refuses a new bundle and a new image refuses an old bundle, so a job started with the wrong pair fails at start-up instead of predicting; folders that still hold a previous-generation bundle appear in the backend's model-folder list and must not be selected with the new image. Rollback: re-deploy the previous image tag and select the previous bundle folder (the old generation is archived whole by adopt, so it can also be re-exported from the archive), or rebuild the old worker from the git tag `pre-restructure-2026-10-09`.
