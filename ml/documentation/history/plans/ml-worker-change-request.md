# Change request for the backend: ml-worker reads one report-mapping bundle

**From:** ML. **To:** the backend developer. **When:** deploy together with the ML cutover (WP13 of
[ml-rewrite-plan.md](ml-rewrite-plan.md)); the new worker imports the post-cutover `ml/` layout.

## What changes in the worker

- **One bundle, verified at startup.** The worker loads one report-mapping generation directory (the
  "bundle") and refuses to start if a file its `manifest.json` lists is missing or changed, or if the
  backbone does not match the recorded embedding fingerprint. There are no optional stages any more: the
  case-presence gate, every per-group label-presence head, the per-group thresholds and the uncommon-group
  list are always used.
- **Where the bundle is.** Its root is the parent of the model path variable (`PETBERT_MODEL_PATH` for the
  HTTP worker, `MODEL_PATH` for GCP Batch), which must be `<root>/petbert`. The other path variables can be
  left unset. If one is set, it must point at its place inside the same bundle, or the worker refuses:

  | Variable | Must be |
  |---|---|
  | `LABELS_CSV_PATH` | `<root>/labels/labels.csv` |
  | `GROUP_CLASSIFIER_PATH` | `<root>/checkpoints/group_classifier_best.pt` |
  | `CASE_PRESENCE_CLASSIFIER_PATH` | `<root>/checkpoints/case_presence_classifier.pt` |
  | `LP_THRESHOLDS_JSON_PATH` | `<root>/checkpoints/label_presence/lp_thresholds.json` (moved from `checkpoints/`) |
  | `UNCOMMON_GROUPS_PATH` | `<root>/checkpoints/uncommon_groups.txt` |

- **Response.** Same keys and the same `"1) … 2) …"` numbered format as today, plus `source_version` on every
  prediction: the `generation_id` of the model that produced it (for example `gen-20260927T003905Z`).
  `GET /health` also returns `source_version`. Storing it with each report-based code is what lets a code be
  traced to the model that made it (icd-mapping-strategy.md).
- **Behaviour.** Reports are embedded at `max_length` 512, the length the models are fingerprinted with (the
  old worker used 256), and the uploaded CSV is read as UTF-8, which is how the backend writes it (the old
  worker decoded it as latin-1 for the model). Expect small differences from today's cloud predictions.
- **Image.** Python 3.12, pins aligned with `ml/requirements.txt`. The batch image copies `ml/config.py`,
  `ml/io_utils.py`, `ml/taxonomy/`, `ml/report_mapping/`, `ml/generations/` and `ml/handoff/`.

## What the backend needs to change

1. **Model upload to GCS.** Upload the whole generation directory, keeping its layout. ML produces it with
   `handoff.py export-bundle` (a tarball plus a `.sha256` sidecar; the tarball holds one top-level folder,
   the generation directory):

   ```
   petbert/                                  (whole directory)
   labels/labels.csv
   checkpoints/case_presence_classifier.pt
   checkpoints/group_classifier_best.pt
   checkpoints/label_presence/*.pt           (new: one per group)
   checkpoints/label_presence/lp_thresholds.json   (moved)
   checkpoints/thresholds.json               (new)
   checkpoints/uncommon_groups.txt
   manifest.json                             (new)
   ```

2. **`gcp_batch_service.py` setup runnable.** Replace the per-file downloads (and their `|| echo` fallbacks,
   which the worker no longer tolerates) with one recursive copy of the bundle into a local directory, then
   set `MODEL_PATH=<local bundle>/petbert`. Drop the other path variables, or set them to the paths in the
   table above.
3. **`docker-compose.yml`, `ml-worker` service.** Default `PETBERT_MODEL_PATH` to
   `/ml/output/report_mapping/current/petbert` and remove the other path defaults. Today's non-empty
   `GROUP_CLASSIFIER_PATH` default (`/ml/output/checkpoints/group/...`) points outside the bundle and would
   stop the worker from starting.
4. **Optional:** store `source_version` with each report-based code, and log `/health`'s `source_version`
   after each deploy.
