# Run inference

How to produce predictions for the whole report corpus with `predict.py`, where they land and how
the embedding cache behaves. For maintainers. The stages that turn embeddings into predictions are in
[report-mapping.md](../concepts/report-mapping.md); every flag is in
[scripts-and-flags.md](../reference/scripts-and-flags.md).

## Predict with a generation

```bash
# the production generation
ml/.venv/Scripts/python.exe ml/scripts/predict.py --generation current --device cuda --local-only

# a trained and calibrated candidate
ml/.venv/Scripts/python.exe ml/scripts/predict.py --generation candidate --device cuda --local-only
```

`--generation` is `current`, `candidate`, or a literal generation directory. The script reads
`config.REPORT_CSV`, embeds the report sections (or loads the cached embeddings), runs the gate, group
and label-presence stages, then keyword correction and lipoma rescue ([stage details](../concepts/report-mapping.md)), with the generation's own thresholds, and writes one predictions CSV. It
refuses a generation whose manifest hashes or embedding fingerprint do not check out, and one whose
`calibration.status` is not `calibrated` ([train-and-promote.md](train-and-promote.md), calibrate step).

| Flag | Use |
|---|---|
| `--embed-only` | Fill the embedding cache and stop; no classification, no predictions file. Allowed on an uncalibrated candidate that already has its head checkpoints. |
| `--model` | A Hugging Face checkpoint directory or name to embed with instead of the generation's own `petbert/`. For experiments only: heads trained on one backbone do not match embeddings from another. |
| `--out` | Write the predictions CSV to this path instead of the default. |
| `--local-only` | Do not download from Hugging Face; use only the local cache and directories. |
| `--cache-dir` | Use this directory as the embedding cache instead of `config.EMBEDDING_CACHE_DIR`. |
| `--device` | `auto` (default: cuda, then xpu, then mps, then cpu), `cpu`, `cuda`, `mps` or `xpu`. |

## Where predictions land

By default `output/predictions/<generation_id>_predictions.csv` (`config.PREDICTIONS_DIR`), UTF-8,
one row per predicted diagnosis:

`case_id, diagnosis_index, predicted_term, predicted_group, predicted_code, case_presence_prob,
confidence, group_prob, method, generation_id`

`generation_id` stamps which generation produced the row. `method` is `label_presence`,
`lipoma_rescue`, `rejected_by_case_presence` (the gate said non-cancer; term and group read
`Non-Cancer`) or `unidentified_cancer` (cancer, but no group; term and group read `Unidentified
Group`, no code). A case the pipeline leaves empty gets no row. The file holds report-derived
predictions, not report text. It is the input to `evaluate.py`, `promote.py` and
`code_cases.py`; see [evaluation.md](../concepts/evaluation.md) and
[coding.md](../concepts/coding.md). The `predictions` set is synced to S3
([sync-with-s3.md](sync-with-s3.md)).

## The embedding cache

Embedding the full corpus is the slow part (about 9 minutes on the RTX 5070 Ti); classification
takes about 2 minutes once the embeddings exist. Embeddings are cached in
`output/report_mapping/embedding_cache/<key>.npz`, one file per distinct combination of inputs.

The key is the sha256 of three things: the bytes of `report.csv`, the bytes of the generation's
`labels/labels.csv`, and the generation's embedding fingerprint (backbone hash, section-spec
version, `max_length`). So the cache is keyed on content, not file times:

- A different backbone, a changed section spec, a new `report.csv` or an edited labels file gives a
  new key. The old entry is simply not used; nothing needs invalidating by hand.
- The same inputs give the same key on any machine, and on a hit nothing is embedded.
- Training the heads builds the cache under the same key, so the predict step after training is a hit.
- A missing or unreadable cache file counts as a miss and is rebuilt.
- Calibration never embeds. It needs the cache to exist, and stops with a message to run
  `predict.py --embed-only` if it does not.

To force a re-embed without touching the real cache, point `--cache-dir` at an empty scratch
directory. Old entries that a promoted generation can no longer use are moved to the archive at
promotion. The cache is not synced to S3; it rebuilds locally after a `pull-model`.

_Last verified against code: 2026-09-29_
