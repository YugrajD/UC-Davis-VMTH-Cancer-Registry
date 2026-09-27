# Training Guide

The full retraining cycle, with exact commands and runtimes. For the model design and calibration
mechanics see [report-mapping.md](report-mapping.md); for the promotion/trigger logic see
[generations.md](generations.md); for the strategy behind retraining at all see
[icd-mapping-strategy.md](icd-mapping-strategy.md), "The full cycle".

Most of this cycle is `retrain_cycle.py` in one call ([generations.md](generations.md)). This page
spells out each of its steps individually — use it when you need to stop between steps, rerun just
one, or do a cold start (a backbone retrain, which `retrain_cycle.py` also drives with `--backbone`).

## Prerequisites

- **Python venv:** `ml/.venv/Scripts/python.exe` (Windows) / `ml/.venv/bin/python` (macOS/Linux).
  Every `scripts/*.py` adds `ml/` to `sys.path` itself — no `PYTHONPATH` needed.
- **Device:** `--device cuda` on the RTX 5070 Ti (PyTorch 2.6+ / CUDA 12.8 for Blackwell sm_120
  support). `--device auto` (the default on most scripts) picks cuda → mps → xpu → cpu.
  `train.py`/`predict.py`/`calibrate.py` all accept it.
- **A split**: `config.DEFAULT_SPLIT_ID` (`three-way-v1`) must exist (`scripts/split.py
  import-legacy` then `create --parent legacy-80-20 --id three-way-v1`, one-time). Every training
  and calibration command below defaults `--split` to it.
- **A silver generation** to train on: an existing `silver_id` (`silver-0-legacy`, or a fresh one
  from Step 1).

## Step 1 — Diagnosis mapping (skip if the silver generation you need already exists)

```bash
ml/.venv/Scripts/python.exe ml/scripts/map_diagnoses.py run --id silver-1
```

Runs the full cascade + cleanup over `config.DIAGNOSES_CSV` and writes a new, immutable silver
generation (`output/silver/silver-1/`). See [diagnosis-mapping.md](diagnosis-mapping.md). Needs LM
Studio running locally. Runtime: tens of minutes on a full corpus (cascade + cleanup).

## Step 2 — Corrected annotations

```bash
ml/.venv/Scripts/python.exe ml/scripts/code_cases.py corrected --silver silver-1 --split three-way-v1
```

Builds the report mapping's training labels for the train partition: gold-train where it exists,
silver elsewhere ([coding.md](coding.md)). Writes `config.CORRECTED_ANNOTATIONS_CSV`. This is the
`--labels` value every training/calibration command below reads.

## Step 3 — Adapt the backbone (cold start only)

Skip this step and Step 4 unless the backbone itself needs to change (a section-spec change, or a
fresh contrastive adaptation). Heads-only retraining (the common case) reuses the current
generation's `petbert/` untouched.

```bash
ml/.venv/Scripts/python.exe ml/scripts/train.py --stage backbone \
    --labels ml/output/coding/corrected_annotations.csv --split three-way-v1 \
    --device cuda --out candidate --local-only
```

Hyperparameters come from `report_mapping.training.recipe.BACKBONE` (see
[report-mapping.md, Training](report-mapping.md#training)). Writes the full HuggingFace checkpoint to
`candidate/petbert/`.

## Step 4 — Rebuild the embedding cache (cold start only)

```bash
ml/.venv/Scripts/python.exe ml/scripts/predict.py --generation candidate --embed-only --device cuda
```

A new backbone changes the embedding fingerprint, so the content-hash cache automatically misses and
this rebuilds it (`output/report_mapping/embedding_cache/<key>.npz`) — no manual invalidation step;
the cache is keyed on content hash, not mtime (see [report-mapping.md](report-mapping.md)).
**Runtime: ~9 minutes** on the RTX 5070 Ti for the full corpus.

## Step 5 — Train the heads

```bash
ml/.venv/Scripts/python.exe ml/scripts/train.py --stage heads \
    --labels ml/output/coding/corrected_annotations.csv --split three-way-v1 \
    --seed 42 --device cuda --out candidate --local-only
```

Trains CasePresence, Group and LabelPresence in sequence against `candidate/`'s backbone (the
current generation's, unless Step 3 ran first), reusing one embedding cache. Hyperparameters come
from `report_mapping.training.recipe` (see [report-mapping.md, Training](report-mapping.md#training)).

To train one head only, pass `--stage case-presence`, `--stage group` or `--stage label-presence`
instead of `heads`. Writes a candidate manifest with `calibration.status: "pending"`.
**Runtime: ~5 minutes** total on the RTX 5070 Ti (heads-only, cache already built).

## Step 6 — Calibrate thresholds

```bash
ml/.venv/Scripts/python.exe ml/scripts/calibrate.py --generation candidate \
    --labels ml/output/coding/corrected_annotations.csv --split three-way-v1
```

Fits every threshold on the split's **calibration** partition, from cached embeddings (never
re-embeds): per-LP thresholds first (0.05-step grid, F1-maximizing), then gate/group/tail jointly
(a small fixed grid, per-code G+S objective) — see [report-mapping.md](report-mapping.md#calibration)
for the exact grids. Writes `checkpoints/thresholds.json` and
`checkpoints/label_presence/lp_thresholds.json`, and flips the manifest's `calibration.status` to
`"calibrated"`. **Runtime: ~2 minutes** on the RTX 5070 Ti.

## Step 7 — Predict and evaluate

```bash
ml/.venv/Scripts/python.exe ml/scripts/predict.py --generation candidate --device cuda
```

Runs every stage over `config.REPORT_CSV` and writes a stamped predictions CSV
(`output/predictions/<generation_id>_predictions.csv`). **Runtime: ~2 minutes** on the RTX 5070 Ti
(embeddings already cached).

```bash
ml/.venv/Scripts/python.exe ml/scripts/evaluate.py silver --predictions PATH --labels silver-1 --split three-way-v1 --partition test
ml/.venv/Scripts/python.exe ml/scripts/evaluate.py gold --predictions PATH --silver silver-1 --split three-way-v1
```

`evaluate.py silver` is the cheap ruler (bronze vs the labels table on the test partition —
[evaluation.md](evaluation.md)); `evaluate.py gold` is the four gold-eval results with confidence
intervals, once gold-eval exists.

## Step 8 — Promote

```bash
ml/.venv/Scripts/python.exe ml/scripts/promote.py \
    --candidate-predictions ml/output/predictions/<candidate_id>_predictions.csv \
    --incumbent-predictions ml/output/predictions/<current_id>_predictions.csv
```

Recommends only, by default. Add `--apply` to carry it out (archives `current/`, swaps `candidate/`
in) once you've reviewed the recommendation. See [generations.md](generations.md#promotion-rule-promotepy)
for the rule and the retraining triggers it checks.

## Doing all of the above in one call

```bash
ml/.venv/Scripts/python.exe ml/scripts/retrain_cycle.py --silver silver-1 --device cuda --local-only
```

Runs Steps 2, 5–8 as subprocesses (fresh processes free GPU memory between steps), stopping early if
gold-eval doesn't exist yet or no retraining trigger is met (`--force` overrides the trigger check;
`--backbone` adds Steps 3–4 first for a cold start). See
[generations.md](generations.md#retrain_cyclepy--the-local-lane-in-one-go).

## Retraining a single head

When only one head changes, the backbone and its embedding cache are still valid — skip Steps 3–4
and rerun only the affected `--stage`, then Step 6 (calibration depends on every head, since the
gate/group/tail grid search categorizes with whichever heads are loaded) and Step 7.

## Troubleshooting

**A run scores far below the ~61.76% eval-half reference.** Almost always a stale or mismatched
generation — check `manifest.json`'s `embedding_fingerprint` and `calibration.status`;
`load_generation` refuses to load a mismatched or uncalibrated generation, so this should surface as
an error rather than silently-wrong numbers, but a placeholder `thresholds.json` copied from the
parent generation (before calibration) will still *load* successfully.

**`GenerationError: embedding fingerprint mismatch`.** The backbone, section spec, or inference
`max_length` changed since this generation's classifiers were trained — a real cold-start boundary,
not a bug. Start from Step 3.

**GroupClassifier diverges to all-1s.** Missing `weight_decay=1e-3` or `max_class_weight=50` — both
are required guards in `recipe.GROUP`, not optional tuning knobs.

**`GuardViolation` from `train.py`/`calibrate.py`/`promote.py`.** One of the leakage guards failed
([generations.md](generations.md#leakage-guards-guardspy)) — read the violation message (case
counts + a few example case_ids) before overriding anything; it names exactly which invariant broke.
