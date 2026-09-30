# Train and promote a generation

Exact commands to train a candidate report-mapping generation, calibrate it, predict with it and
promote it. For maintainers running a retrain. The design behind each step is in
[report-mapping.md](../concepts/report-mapping.md); the promotion rule, retraining triggers and
archive are in [generations.md](../concepts/generations.md); every flag is in
[scripts-and-flags.md](../reference/scripts-and-flags.md).

## Where things stand

Production `current/` is `gen-0-legacy`, trained on `silver-0-legacy` with the legacy hand-picked
thresholds. There is no `candidate/`, and no gold exists yet. So **`retrain_cycle.py` currently
prints `STOP: no gold-eval rows ...` and trains nothing**, and `promote.py` refuses for the same
reason (promotion is decided on gold-eval). Steps 1 to 6 below still run; the gold half of step 7, and step 8, need gold.
Runs on mock gold are test-only and their numbers are never quoted.

## Training order

```mermaid
flowchart LR
    bb["Backbone<br>(optional)"] --> cp["Case-presence head"]
    cp --> grp["Group head"]
    grp --> lp["Label-presence heads"]
    lp --> cal["Calibrate"]
    cal --> pred["Predict"]
    pred --> prom["Promote"]
```

`train.py --stage heads` runs the three head stages in the order shown. The backbone runs first and
only when the embeddings themselves must change (see [Cold start](#cold-start-a-new-backbone)).
Calibration fits every threshold and flips the candidate from `pending` to `calibrated`; the loader
refuses to predict with an uncalibrated generation.

## Prerequisites

- **Python:** `ml/.venv/Scripts/python.exe` on Windows, `ml/.venv/bin/python` on macOS and Linux
  (commands below use the Windows path). Set-up is in
  [new-machine-setup.md](new-machine-setup.md). Scripts add `ml/` to `sys.path` themselves.
- **Device:** `--device cuda` on the RTX 5070 Ti. `--device auto` (the default of `train.py`,
  `predict.py` and `retrain_cycle.py`) picks cuda, then xpu, then mps, then cpu. `calibrate.py`
  defaults to `cpu`, so pass `--device cuda` there if you want the GPU. `xpu` is a legacy Intel
  build that is still accepted.
- **A split:** `three-way-v1` (`config.DEFAULT_SPLIT_ID`) exists; every command below defaults to it.
- **A silver generation** to train on: an existing `silver_id` (`silver-0-legacy`) or a new one from
  step 1.
- **No `candidate/` directory.** `retrain_cycle.py` refuses to start if one exists; promote or delete
  it first. Running the manual stages into an existing `candidate/` reuses its `petbert/` and
  overwrites its heads.

## Route 1: `retrain_cycle.py`

The local lane in one call. Each step runs as its own process, so GPU memory is freed between steps.

```bash
ml/.venv/Scripts/python.exe ml/scripts/retrain_cycle.py --silver silver-0-legacy --device cuda --local-only
```

It stops early when there is no gold-eval (today), or when no retraining trigger is met (`--force`
trains anyway). `--backbone` adds a backbone retrain first. `--gold-csv PATH --export-id ID
--reviewer NAME` ingests a gold export first. It finishes with `promote.py` **without** `--apply`,
so nothing is promoted; you read the recommendation and apply it yourself (step 8). It never runs
the LLM cascade. The step list and trigger rules are in [generations.md](../concepts/generations.md).

## Route 2: stage by stage

Use this to stop between steps, rerun one step, or when the automatic route STOPs.

### 1. Diagnosis mapping (skip if the silver generation you need exists)

```bash
ml/.venv/Scripts/python.exe ml/scripts/map_diagnoses.py run --id silver-1
```

Writes a new immutable silver generation under `output/silver/silver-1/`. Needs LM Studio running
locally; takes tens of minutes on the full corpus. See
[diagnosis-mapping.md](../concepts/diagnosis-mapping.md).

### 2. Check whether a retrain is warranted

```bash
ml/.venv/Scripts/python.exe ml/scripts/generations.py status --silver silver-1
```

Prints `current/` and `candidate/` (ids, calibration status, silver, split) and which retraining
triggers are met. Note the `current/` generation id: step 7 needs it.

### 3. Corrected annotations

```bash
ml/.venv/Scripts/python.exe ml/scripts/code_cases.py corrected --silver silver-1 --split three-way-v1
```

Writes `output/coding/corrected_annotations.csv`: the training labels for the **train partition
only** (gold-train where it exists, silver elsewhere; see [coding.md](../concepts/coding.md)).
This is the `--labels` value for training. It is never the `--labels` value for calibration.

### 4. Train the heads

```bash
ml/.venv/Scripts/python.exe ml/scripts/train.py --stage heads \
    --labels ml/output/coding/corrected_annotations.csv --split three-way-v1 \
    --seed 42 --device cuda --out candidate --local-only
```

Trains case-presence, group and label-presence in turn into `candidate/`, on the current
generation's backbone (copied into `candidate/petbert/`), building the embedding cache on the way.
To train one head, pass `--stage case-presence`, `--stage group` or `--stage label-presence`
instead. The candidate's manifest starts as `calibration.status: pending` with placeholder
thresholds copied from `current/`. Leakage guards run first for `heads` and `backbone` stages and
abort on a violation ([generations.md](../concepts/generations.md)). Roughly 5 minutes with a warm
cache on the RTX 5070 Ti.

### 5. Calibrate

```bash
ml/.venv/Scripts/python.exe ml/scripts/calibrate.py --generation candidate \
    --labels silver-1 --split three-way-v1 --device cuda
```

Fits all thresholds on the split's calibration partition from the cached embeddings (it never
embeds; the cache must exist from step 4). **`--labels` is the silver id**, as `retrain_cycle.py`
does: the corrected table covers the train partition only, so calibration refuses it with
"labels ... have no rows on the 'calibration' partition". The grids and objective are in
[report-mapping.md](../concepts/report-mapping.md). Writes `checkpoints/thresholds.json` and
`checkpoints/label_presence/lp_thresholds.json` and sets `calibration.status: calibrated`. Roughly
2 minutes.

### 6. Predict with both generations

```bash
ml/.venv/Scripts/python.exe ml/scripts/predict.py --generation candidate --device cuda --local-only
ml/.venv/Scripts/python.exe ml/scripts/predict.py --generation current --device cuda --local-only
```

Each writes `output/predictions/<generation_id>_predictions.csv`. Skip the second if that file
exists. Details in [run-inference.md](run-inference.md).

### 7. Evaluate

```bash
ml/.venv/Scripts/python.exe ml/scripts/evaluate.py silver --predictions PATH --labels silver-1 --split three-way-v1 --partition test
ml/.venv/Scripts/python.exe ml/scripts/evaluate.py gold --predictions PATH --silver silver-1 --split three-way-v1
```

`evaluate.py silver` scores against the labels table on one partition and works today. Its
`--generation` defaults to `current`, and the run is refused when the predictions carry another
generation's id, so add `--generation candidate` for the candidate's predictions (same for
`evaluate.py gold`). `evaluate.py gold` needs gold-eval, so it cannot run for real until the reviewer
returns gold. Report Good and
Slight separately, with the split and n ([evaluation.md](../concepts/evaluation.md)).

### 8. Promote

```bash
ml/.venv/Scripts/python.exe ml/scripts/promote.py \
    --candidate-predictions ml/output/predictions/<candidate_id>_predictions.csv \
    --incumbent-predictions ml/output/predictions/<current_id>_predictions.csv
```

This step needs gold-eval and refuses today. Without `--apply` it only prints the recommendation (guards, gold-eval comparison, triggers). To
carry it out, rerun the same command with `--apply`: a winning candidate replaces `current/` and the
old generation moves whole to `output/archive/YYYY-MM-DD_<description>/`; a losing candidate is
deleted. Add `--description short-name` to name the archive folder, and `--publish` (with
`--apply`) to push the new `current/` to S3 afterwards, which needs AWS credentials
([sync-with-s3.md](sync-with-s3.md)). There is no restore-from-archive command.

```bash
ml/.venv/Scripts/python.exe ml/scripts/promote.py --candidate-predictions ... --incumbent-predictions ... --apply --publish
```

## Cold start: a new backbone

Only when the section spec or the backbone must change; heads-only retraining reuses `current/`'s
backbone. Train the backbone first, into the same candidate, then continue from step 4:

```bash
ml/.venv/Scripts/python.exe ml/scripts/train.py --stage backbone \
    --labels ml/output/coding/corrected_annotations.csv --split three-way-v1 \
    --device cuda --out candidate --local-only
```

`--stage backbone` defaults `--model` to `SAVSNET/PetBERT` (the base model, which must be in the
local Hugging Face cache with `--local-only`), not to `current/`'s adapted backbone; the other stages
default to the generation's own `petbert/`. The result is written to `candidate/petbert/`, and the
following heads stage reuses it. There is no separate embedding step: the new backbone changes the
embedding fingerprint, the cache misses, and heads training rebuilds it (about 9 minutes). With
`retrain_cycle.py`, use `--backbone`.

## Retraining one head

The backbone and cache stay valid, so rerun only that `--stage` (step 4), then calibrate (step 5:
the threshold search uses every head) and predict.

## Troubleshooting

**Scores far below the incumbent.** The loader refuses a fingerprint mismatch and any uncalibrated
generation, so a mismatched generation shows up as an error, not as wrong numbers. If you get scores
that are simply low, check that both runs use the same split, the same silver generation and the same
`evaluate.py` scope, and look at `manifest.json` (`calibration.status`) and `checkpoints/thresholds.json` for the
generation you scored.

**`GenerationError: embedding fingerprint mismatch`.** The backbone, section spec or `max_length`
changed after the classifiers were trained. Retrain from the backbone stage.

**`GenerationError: calibration.status is 'pending'`.** Run step 5. Only `calibrate.py` and
`predict.py --embed-only` may load an uncalibrated generation.

**Calibration: "no embedding cache".** Calibration never embeds. Run `predict.py --generation
candidate --embed-only`, or train heads (which builds the cache).

**Group head diverges to all ones.** `weight_decay=1e-3` and `max_class_weight=50` in
`report_mapping/training/recipe.py` are required guards, not tuning knobs.

**`GuardViolation`.** A leakage guard failed. The message gives case counts and a few example ids;
it names the invariant that broke. See [generations.md](../concepts/generations.md).

_Last verified against code: 2026-09-29_
