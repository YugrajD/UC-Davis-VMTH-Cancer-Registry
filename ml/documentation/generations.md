# Generations

Versioning infrastructure shared by every generation kind (splits, silver, report-mapping): a
manifest, immutability, leakage guards, promotion and retraining triggers. Package:
`ml/generations/` (`manifest.py`, `splits.py`, `guards.py`, `promote.py`, `triggers.py`). Entry
points: `scripts/split.py`, `scripts/generations.py`, `scripts/promote.py`,
`scripts/retrain_cycle.py`.

## Manifests (`manifest.py`)

Every generation directory (a split, a silver generation, `current/`/`candidate/`) gets a
`manifest.json`: the caller's fields plus `created_at` (ISO UTC), `git_sha` (the code's HEAD), and
`files` — a sha256 for every file under the directory. `verify_manifest` refuses a directory whose
listed files are missing or changed, or that holds a file the manifest doesn't list.
`update_manifest` changes top-level fields in place (used by `promote.py` to flip `status`) without
re-stamping `files`/`created_at`/`git_sha`.

## Splits (`splits.py`)

`config.SPLITS_DIR/<split_id>/{train,calibration,test}_cases.txt` + manifest. **Immutable once
written** — every writer refuses to overwrite an existing `<split_id>/`; `load_split` verifies the
manifest, then reads only the partition files it lists (a partition it doesn't list — e.g. no
calibration file in a legacy split — loads as an empty set).

```
ml/.venv/Scripts/python.exe ml/scripts/split.py create --parent legacy-80-20 --id three-way-v1
ml/.venv/Scripts/python.exe ml/scripts/split.py check --split three-way-v1
```

- `legacy-80-20` (46,652 train / 11,661 test; cancer cases stratified by first `matched_group`, no
  calibration partition) and `legacy-temporal` (56,905 / 1,408; drift check, no calibration
  partition either; the cutoff year isn't recorded by the legacy tooling — `cutoff_year: null`) are
  two split generations imported once, before cutover, from the pre-rewrite tree's flat split files
  (byte-identical). The one-time importer is gone; both splits are on disk to stay.
- `create --parent P --id ID` derives a **three-way split** from a two-way parent: train copied
  unchanged; the parent's test cases split by the **md5 half rule**
  (`int(hashlib.md5(case_id.encode()).hexdigest(), 16) % 2 == 0` → calibration/"sweep" half, else test/"eval"
  half — `generations.splits.in_sweep_half`). `three-way-v1` (parent `legacy-80-20`): 46,652 train /
  5,828 calibration / 5,833 test. This is the split every threshold is fitted on going forward: the
  sweep half fitted the legacy LP thresholds and is now the calibration partition; the eval half
  scored the 61.76%/62.1% baselines and is now the test partition.
- `check --split ID` runs every applicable leakage guard (below) and exits 1 on any violation. It
  also runs automatically inside `train.py`, `calibrate.py`, `evaluate.py gold` and `promote.py`.

`config.DEFAULT_SPLIT_ID = "three-way-v1"` — every script that takes `--split` defaults to it.

## Leakage guards (`guards.py`)

Pure checks over case-id sets/dataframes plus a `Split`. Each raises `GuardViolation` naming the
count and a few example case_ids (IDs only, never text). `check_all` loads whichever stores exist
and runs every applicable guard:

- Partition disjointness and coverage.
- `gold_eval() ⊆ test` (gold-eval never trains).
- An eval-side (calibration/test) review-queue gold row never enters `gold_train()`, hence never the
  corrected annotations.
- The corrected annotations' `matched_term`/`matched_group`/`matched_code` come only from labels
  train partition (`check_labels_train_only`) — trainers must run this **on the frame already
  filtered to train**, never the whole-universe table (which fails by design).
- Calibration reads calibration-partition cases only (`check_calibration_inputs`). Separately,
  `report_mapping.training.calibrate` refuses to calibrate a generation on cases it trained on.
- The audit store never feeds corrected annotations (`check_corrected_sources` — a structural
  guarantee, since `coding.corrected` never opens the audit store at all).

`GOLD_ORIGINS = {eval_batch, review_queue, random_slice}`; `GOLD_EVAL_ORIGINS = {eval_batch,
random_slice}`.

## Report-mapping generation lifecycle

A report-mapping generation (layout in [report-mapping.md](report-mapping.md)) has a `status`:
`candidate` (just trained/calibrating) → `current` (production) → `archived`. `scripts/train.py`
writes into `candidate/`; `scripts/calibrate.py` fills in its thresholds and flips
`calibration.status` to `calibrated`; `scripts/promote.py` decides whether it replaces `current/`.

### Promotion rule (`promote.py`)

Primary metric: weighted per-code **exact accuracy** (the `good` share); G+S is secondary. Both the
challenger (`candidate/`) and the incumbent (`current/`) are scored on the **current** gold-eval —
the same cases, the incumbent re-scored every time (gold-eval grows between runs) — each with its
own generation's uncommon groups. **Promote only if:**

(a) at least one retraining trigger is met (below), **and**
(b) the lower 95% bound of the paired (challenger − incumbent) `good`-share difference, from a
    stratified case-cluster paired bootstrap, is ≥ `MARGIN` (−2.0 pp).

```
ml/.venv/Scripts/python.exe ml/scripts/promote.py --candidate-predictions PATH --incumbent-predictions PATH
ml/.venv/Scripts/python.exe ml/scripts/promote.py --candidate-predictions PATH --incumbent-predictions PATH --apply
```

Without `--apply`, `promote.py` only recommends. With `--apply`: a winning candidate is swapped in
after the incumbent is archived (both same-filesystem `os.rename` — `current/` →
`ARCHIVE_ROOT/YYYY-MM-DD_<desc>/`, then `candidate/` → `current/`; if the second fails, the first is
undone, so `current/` is never left half-moved or empty); a losing candidate is deleted. The
candidate is re-verified (manifest, embedding fingerprint, `calibration.status == "calibrated"`)
immediately before anything moves. `ARCHIVE_ROOT` (`output/archive/`) is written only here; nothing
loads models or data from it.

### Retraining triggers (`triggers.py`)

Each `Trigger` reports met/not-met plus the numbers behind it. Thresholds are the strategy's
placeholders, to settle on real data:

1. **`new_silver_lineage`** — the challenger's `parents.silver_id` differs from the incumbent's. Not
   met when either side is unknown.
2. **`random_slice_drop`** — on the latest upload period's random slice, the incumbent's weighted
   `good` share is below its share on the rest of gold-eval, and the two 95% stratified
   case-cluster bootstrap intervals don't overlap. The latest period is excluded from the reference
   sample so the two are disjoint.
3. **`gold_train_growth`** — gold-train codes now minus the count the incumbent trained on
   (`parents.gold_train_codes`; 0 for gen-0) is at least `GOLD_TRAIN_GROWTH` (200). A re-reviewed
   case whose code set is merely replaced doesn't count as new.

```
ml/.venv/Scripts/python.exe ml/scripts/generations.py status --silver silver-0-legacy
```

`generations.py status` prints `current/` and `candidate/`'s generation_id, calibration status,
parents, and whether a challenger trained on `--silver` would meet a trigger. `current/`'s gen-0 and
the embedding cache's first (re-keyed) entry were imported once, before cutover, from the legacy
checkpoints and the legacy embedding cache; the one-time importers are gone.

## `retrain_cycle.py` — the local lane in one go

`icd-mapping-strategy.md`'s "the local lane is the retraining contract" — recommend-only, never
applies a promotion itself:

```
ml/.venv/Scripts/python.exe ml/scripts/retrain_cycle.py --silver silver-1 --device cuda --local-only
ml/.venv/Scripts/python.exe ml/scripts/retrain_cycle.py --silver silver-1 --device cuda \
    --gold-csv PATH --export-id 2026-10-export --reviewer "Dr. Smith" --force
```

Each step is the existing entry point run as its own process (a fresh process frees GPU memory
between steps): import gold if given → stop unless gold-eval exists → `predict.py` for `current/` if
missing, then stop unless a trigger fires (`--force` overrides) → `code_cases.py corrected` →
`train.py --stage heads` (or `backbone` too, with `--backbone`, for a cold start) →
`calibrate.py` → `predict.py` for `candidate/` → `promote.py` (no `--apply`). Refuses to start if
`candidate/` already exists (promote or delete it first). Never runs the LLM cascade — a new silver
generation is its own step (`map_diagnoses.py`).

## Archive layout

`config.ARCHIVE_ROOT` (`output/archive/`), sibling of `output/report_mapping/`, never referenced by
any loader. A promoted-out incumbent moves to `ARCHIVE_ROOT/YYYY-MM-DD_<desc>/` whole (backbone +
all checkpoints + manifest). Promotion also moves any embedding-cache entries the new `current/`
can't use into the archive (their key includes the backbone's fingerprint, so a stale entry is
otherwise just dead weight beside the shared cache) — listed in the archived generation's manifest,
so an archived generation stays independently verifiable after its cache entry moves.
