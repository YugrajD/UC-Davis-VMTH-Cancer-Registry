# Production Pipeline

What `ml/scripts/run_production.py` does today. Authoritative reference for runtime behavior.

## What it does

Reads a CSV of veterinary pathology reports, embeds each report into a 2304-dim vector through PetBERT, and runs a 4-stage classifier pipeline that produces ranked Vet-ICD-O cancer label predictions per case.

```bash
ml/.venv/Scripts/python.exe ml/scripts/run_production.py \
  --case-presence-threshold 0.85 \
  --group-classifier-threshold 0.85 \
  --device cuda --local-only
```

`run_production.py` is a thin entry point — it pre-fills production-default paths (model, embedding cache, the three classifier checkpoints, the LP thresholds JSON, and the output dir) plus `--local-only`, then calls `production.petbert_pipeline.run_scan` (the CLI is built by `build_parser()` / `build_config()` in `production/petbert_pipeline/cli.py`). See the [CLI flags](#cli-flags) section below for the full list.

## Inputs

| Path | Required columns |
|---|---|
| `ml/data/report.csv` (default; override with `--csv`) | `case_id`, `HISTOPATHOLOGICAL SUMMARY`, `FINAL COMMENT`, `COMMENT`, `ANCILLARY TESTS` |
| `ml/ICD_labels/labels.csv` | Vet-ICD-O taxonomy (term, group, code) |
| `ml/output/checkpoints/contrastive/` | Adapted PetBERT backbone |
| `ml/output/checkpoints/case_presence/case_presence_classifier.pt` | Stage 1 gate |
| `ml/output/checkpoints/group/group_classifier_best.pt` | Stage 2 |
| `ml/output/checkpoints/label_presence/*.pt` | Stage 3a (one .pt per group + optional `uncommon.pt`) |
| `ml/output/checkpoints/label_presence/lp_thresholds.json` | Per-LP thresholds (optional but recommended) |
| `ml/output/training/group/uncommon_groups.txt` | Group names merged into the "Uncommon" bucket |
| `ml/output/training/embedding_cache.npz` | PetBERT embeddings cache (built on first run) |

Report CSV is read with `latin-1` encoding; BOM artifacts are stripped from column names. Missing cells are normalized to empty strings.

## Internal steps

`production/petbert_pipeline/pipeline.py::run_scan` runs these in order:

1. **Load report CSV.** Strip BOM, validate `case_id` column, optional `--max-rows` truncation.
2. **Build concat-3 section views.** Three synthetic per-row columns:
   - `__sec_0__` = `HISTOPATHOLOGICAL SUMMARY`
   - `__sec_1__` = `FINAL COMMENT` + newline + `COMMENT`
   - `__sec_2__` = `ANCILLARY TESTS`

   Defined in `pipeline.py::CONCAT_3_SECTIONS`. Empty cells become empty strings (tracked via `has_content` masks).
3. **Embed each section, then concat.** Either load `--embedding-cache` (validates against current model name + report CSV mtime + labels CSV mtime — see `embedding_cache.py`), or run PetBERT fresh: each `__sec_N__` is tokenized with `--max-length 512`, mean-pooled to 768-dim, and the three vectors concatenated per row into a 2304-dim view stored under cache key `concat_3`. A 768-dim masked-mean across non-empty sections is also computed for cosine-similarity fallbacks against label embeddings.
4. **Run the 4-stage pipeline** (`stages/__init__.py::categorize_per_case`). Detailed below.
5. **Resolve top-k indices to (term, group, code).** `ICD_labels.resolve_taxonomy_matches`.
6. **Write outputs.** PCA-2D visualization, predictions CSV, provenance, similarity matrix, embeddings NPZ, summary JSON, optional neighbors CSV.

If `--embed-only` is set, the pipeline stops after step 3 — useful for building the cache before training without running classifiers.

## The four stages

Each stage lives in its own module under `ml/production/petbert_pipeline/stages/`. The dispatcher in `stages/__init__.py` loops per case.

### Stage 1 — CasePresenceClassifier (gate)

File: `stages/case_presence_classifier.py`. Input: 2304-dim concat-3 vector. Output: per-case cancer probability. Cases below `--case-presence-threshold` (default 0.5; recommended 0.85) skip Stages 2–4 and are emitted as `Uncategorized`. Trained with `recall_weight=0.7` so it errs toward letting uncertain cases through.

### Stage 2 — GroupClassifier

File: `stages/group_classifier.py`. Input: 2304-dim concat-3 vector. Output: sigmoid probability per ICD group (25 groups in production). Groups above `--group-classifier-threshold` (default 0.3; recommended 0.85) advance. When no group clears the threshold, argmax fallback selects the top-scoring group (disable with `--no-group-classifier-fallback-to-argmax`).

After thresholding, a **tail gate** trims wide tails:
- At most `--tail-max-predictions` group predictions per case (default 2).
- Tail predictions whose probability is more than `--tail-max-group-prob-gap` (default 0.08) below the top group's probability are dropped.

Set `gap=1.0` to disable the gate. Recalibrate after a GroupClassifier retrain with `ml/scripts/sweep_tail_gate.py`.

### Stage 3a — per-group LabelPresenceClassifier

File: `stages/label_presence_classifier.py`. For each surviving group, looks up a `.pt` checkpoint by safe filename in `--label-presence-classifier-dir`. The head is built with `n_cols=3, col_pair_mode=True, col_combine="learned"`: the 2304-dim concat is split into three 768-dim section views, each section forms a `[section_emb | label_emb]` pair (1536-dim), runs through a shared 1536→512→1 MLP, and per-section logits are combined by a learned `Linear(3 → 1)`.

Per-LP threshold lookup order:
1. `--label-presence-thresholds-json` → group name → threshold (default `ml/output/checkpoints/label_presence/lp_thresholds.json`).
2. `--label-presence-threshold` (default 0.5) for any group missing from the JSON.

Labels above threshold are selected; argmax fallback applies when nothing passes. Groups without a corresponding `.pt` (e.g. `Uncommon` if no `uncommon.pt`) fall through directly to Stage 3b. Pass `--label-presence-classifier-dir ""` to disable Stage 3a entirely.

### Stage 3b / Stage 4 — keyword correction

File: `stages/keyword_correction.py`. Applied to whichever pool Stage 3a produced (or the full in-group label pool when Stage 3a is absent).

1. **Behavior-code filter** — `ICD_labels/behavior_keywords.py` scores the report text for ICD-O behavior digits (`/0` benign, `/1` borderline, `/2` in situ, `/3` malignant, `/6` metastatic). The highest-ranked digit narrows the pool to labels with matching codes. When no signal is found, the pool passes through.
2. **Subtype keyword filter** — `ICD_labels/subtype_keywords.py` applies group-specific discriminators for 6 groups: Mast cell neoplasms, Blood vessel tumors, Melanocytoma and Melanomas, Meningiomas, Osseous and chondromatous neoplasms, Gliomas. Each group has an ordered list of `(regex, label_substr)` rules; first matching rule narrows the pool.

When an LP head is present, cosine-similarity is not used inside the pool — the LP score is the final rank. When Stage 3a is absent, cosine similarity (768-dim masked-mean vs 768-dim label embeddings) breaks ties within the post-filter pool.

## Output files

Written under `--out-dir` (default `ml/output/production/`).

| File | Contents |
|---|---|
| `petbert_predictions.csv` | One row per (case, rank). Columns: `case_id`, `diagnosis_index`, `predicted_term`, `predicted_group`, `predicted_code`, `confidence`, `method`. |
| `petbert_provenance.csv` | Per-case traceability — merged input text, token counts, final label, embedding top-1 fallback. |
| `petbert_similarity_scores.csv` | Full N×M label score matrix. |
| `petbert_visualization.csv` | PCA-2D coordinates per case. |
| `petbert_embeddings.npz` | 768-dim masked-mean embeddings + case IDs + text. |
| `petbert_summary.json` | Run metadata + aggregate prediction counts. |
| `petbert_neighbors.csv` | Top-k nearest cases (only when `--task neighbors` or `--task both`). |

The `method` column in predictions takes values `embedding` (LP head or cosine), `label_presence`, `low_confidence` (gate-rejected → `Uncategorized`), `unidentified_cancer` (gate passed but no group), or `empty` (empty text).

## CLI flags

Source of truth: `production/petbert_pipeline/cli.py::build_parser` and the `ScanConfig` defaults in `types.py`. `run_production.py` overrides several of these defaults via `parser.set_defaults(...)`: `--model`, `--embedding-cache`, `--group-classifier`, `--case-presence-classifier`, `--label-presence-classifier-dir`, `--label-presence-thresholds-json`, `--out-dir`, and `--local-only`. The table below shows the **effective defaults when invoked via `run_production.py`**.

| Flag | Default | Description |
|---|---|---|
| `--csv` | `ml/data/report.csv` | Input report CSV |
| `--id-col` | `case_id` | Case ID column name |
| `--model` | `ml/output/checkpoints/contrastive/` (via `run_production.py`) | PetBERT model dir or HF name |
| `--local-only` | True (via `run_production.py`) | Disable HuggingFace download |
| `--out-dir` | `ml/output/production/` | Where to write outputs |
| `--max-rows` | None | Truncate input (debugging) |
| `--batch-size` | 16 | Embedding batch size |
| `--max-length` | 512 | Tokenizer max_length per section |
| `--neighbors-k` | 3 | k for `--task neighbors` |
| `--task` | `categorize` | `categorize` / `neighbors` / `both` |
| `--embedding-min-sim` | 0.6 | Min cosine for embedding fallback |
| `--device` | `auto` | `auto` / `cpu` / `cuda` / `mps` / `xpu` |
| `--labels-csv` | `ml/ICD_labels/labels.csv` | Taxonomy CSV |
| `--embedding-cache` | `ml/output/training/embedding_cache.npz` (via `run_production.py`) | Cache NPZ path |
| `--case-presence-classifier` | `ml/output/checkpoints/case_presence/case_presence_classifier.pt` | Stage 1 checkpoint. Pass `""` to skip the gate. |
| `--case-presence-threshold` | 0.5 | Gate threshold (recommended 0.85) |
| `--group-classifier` | `ml/output/checkpoints/group/group_classifier_best.pt` | Stage 2 checkpoint |
| `--group-classifier-threshold` | 0.3 | Group threshold (recommended 0.85) |
| `--no-group-classifier-fallback-to-argmax` | argmax on | Disable Stage 2 argmax fallback |
| `--label-presence-classifier-dir` | `ml/output/checkpoints/label_presence/` | Stage 3a checkpoint dir. Pass `""` to disable. |
| `--label-presence-threshold` | 0.5 | Per-LP fallback threshold |
| `--label-presence-thresholds-json` | `ml/output/checkpoints/label_presence/lp_thresholds.json` | Per-LP threshold map. Missing file → warn + use global threshold. |
| `--tail-max-predictions` | 2 | Cap on group predictions per case |
| `--tail-max-group-prob-gap` | 0.08 | Drop tail groups more than this below the top group |
| `--rerank-stage3` | False | Re-rank Stage 3 winners by `(lp_score − lp_threshold) × group_prob` (only meaningful with `--tail-max-predictions > 1`) |
| `--embed-only` | False | Stop after embedding step (build cache only) |

## Embedding cache behavior

`production/petbert_pipeline/embedding_cache.py` saves and validates against:
- the model name string in `--model`
- mtime of the report CSV
- mtime of the labels CSV
- the expected section column names

Any mismatch → cache miss → re-embed. The cache stores the 2304-dim `concat_3` view, the per-section 768-dim views, the 768-dim masked-mean, token counts, and the M×768 label embedding matrix. Per `CLAUDE.md`, archive the old generation (embeddings + backbone + classifiers) under `ml/output/archive/YYYY-MM-DD_<desc>/` before retraining when embeddings change — stale classifiers load silently against new embeddings and produce wrong results with no error.

## Source of truth

If this document and the code disagree, the code wins. Read in this order:
- `ml/scripts/run_production.py` (entry point, sets defaults)
- `ml/production/petbert_pipeline/pipeline.py::run_scan` (orchestration)
- `ml/production/petbert_pipeline/cli.py` (CLI surface)
- `ml/production/petbert_pipeline/types.py` (`ScanConfig` defaults)
- `ml/production/petbert_pipeline/stages/__init__.py::categorize_per_case` (per-case dispatcher)
- `ml/config.py` (path constants)
