# ML Directory — Overview

A machine learning system that maps free-text veterinary pathology reports to standardized Vet-ICD-O-canine-1 cancer labels (term, group, ICD code). Production inference runs `ml/scripts/run_production.py`, which loads a per-section contrastive PetBERT backbone, embeds each report as a 2304-dim concat-3 vector, and runs a 4-stage classifier pipeline (case-presence gate → group classifier → per-group label-presence head → keyword correction).

Current production baseline: **G+S 62.1% on the held-out eval-half** (Good 46.1, Slight 16.0, CO 14.7, FP 2.3, FN 20.8).

---

## Directory map

```
ml/
├── data/                Raw input CSVs (report.csv, diagnoses.csv) — gitignored
├── ICD_labels/          Vet-ICD-O taxonomy, behavior/subtype keyword filters, labels.csv
├── model/               Neural-network module definitions
├── annotation/          Diagnosis → ICD label annotation (LLM cascade + cleanup)
├── training/            Per-stage training scripts (binary, group, label_presence, contrastive)
├── production/          4-stage inference pipeline (production/petbert_pipeline/)
├── evaluation/          Verdict scoring (end-to-end + per-stage + per-LP)
├── analysis/            Annotation coverage statistics
├── scripts/             Top-level entry points — only place a user runs Python
├── utils/               Shared helpers (encoding, csv I/O)
├── output/              All generated artifacts (gitignored — see "Where outputs go")
├── documentation/       This directory
├── config.py            Project-wide path constants — every script reads from here
├── requirements.txt     Pinned Python dependencies
└── .venv/               Python virtual environment
```

| Subdir | Purpose |
|---|---|
| `ICD_labels/` | Loads `labels.csv` (845 Vet-ICD-O terms, 52 groups). Hosts `behavior_keywords.py` and `subtype_keywords.py` consumed by Stage 4 (keyword correction). |
| `model/` | `CasePresenceClassifier`, `GroupClassifier`, `LabelPresenceClassifier` module defs plus `constants.py` (PETBERT_EMB_DIM=768, DEFAULT_HIDDEN_DIM=512). |
| `annotation/llm_pipeline/` | Three-tier diagnosis cascade (exact → fuzzy → LLM) + ensemble verification cleanup. Produces `ml/output/annotation/llm_annotation.csv`; the cleaned version is promoted to `annotation.csv` for training. |
| `training/binary/` | CasePresenceClassifier (Stage 1) dataset build + training. |
| `training/group/` | GroupClassifier (Stage 2) dataset build + training. |
| `training/label_presence/` | Per-group LabelPresenceClassifier (Stage 3a) — one model per ICD group. |
| `training/contrastive/` | PetBERT backbone adaptation via InfoNCE on per-section `(report_text, label_text)` pairs. |
| `training/data/` | One-shot train/test split generator (`create_split.py`). |
| `production/petbert_pipeline/` | 4-stage inference pipeline. `pipeline.py` orchestrates; `stages/` holds one module per stage. |
| `evaluation/` | End-to-end (`evaluate.py`), case-based, common-labels, plus per-stage scorers for Stage 1/2/3. |
| `analysis/annotation_stats.py` | Annotation coverage stats (cases-per-group, collisions, etc.). |
| `scripts/` | The only Python entry points: `run_annotation.py`, `run_training.py`, `run_production.py`, `run_evaluation.py`, `run_data_analysis.py`, `sweep_lp_thresholds.py`, `sweep_tail_gate.py`. |
| `utils/` | `encoding.py` (safe filename + NPZ key helpers), `csv_io.py` (BOM stripping). |

---

## Quick start

Use `ml/.venv/Scripts/python.exe` on Windows, `ml/.venv/bin/python3` on macOS/Linux. Every `scripts/*.py` adds `ml/` to `sys.path` automatically.

**Annotate diagnoses** (produces training labels — needs LM Studio running locally):
```bash
ml/.venv/Scripts/python.exe ml/scripts/run_annotation.py
```

**Train (cold start)** — adapt backbone, then train the three classifier heads in order:
```bash
ml/.venv/Scripts/python.exe ml/scripts/run_training.py --mode adapt-backbone --device cuda --local-only
rm -f ml/output/training/embedding_cache.npz
ml/.venv/Scripts/python.exe ml/scripts/run_production.py --embed-only --device cuda
ml/.venv/Scripts/python.exe ml/scripts/run_training.py --mode train-case-presence  --device cuda --local-only --train-cases ml/output/splits/train_cases.txt
ml/.venv/Scripts/python.exe ml/scripts/run_training.py --mode train-groups         --device cuda --local-only --train-cases ml/output/splits/train_cases.txt --epochs 300 --lr 5e-5 --dropout 0.1
ml/.venv/Scripts/python.exe ml/scripts/run_training.py --mode train-label-presence --device cuda --local-only --train-cases ml/output/splits/train_cases.txt
```
See [training-guide.md](training-guide.md) for the full retraining cycle including threshold calibration.

**Run inference** — scores every row in `ml/data/report.csv`:
```bash
ml/.venv/Scripts/python.exe ml/scripts/run_production.py --device cuda --local-only
```
All four checkpoint paths and the gate/group thresholds (0.80 / 0.85) default to the production values via `run_production.py`'s `parser.set_defaults(...)`; you only override flags you want to change. See [production-pipeline.md](production-pipeline.md) for the full flag list.

---

## Pipeline at a glance

```
report.csv (latin-1)
  │
  ▼  Build three section views per case
     __sec_0__ = HISTOPATHOLOGICAL SUMMARY
     __sec_1__ = FINAL COMMENT + COMMENT
     __sec_2__ = ANCILLARY TESTS
  │
  ▼  Embed each section through PetBERT (concat-3 backbone) → (N, 3×768)
     Concatenate → (N, 2304)   [cache: ml/output/training/embedding_cache.npz]
  │
  ▼  Stage 1: CasePresenceClassifier   2304 → cancer probability   (threshold 0.80)
  ▼  Stage 2: GroupClassifier          2304 → 25 group probs       (threshold 0.85 + tail-gate K=2, gap=0.08)
  ▼  Stage 3a: per-group LabelPresenceClassifier   2304+label → in-group label score
              (per-LP threshold from lp_thresholds.json; argmax fallback)
  ▼  Stage 3b/4: keyword correction    behavior digit + subtype filter
  ▼  Post-Stage-3 rescue: Lipoma keyword RESCUE (appends Lipoma, NOS when applicable)
  │
  ▼  predictions.csv + provenance + similarity + visualization + embeddings + summary
```

Gate-rejected cases become `Uncategorized`. Cases that pass the gate but fail every group threshold become `Unidentified Cancer` (unless argmax fallback is on — default).

---

## Where outputs go

All under `ml/output/` (gitignored).

| Path | Contents |
|---|---|
| `output/annotation/annotation.csv` | Canonical training supervision (`config.ANNOTATION_CSV`); promoted from the cleaned annotation below |
| `output/annotation/llm_annotation.csv` | Raw three-tier annotation result |
| `output/annotation/llm_annotation_cleaned.csv` | After ensemble cleanup pass; promote to `annotation.csv` to use for training |
| `output/annotation/llm_summary.{json,md}` | Annotation statistics |
| `output/splits/train_cases.txt` / `test_cases.txt` | 80/20 stratified case-ID lists (generated once) |
| `output/training/embedding_cache.npz` | Per-section + concat-3 + mean + label embeddings. Auto-invalidates on model change. |
| `output/training/contrastive/contrastive_pairs.csv` | Per-section (report, label) pairs for backbone adaptation |
| `output/training/group/group_training_data.npz` | Multi-hot targets for GroupClassifier |
| `output/training/group/uncommon_groups.txt` | Group names merged into the "Uncommon" bucket |
| `output/training/binary/case_presence_dataset.npz` | Cancer/no-cancer dataset for CasePresenceClassifier |
| `output/training/label_presence/{safe_group}_pairs.csv` | Within-group (case, label, target) pairs |
| `output/checkpoints/contrastive/` | Adapted PetBERT backbone (HuggingFace format) |
| `output/checkpoints/case_presence/case_presence_classifier.pt` | Stage 1 gate |
| `output/checkpoints/group/group_classifier_best.pt` | Stage 2 |
| `output/checkpoints/label_presence/{safe_group}.pt` | Stage 3a — one per ICD group + `uncommon.pt` |
| `output/checkpoints/label_presence/lp_thresholds.json` | Per-LP thresholds from `sweep_lp_thresholds.py` |
| `output/production/petbert_predictions.csv` | Top-k predictions per case (current run) |
| `output/production/petbert_{provenance,similarity_scores,visualization,embeddings,summary}.*` | Debug artifacts |
| `output/evaluation/{subdir}/evaluation*.csv` + `*_history.csv` | End-to-end verdicts + cycle history |
| `output/evaluation/{subdir}/{case_presence,groups,label_presence}_evaluation*.csv` | Per-stage scorer outputs |
| `output/data_analysis/` | Annotation coverage tables + plots |

---

## Where to read next

| File | When to read it |
|---|---|
| [production-pipeline.md](production-pipeline.md) | You're invoking `run_production.py` or debugging an inference run |
| [classifiers.md](classifiers.md) | You want the architecture, input/output shapes, or threshold mechanics of any classifier |
| [label-annotation.md](label-annotation.md) | You're running or debugging `run_annotation.py` |
| [model-training.md](model-training.md) | You want the reasoning behind the 4-stage design and concat-3 representation |
| [training-guide.md](training-guide.md) | You're retraining and need exact commands + expected runtimes |
| [archive/training-log/](archive/training-log/) | Historical phase logs — do not consult for current behavior |
