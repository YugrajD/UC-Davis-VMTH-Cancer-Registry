# Resuming on Another Machine

Git carries the code and these docs. It carries **none** of the data, weights, or
generated artifacts — `.gitignore` blocks `ml/data/`, `ml/output/`, `.venv/`, and every
`*.csv` / `*.npz` / `*.pt` / `*.safetensors` in the tree, because they contain patient
records or are hundreds of MB. A fresh `git clone` therefore gives you a pipeline that
imports cleanly and cannot run.

This page is the checklist for closing that gap.

---

## 1. What crosses by git, what you must carry yourself

| Thing | Path | In git? | How to get it on the new machine |
|---|---|---|---|
| All Python code, `config.py`, docs | `ml/**.py`, `ml/documentation/` | yes | `git clone` / `git pull` |
| Taxonomy | `ml/ICD_labels/labels.csv` | yes (explicit whitelist) | comes with the clone |
| Raw patient input | `ml/data/` (~194 MB) | **no** | copy from the old machine or re-export from the client's Box |
| Annotation corpus | `ml/output/annotation/` (~48 MB) | **no** | copy — expensive to regenerate (~30–60 min of LLM calls) |
| Train/test split | `ml/output/splits/` (~2 MB) | **no** | **copy — do not regenerate.** A new split reshuffles which cases are held out and silently invalidates every published number |
| Model weights | `ml/output/checkpoints/` (~561 MB) | **no** | copy — regenerating means a full cold-start retrain |
| Embedding cache | `ml/output/training/embedding_cache.npz` | **no** | skip; it auto-rebuilds (~25 min on GPU) and auto-invalidates when the backbone changes |
| Scratch / experiments | `ml/output/{experiments,checkpoints_recency_scratch,production,evaluation}` (~11.5 GB) | **no** | skip unless you specifically need a past run |
| Virtualenv | `ml/.venv/` (7.7 GB) | **no** | rebuild locally (§2) — never copy a venv between machines |

**Minimum transfer to have a working pipeline: `ml/data/` + `ml/output/{annotation,splits,checkpoints}` ≈ 805 MB.**
Everything else under `ml/output/` regenerates.

Transfer over an encrypted channel — this is identifiable veterinary patient data.
UC Davis SSO Box is the approved location; see
[box-rclone-sync-proposal.md](box-rclone-sync-proposal.md) for the standing (not yet
implemented) proposal to automate exactly this.

---

## 2. Environment

Python 3.12 (current machine: 3.12.10).

```bash
python -m venv ml/.venv
ml/.venv/Scripts/python.exe -m pip install -r ml/requirements.txt
# Then reinstall torch from the CUDA 12.8 index — PyPI's default wheel has no
# Blackwell (sm_120) kernels and will fail on an RTX 50-series card:
ml/.venv/Scripts/python.exe -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
```

Interpreter path differs by OS — `ml/.venv/Scripts/python.exe` on Windows,
`ml/.venv/bin/python3` on macOS/Linux. Every command in these docs is written for
Windows; substitute accordingly.

Verify the GPU is actually visible before starting a long run:
```bash
ml/.venv/Scripts/python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Expected on the current machine: `2.9.1+cu128 True NVIDIA GeForce RTX 5070 Ti`.
If `cuda.is_available()` is `False`, pass `--device cpu` — every entry point accepts it,
and training will be far slower but correct.

**Annotation only:** Tier 3 calls a local [LM Studio](https://lmstudio.ai) server.
Copy `ml/annotation/llm_pipeline/.env` (gitignored) or recreate it with the host/port
of the new machine's LM Studio, and load a model — see
[label-annotation.md](label-annotation.md).

---

## 3. Smoke test the transfer

```bash
# Paths resolve and the checkpoints are where config.py expects them
ml/.venv/Scripts/python.exe -c "import sys; sys.path.insert(0,'ml'); import config, os; \
print(all(os.path.exists(p) for p in [config.REPORTS_CSV, config.ANNOTATION_CSV, \
config.TRAIN_CASES_TXT, config.TEST_CASES_TXT, config.CHECKPOINT_GROUP_DIR]))"

# End-to-end inference on a handful of cases
ml/.venv/Scripts/python.exe ml/scripts/run_production.py --device cuda --local-only --max-rows 50
```

A run that finishes but scores far below the published baseline (G+S 62.1% on the
eval-half) almost always means a **stale or mismatched checkpoint generation**, not a
code bug. This project's signature failure mode is that a stale classifier loads
silently and produces wrong results with no error. Checkpoints, the backbone, and the
embedding cache are one generation and must travel together.

---

## 4. Where the work stands (2026-08-19)

Branch: `Classifier-Training`. Production pipeline is unchanged and is the 4-stage
concat-3 system described in [README.md](README.md).

Active workstream is the **annotation redesign, Phase 0** —
[annotation-redesign-plan.md](annotation-redesign-plan.md) is the authoritative status
page; read it before touching anything below.

Landed since the last commit:

1. **`decision_stage` on the annotation corpus.** `method` records only the winning
   tier, so a `No Match` row was ambiguous — the LLM might have declined, or might
   never have been asked. `decision_stage` disambiguates. Backfill for a corpus
   generated before the column existed is `run_annotation.py --backfill-stage`
   ([label-annotation.md](label-annotation.md)).
2. **Gold review surface moved from Excel to plain CSV.** `xlsx_io.py`, the
   `consistency` subcommand, and the `openpyxl` dependency are gone; `csv_io.py`
   replaces them, and term/group typo-catching moved into `ingest` as a hard
   validation error.
3. **Batch 1 re-cut as a row-level Tier-3 audit.** Sampling rows rather than cases
   took the batch from 675 rows (3% useful) to 200 rows (100% useful).

Next real-world step, and it is not a coding step: **a veterinary professional fills in
`ml/output/annotation/tier3_audit_review.csv`** (hand over the two sidecars with it).
Then:

```bash
ml/.venv/Scripts/python.exe ml/scripts/run_gold_annotation.py ingest --verified-by "<name>"
ml/.venv/Scripts/python.exe ml/scripts/run_gold_annotation.py check-split
```

and read the per-stratum rates, weighting by `sample_weight`. **Do not** run
`run_evaluation.py` against this store — it is row-level and `evaluate.py` scores per
case; the un-sampled rows of a partially-covered case would read as false positives.

If you are resuming purely to continue this audit, you need only `ml/data/` and
`ml/output/annotation/` + `ml/output/splits/` — no GPU, no checkpoints.
