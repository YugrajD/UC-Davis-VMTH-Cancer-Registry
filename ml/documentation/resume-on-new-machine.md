# Resuming on Another Machine

Git carries the code and these docs. It carries **none** of the data, weights, or generated
artifacts — `.gitignore` blocks `ml/data/`, `ml/output/`, `.venv/`, and every `*.csv` / `*.npz` /
`*.pt` / `*.safetensors` in the tree, because they contain patient records or are hundreds of MB. A
fresh `git clone` therefore gives you a pipeline that imports cleanly and cannot run.

This page is the checklist for closing that gap.

---

## 1. What crosses by git, what you must carry yourself

| Thing | Path | In git? | How to get it on the new machine |
|---|---|---|---|
| All Python code, `config.py`, docs | `ml/**.py`, `ml/documentation/` | yes | `git clone` / `git pull` |
| Taxonomy | `ml/taxonomy/labels.csv` | yes (in-tree asset) | comes with the clone |
| Raw patient input | `ml/data/` | **no** | copy from the old machine or re-export from the client's Box |
| Silver generations | `ml/output/silver/` | **no** | copy — expensive to regenerate (an LLM cascade run) |
| Manual-audit stores | `ml/output/manual_audit/` | **no** | copy — these are specialist review results, not reproducible |
| Splits | `ml/output/splits/` | **no** | **copy — do not regenerate.** A new split reshuffles which cases are held out and silently invalidates every published number |
| Report-mapping generations | `ml/output/report_mapping/current/`, `candidate/` | **no** | copy — regenerating means a full retrain |
| Embedding cache | `ml/output/report_mapping/embedding_cache/` | **no** | skip; it auto-rebuilds (content-hash keyed, ~9 min on GPU) and auto-invalidates when the backbone or report/labels files change |
| Predictions, eval history | `ml/output/predictions/`, `ml/output/eval/` | **no** | skip unless you specifically need a past run |
| Virtualenv | `ml/.venv/` | **no** | rebuild locally (§2) — never copy a venv between machines |

**Minimum transfer to have a working pipeline:** `ml/data/` + `ml/output/{splits,silver,
report_mapping/current,manual_audit,coding}`. Everything else under `ml/output/` regenerates.

Transfer over an encrypted channel — this is identifiable veterinary patient data. UC Davis SSO Box
is the approved location; see [box-rclone-sync-proposal.md](box-rclone-sync-proposal.md) for the
standing (not yet implemented) proposal to automate exactly this.

---

## 2. Environment

Python 3.12.

```bash
python -m venv ml/.venv
ml/.venv/Scripts/python.exe -m pip install -r ml/requirements.txt
# Then reinstall torch from the CUDA 12.8 index — PyPI's default wheel has no
# Blackwell (sm_120) kernels and will fail on an RTX 50-series card:
ml/.venv/Scripts/python.exe -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
```

Interpreter path differs by OS — `ml/.venv/Scripts/python.exe` on Windows, `ml/.venv/bin/python` on
macOS/Linux. Every command in these docs is written for Windows; substitute accordingly.

Verify the GPU is actually visible before starting a long run:
```bash
ml/.venv/Scripts/python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
If `cuda.is_available()` is `False`, pass `--device cpu` — every entry point accepts it, and
training will be far slower but correct.

**Diagnosis mapping only:** Tier 3 calls a local [LM Studio](https://lmstudio.ai) server. Copy
`ml/diagnosis_mapping/.env` (gitignored) or recreate it with the host/port of the new machine's LM
Studio, and load a model — see [diagnosis-mapping.md](diagnosis-mapping.md).

---

## 3. Smoke test the transfer

```bash
# Run the test suite (synthetic fixtures only — needs no data or checkpoints)
ml/.venv/Scripts/python.exe -m pytest ml -q -p no:cacheprovider

# Paths resolve and the current generation is where config.py expects it
ml/.venv/Scripts/python.exe -c "import sys; sys.path.insert(0,'ml'); import config, os; \
print(all(os.path.exists(p) for p in [config.REPORT_CSV, config.REPORT_MAPPING_CURRENT_DIR, config.SPLITS_DIR]))"

# End-to-end inference on the current generation
ml/.venv/Scripts/python.exe ml/scripts/predict.py --generation current --device cuda --local-only
```

A run that finishes but scores far below the ~61.76% eval-half reference almost always means a
**stale or mismatched generation**, not a code bug — but `load_generation` now refuses to load a
generation whose embedding fingerprint or manifest doesn't check out (see
[report-mapping.md](report-mapping.md)), so this should surface as an error rather than silently
wrong numbers.

---

## 4. Where the work stands

Branch: `Revised-ICD-Mapping`. The ML rewrite ([ml-rewrite-plan.md](ml-rewrite-plan.md)) is the
authoritative status page — read it before touching anything below; it records exactly which work
packages are done and what's next. The `ml/next` → `ml` cutover (WP13) is complete; the tree
described in this doc is the current one.

If you are resuming purely to continue the manual audit or eval-batch review (no training), you need
only `ml/data/`, `ml/output/silver/` and `ml/output/manual_audit/` — no GPU, no checkpoints. See
[manual-audit.md](manual-audit.md).
