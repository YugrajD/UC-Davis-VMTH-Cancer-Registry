# Set up a new machine

The checklist for getting the ML pipeline running on a fresh checkout: environment, GPU check, S3
credentials, the first pull and a smoke test. For anyone starting on a new computer.

Git carries the code and these docs. It carries **none** of the data, weights or generated
artifacts: `.gitignore` blocks `ml/data/`, `ml/output/`, `.venv/` and every `*.csv`, `*.npz`, `*.pt`
and `*.safetensors` in the tree, because they contain patient records or are hundreds of MB. A fresh
`git clone` therefore imports cleanly but cannot run until the pull in step 4 has fetched the data.

## What comes from where

| Thing | Path | How to get it |
|---|---|---|
| Code, `config.py`, docs | `ml/**.py`, `ml/documentation/` | `git clone` / `git pull` |
| Taxonomy | `ml/taxonomy/labels.csv` | in git |
| Raw patient input | `ml/data/` | `sync.py pull data --apply` |
| Silver generations | `ml/output/silver/` | S3 set `silver` (expensive to regenerate: an LLM cascade run) |
| Manual-audit stores | `ml/output/manual_audit/` | S3 set `manual_audit` (specialist reviews, not reproducible) |
| Splits | `ml/output/splits/` | S3 set `splits`. **Do not regenerate**: a new split reshuffles which cases are held out and silently invalidates every published number |
| Coding tables | `ml/output/coding/` | S3 set `coding` |
| Production generation | `ml/output/report_mapping/current/` | `sync.py pull-model --apply`; regenerating means a full retrain. `candidate/` is unpromoted work and is not synced |
| Predictions, eval history | `ml/output/predictions/`, `ml/output/eval/` | S3 sets `predictions`, `eval`; only if you need a past run |
| Virtualenv | `ml/.venv/` | rebuild locally (step 2); never copy a venv between machines |
| Root `CLAUDE.md` | repo root | gitignored and not synced; copy it by hand |

Minimum for a working pipeline: `ml/data/` plus `ml/output/{splits,silver,manual_audit,coding}` and
`report_mapping/current/`. Everything else under `ml/output/` regenerates. The set names and how sync
works are in [sync-with-s3.md](sync-with-s3.md); the full path list is in
[paths.md](../reference/paths.md).

## 1. Clone

```bash
git clone <repo-url>
cd UC-Davis-VMTH-Cancer-Registry
```

## 2. Environment

Python 3.12. The interpreter is `ml/.venv/Scripts/python.exe` on Windows and `ml/.venv/bin/python`
on macOS and Linux; every command in these docs is written for Windows, so substitute accordingly.

```bash
python -m venv ml/.venv
ml/.venv/Scripts/python.exe -m pip install -r ml/requirements.txt
# Then reinstall torch from the CUDA 12.8 index, as ml/requirements.txt notes. PyPI's default wheel
# has no Blackwell (sm_120) kernels and fails on an RTX 50-series card. The version is already
# satisfied by the PyPI wheel, so force the swap (torch only; its dependencies are installed):
ml/.venv/Scripts/python.exe -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps
```

The pins that matter are torch 2.9.1, transformers 5.18.0, tokenizers 0.23.2, huggingface_hub 1.33.0
and safetensors 0.8.0. A generation is trained and served on these versions, so
`ml-worker/requirements.txt` carries the same five (a test checks that they agree). A machine with no
GPU installs the CPU build of the same torch version instead
(`--index-url https://download.pytorch.org/whl/cpu`); that is enough for inference and the test
suite, and it is what the cloud worker image uses.

## 3. Check the GPU

```bash
ml/.venv/Scripts/python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA device')"
```

If `cuda.is_available()` is `False`, pass `--device cpu`. `train.py`, `predict.py`, `calibrate.py`,
`oof.py`, `retrain_cycle.py` and `generations.py recalibrate` accept `--device` (`auto`, `cpu`,
`cuda`); `auto` picks cuda when available, else cpu. Training on cpu is correct but far slower.

## 4. S3 credentials and the first pull

Install **AWS CLI v2** and run `aws configure` (region `us-west-2`) with the access key the bucket
owner issued you. The tool uses boto3's default credential chain, so `AWS_PROFILE` also works, and no
keys are stored in the repo. Then, after the environment exists (boto3 is in `requirements.txt`):

```bash
ml/.venv/Scripts/python.exe ml/scripts/sync.py pull            # dry run: every file set
ml/.venv/Scripts/python.exe ml/scripts/sync.py pull --apply
ml/.venv/Scripts/python.exe ml/scripts/sync.py pull-model --apply
```

If the machine already holds some of these files (a USB copy, an older checkout), read "Joining with
existing files" in [sync-with-s3.md](sync-with-s3.md) first: a first pull replaces differing files
with the S3 version and keeps local-only files, which the next push would upload. This is
identifiable veterinary patient data: it travels only through the sync (encrypted at rest) or
another approved encrypted channel.

## 5. Smoke test

```bash
# The test suite: synthetic fixtures only, needs no data or checkpoints
ml/.venv/Scripts/python.exe -m pytest ml -q -p no:cacheprovider

# Paths resolve and the current generation is where config.py expects it
ml/.venv/Scripts/python.exe -c "import sys; sys.path.insert(0,'ml'); import config, os; \
print(all(os.path.exists(p) for p in [config.REPORT_CSV, config.REPORT_MAPPING_CURRENT_DIR, config.SPLITS_DIR]))"

# current/, candidate/ and the archive, with each one's architecture and calibration status
ml/.venv/Scripts/python.exe ml/scripts/generations.py status

# End-to-end inference on the current generation
ml/.venv/Scripts/python.exe ml/scripts/predict.py --generation current --device cuda
```

The loader refuses a generation whose manifest hashes, architecture or calibration do not check out
(the spec check, [generations.md](../concepts/generations.md#the-spec-check)), so a broken transfer
shows up as an error, not as silently wrong numbers. The first training run downloads the base
checkpoint `thomas-sounack/BioClinical-ModernBERT-base` from Hugging Face (or reads it from the local
cache); inference reads only the generation's own `encoder/` directory.

## Optional: diagnosis mapping

Tier 3 of the diagnosis cascade calls a local [LM Studio](https://lmstudio.ai) server. Create
`ml/diagnosis_mapping/.env` (gitignored) with `LLM_HOST`, `API_PORT` and `LLM_MODEL` for this
machine's LM Studio, and load a model. See [diagnosis-mapping.md](../concepts/diagnosis-mapping.md).

## Review-only work

To continue the manual audit or eval-batch review without training, you need only `ml/data/`,
`ml/output/silver/` and `ml/output/manual_audit/`: no GPU and no checkpoints
([run-audit-cycle.md](run-audit-cycle.md)).

## Where to read next

[../README.md](../README.md) maps the documentation and says where the project stands;
[../history/README.md](../history/README.md) indexes the plans, decisions and experiments behind it.

_Last verified against code: 2026-10-09_
