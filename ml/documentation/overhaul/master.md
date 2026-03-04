# ml/ Overhaul Part 2 — Master Plan

## Goal
Clean up the ml/ directory structure for clarity, modularity, and extensibility.

## Design Constraints
- No subprocesses; all orchestration via direct function imports
- No `env PYTHONPATH=ml`; scripts set `sys.path` internally
- Each training module exposes a typed API function; `main()` is a thin argparse wrapper
- Drop the `_call()` / sys.argv-swapping anti-pattern from `run_cycle.py`

---

## Steps

| # | File | Summary |
|---|---|---|
| 1 | [step1-rename-packages.md](step1-rename-packages.md) | `petbert_scan/` → `petbert_pipeline/`, `keyword_scan/` → `keyword_pipeline/` |
| 2 | [step2-restructure-training.md](step2-restructure-training.md) | Move training files into `binary/`, `group/`, `finetune/` subdirectories |
| 3 | [step3-extract-apis.md](step3-extract-apis.md) | Extract typed API functions from 4 training scripts |
| 4 | [step4-update-imports.md](step4-update-imports.md) | Update cross-module imports; refactor `run_cycle.py` to drop `_call()` |
| 5 | [step5-create-scripts.md](step5-create-scripts.md) | Create `ml/scripts/run_pipeline.py` and `run_training.py` |
| 6 | [step6-dry-run.md](step6-dry-run.md) | Verify everything works end-to-end |

---

## Final Structure

```
ml/
├── petbert_pipeline/      ← renamed from petbert_scan/
├── keyword_pipeline/      ← renamed from keyword_scan/
├── model/                 (unchanged)
├── labels/                (unchanged)
├── training/
│   ├── binary/            ← PresenceClassifier training
│   │   ├── build_training_pairs.py
│   │   ├── train.py
│   │   ├── run_cycle.py
│   │   ├── evaluate.py
│   │   ├── update_co_bank.py
│   │   └── log_evaluation.py
│   ├── group/             ← GroupClassifier training
│   │   ├── build_training_data.py
│   │   └── train.py
│   └── finetune/          ← PetBERT fine-tuning (placeholder)
└── scripts/
    ├── run_pipeline.py    ← production inference entry point
    └── run_training.py    ← end-to-end training entry point
```
