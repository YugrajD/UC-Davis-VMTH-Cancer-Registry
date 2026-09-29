# 0007 Train and embed on an NVIDIA CUDA GPU instead of Intel XPU

**Status:** accepted · **Date:** 2026-05-18

## Context
Training first ran on Apple MPS (March 2026) and then on an Intel Arc GPU through PyTorch's XPU build (support added 2026-02-28; a Windows setup guide existed). The XPU stack needed one single `pip install` from the XPU wheel index (splitting it caused Intel runtime version conflicts) and pinned older libraries than the rest of the project now uses. Long runs were slow: re-embedding the corpus took about 25 minutes and the per-section backbone about 44 minutes for 3 epochs.

## Decision
Standardise on an NVIDIA RTX 5070 Ti (Blackwell, sm_120) with the CUDA 12.8 PyTorch build (torch 2.9.1 from the cu128 index; sm_120 needs PyTorch 2.6 or newer). Device selection is `--device auto`, which prefers CUDA.

## Consequences
- `--device auto` order is cuda, xpu, mps, cpu (`report_mapping/model/backbone.py`). XPU is legacy but still an accepted choice, and `requirements.txt` keeps a commented XPU install line. `calibrate.py` defaults to `cpu` explicitly.
- On CUDA the re-embed of all 58,313 cases took 9.2 minutes and a 3-epoch backbone on 56,109 section pairs took 27 minutes (rewrite parity runs L2b and L4).
- The old Intel guide's library pins (`transformers==4.46.3`, `scikit-learn==1.4.0`, `pandas==2.2.0`) conflict with `ml/requirements.txt`; the guide was removed. The separate worker image may still pin different versions.
- The two timing pairs are from different runs and machines, so they show scale, not a controlled benchmark.

## Evidence
No accuracy experiment; a hardware decision. Rewrite parity timings in [the rewrite decision](0005-ml-rewrite-and-cutover.md). Setup today: [../how-to/new-machine-setup.md](../../how-to/new-machine-setup.md).
