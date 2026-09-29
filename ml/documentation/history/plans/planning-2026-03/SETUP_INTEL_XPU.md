# Intel Arc GPU Setup Guide (Windows)

How to run the `petbert_scan` pipeline on an Intel Arc GPU (A-series or B-series) on Windows.

## Prerequisites

- Windows 11
- Intel Arc GPU (A-series or B-series)
- **Up-to-date Intel Arc GPU driver** — update via the Intel Arc Control app or [Intel's download page](https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html)
- Python 3.12 installed — verify with `py -3.12 --version`

> **No Intel oneAPI Base Toolkit or Intel Deep Learning Essentials needed.**
> All required Intel runtime DLLs (`sycl8.dll`, `pti_view-0.dll`, MKL, etc.) are installed automatically as pip packages.

---

## Setup

### 1. Create the virtual environment

```powershell
cd C:\path\to\UC-Davis-VMTH-Cancer-Registry
py -3.12 -m venv ml\.venv
```

### 2. Install all dependencies in one command

This **must be a single `pip install` command**. Splitting it into multiple installs causes Intel runtime version conflicts that prevent torch from loading.

```powershell
.\ml\.venv\Scripts\pip install `
  "torch==2.9.1" `
  "transformers==4.46.3" `
  "scikit-learn==1.4.0" `
  "numpy>=1.26.4" `
  "pandas==2.2.0" `
  --index-url https://download.pytorch.org/whl/xpu
```

pip will automatically resolve and install all Intel runtime packages (`intel-sycl-rt`, `intel-pti`, `mkl`, etc.) at the exact versions torch was compiled against.

### 3. Verify

```powershell
.\ml\.venv\Scripts\python -c "import torch; print(torch.__version__); print(torch.xpu.is_available())"
```

Expected output:

```
2.9.1+xpu
True
```

If `is_available()` returns `False`, make sure your Intel Arc driver is fully up to date.

---

## Running the Pipeline

```powershell
# Use XPU explicitly
.\ml\.venv\Scripts\python ml\scripts\petbert_scan.py --device xpu

# Or let it auto-detect (priority: CUDA > XPU > MPS > CPU)
.\ml\.venv\Scripts\python ml\scripts\petbert_scan.py --device auto
```

---

## Common Pitfalls

| Mistake | Fix |
|---|---|
| Using Intel's IPEX wheel server (`pytorch-extension.intel.com`) | Use the official PyTorch XPU index: `https://download.pytorch.org/whl/xpu` |
| Running two separate `pip install` commands (e.g. torch first, then other packages) | Install everything in **one** `pip install` command so pip pins Intel runtime versions correctly |
| Installing `intel_extension_for_pytorch` (IPEX) | Not needed — XPU support is built directly into `torch 2.9.1+xpu` |
| Installing Intel oneAPI Base Toolkit or Deep Learning Essentials to get DLLs | Not needed — DLLs are installed as pip packages automatically |
| File lock errors (`[WinError 5] Access is denied`) during pip install | Close VS Code entirely and run pip from a standalone PowerShell window |
| `WinError 1114` on `c10.dll` when running the pipeline | `import torch` must happen before `import sklearn`/scipy — already fixed in the entry point script |
