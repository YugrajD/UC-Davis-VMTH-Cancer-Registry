import sys
import warnings
from pathlib import Path

# Import torch before any sklearn/scipy import to prevent a DLL init conflict
# on Windows with Intel XPU builds (WinError 1114 on c10.dll).
import torch  # noqa: F401

# Suppress noisy dependency deprecation warnings and print concise versions.
warnings.filterwarnings("ignore", message=".*resume_download.*", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pandas")
print("[Warning]: `resume_download` is deprecated and will be removed in version 1.0.0.")
print("[Warning]: `pyarrow` will become a required dependency of pandas 3.0.")

ML_DIR = Path(__file__).resolve().parents[1]
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from petbert_scan.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
