import sys
from pathlib import Path

ML_DIR = Path(__file__).resolve().parents[1]
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from petbert_scan.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
