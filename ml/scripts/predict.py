"""Thin entry point: stamped predictions from a report-mapping generation.

All the work lives in ``report_mapping.inference.predict.run_predict``; this script only parses args and calls it.
Writes ``config.PREDICTIONS_DIR/<generation_id>_predictions.csv`` unless ``--out`` is given.

  ml/.venv/Scripts/python.exe ml/scripts/predict.py
  ml/.venv/Scripts/python.exe ml/scripts/predict.py --generation candidate --device cuda --out PATH
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from report_mapping.generation import GenerationError, resolve_generation_dir
from report_mapping.inference.predict import run_predict
from report_mapping.model import resolve_device


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--generation", default="current",
                        help='"current" (default), "candidate", or a literal generation directory path.')
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--out", default=None, help="Predictions CSV path (default: config.PREDICTIONS_DIR).")
    args = parser.parse_args(argv)

    try:
        run_predict(resolve_generation_dir(args.generation), resolve_device(args.device),
                    Path(args.out) if args.out else None)
    except GenerationError as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
