"""Train a report-mapping candidate generation (one step: labels + split -> candidate, calibration pending).

Thin entry point: all training logic lives in ``report_mapping.training.train``. Every run starts from the
base checkpoint (``--model``, default the production encoder) with a fresh head over the train-partition terms.
Thresholds come from ``calibrate.py``.

No env PYTHONPATH needed - this script adds ml/ to sys.path itself.

  ml/.venv/Scripts/python.exe ml/scripts/train.py --labels silver-0-legacy --split three-way-v1 --device cuda
  ml/.venv/Scripts/python.exe ml/scripts/train.py --labels ml/output/coding/corrected_annotations.csv --limit 200 --max-steps 5
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from generations.guards import GuardViolation
from report_mapping.generation import resolve_generation_dir
from report_mapping.model import ENCODER_ID, resolve_device
from report_mapping.training.recipe import PRODUCTION_SEED, Recipe
from report_mapping.training.train import train


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", required=True, help="A silver_id (diagnosis_mapping.silver) or a labels CSV path")
    parser.add_argument("--split", default=None, help=f"Split id (default: {config.DEFAULT_SPLIT_ID})")
    parser.add_argument("--seed", type=int, default=PRODUCTION_SEED)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--out", default="candidate", help="'candidate' or a literal directory (must be new or empty)")
    parser.add_argument("--model", default=ENCODER_ID, help="Base checkpoint dir or HF name to start from")
    parser.add_argument("--limit", type=int, default=0, help="Smoke run: train on the first N cases only")
    parser.add_argument("--max-steps", type=int, default=0, help="Smoke run: stop after N optimizer steps")
    args = parser.parse_args(argv)

    split_id = args.split if args.split is not None else config.DEFAULT_SPLIT_ID
    try:
        manifest = train(args.labels, split_id, resolve_generation_dir(args.out), replace(Recipe(), seed=args.seed),
                         resolve_device(args.device), model_id=args.model, limit=args.limit, max_steps=args.max_steps)
    except (GuardViolation, FileExistsError) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 1
    print(f"candidate {manifest['generation_id']} ({manifest['model']['n_terms']} terms, calibration pending) "
          f"-> {resolve_generation_dir(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
