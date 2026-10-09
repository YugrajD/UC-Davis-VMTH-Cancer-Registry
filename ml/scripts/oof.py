"""k-fold cross-fit over the train partition: out-of-fold gate and term probabilities for the Report-Mapping audit.

Thin entry point: all the work lives in ``report_mapping.training.crossfit``. Each fold trains a fresh model from the
base checkpoint on the other folds (about 4-5 h on GPU for k=5 on the full train partition), so run it once per
silver lineage or retrain. The output is numbers only (case ids, probabilities, term names), written under
``config.OOF_DIR`` by default and read by ``audit.py rm-sample``.

  ml/.venv/Scripts/python.exe ml/scripts/oof.py --labels silver-0-legacy --split three-way-v1 --device cuda
  ml/.venv/Scripts/python.exe ml/scripts/oof.py --labels ml/output/coding/corrected_annotations.csv --k 2 --limit 200 --max-steps 5 --out scratch.csv
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from generations.guards import GuardViolation
from report_mapping.model import ENCODER_ID, resolve_device
from report_mapping.training.crossfit import crossfit
from report_mapping.training.recipe import PRODUCTION_SEED, Recipe


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", required=True, help="A silver_id (diagnosis_mapping.silver) or a labels CSV path")
    parser.add_argument("--split", default=None, help=f"Split id (default: {config.DEFAULT_SPLIT_ID})")
    parser.add_argument("--k", type=int, default=5, help="Number of folds")
    parser.add_argument("--seed", type=int, default=PRODUCTION_SEED)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--out", default=None, help="Output CSV (default: config.OOF_DIR/<labels>_<split>_k<k>.csv)")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output file")
    parser.add_argument("--model", default=ENCODER_ID, help="Base checkpoint dir or HF name each fold starts from")
    parser.add_argument("--limit", type=int, default=0, help="Smoke run: use the first N train cases only")
    parser.add_argument("--max-steps", type=int, default=0, help="Smoke run: stop each fold after N optimizer steps")
    args = parser.parse_args(argv)

    split_id = args.split if args.split is not None else config.DEFAULT_SPLIT_ID
    out = Path(args.out) if args.out is not None else config.OOF_DIR / f"{Path(args.labels).stem}_{split_id}_k{args.k}.csv"
    if out.exists() and not args.force:
        print(f"REFUSED: {out} already exists; pass --force to overwrite it", file=sys.stderr)
        return 1
    try:
        written = crossfit(args.labels, split_id, args.k, replace(Recipe(), seed=args.seed), resolve_device(args.device), out,
                           limit=args.limit, max_steps=args.max_steps, model_id=args.model)
    except GuardViolation as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 1
    print(f"cross-fit ({args.k} folds) -> {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
