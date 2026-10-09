"""Thin entry point: fit a generation's thresholds (gate_t, tau, rho) and review cut on the calibration partition.

All the work lives in ``report_mapping.training.calibrate.calibrate``. The false-positive cap defaults to
``config.FP_CAP`` and the review cut targets ``config.REVIEW_TARGET_SHARE`` of accepted calibration cases.

  ml/.venv/Scripts/python.exe ml/scripts/calibrate.py --labels silver-0-legacy --split three-way-v1
  ml/.venv/Scripts/python.exe ml/scripts/calibrate.py --generation candidate --labels silver-0-legacy --max-fp-share 0.03
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from generations.guards import GuardViolation
from report_mapping.generation import GenerationError, resolve_generation_dir
from report_mapping.model import resolve_device
from report_mapping.training.calibrate import calibrate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--generation", default="candidate",
                        help='"current", "candidate" (default), or a literal generation directory path.')
    parser.add_argument("--labels", required=True, help="silver_id or labels-table CSV path.")
    parser.add_argument("--split", default=None, help=f"split_id holding the calibration partition (default: {config.DEFAULT_SPLIT_ID}).")
    parser.add_argument("--max-fp-share", type=float, default=None,
                        help=f"False-positive cap on the grid (default: config.FP_CAP = {config.FP_CAP}).")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args(argv)

    split_id = args.split if args.split is not None else config.DEFAULT_SPLIT_ID
    max_fp_share = args.max_fp_share if args.max_fp_share is not None else config.FP_CAP
    try:
        block = calibrate(resolve_generation_dir(args.generation), args.labels, split_id, max_fp_share,
                          config.REVIEW_TARGET_SHARE, resolve_device(args.device))
    except (GuardViolation, GenerationError, ValueError) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 1
    print(f"calibrated on {block['partition']} of {block['split_id']} ({block['n_cases']} cases, "
          f"{block['n_cases_with_text']} with text; {block['n_accepted']} accepted)")
    print("  " + ", ".join(f"{k}={v}" for k, v in block["values"].items()) + f", max_fp_share={max_fp_share}")
    print(f"  in-sample G+S {100 * block['partition_gs_share']:.2f}%, Good {100 * block['partition_good_share']:.2f}%; "
          f"review share below cut {100 * block['review_share_below_cut']:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
