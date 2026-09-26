"""Thin entry point: fit a generation's thresholds on the calibration partition.

Replaces ``ml/scripts/sweep_lp_thresholds.py`` and ``sweep_tail_gate.py``. All
the work lives in ``report_mapping.training.calibrate.calibrate``.

  ml/.venv/bin/python ml/next/scripts/calibrate.py --generation candidate --labels silver-0-legacy --split three-way-v1
  ml/.venv/bin/python ml/next/scripts/calibrate.py --generation candidate --labels silver-0-legacy --split three-way-v1 --legacy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from report_mapping.model.generation import resolve_generation_dir
from report_mapping.training.calibrate import CALIBRATION_PARTITION, calibrate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--generation", required=True,
                        help='"current", "candidate", or a literal generation directory path.')
    parser.add_argument("--labels", required=True, help="silver_id or labels-table CSV path.")
    parser.add_argument("--split", required=True, help="split_id holding the calibration partition.")
    parser.add_argument("--partition", default=CALIBRATION_PARTITION)
    parser.add_argument("--legacy", action="store_true",
                        help="Parity L3: fit only the per-LP thresholds; keep gate 0.80, group 0.85, K=2, gap 0.08.")
    parser.add_argument("--device", default="cpu", help="Device for the heads (cpu, cuda, mps, xpu).")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = calibrate(resolve_generation_dir(args.generation), labels=args.labels, split_id=args.split,
                       partition=args.partition, legacy=args.legacy, device=args.device)
    values = result["values"]
    print(f"calibrated on {result['partition']} of {result['split_id']} ({result['n_cases']} cases, "
          f"legacy={result['legacy']})")
    print("  " + ", ".join(f"{k}={v}" for k, v in values.items() if k != "label_presence"))
    print(f"  {len(values['label_presence'])} per-LP thresholds; partition G+S {100 * result['partition_gs_share']:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
