"""Compute annotation coverage statistics and write to output/data_analysis/.

Emits per-analysis CSV + PNG artifacts plus a combined text report.

No PYTHONPATH needed -- this script adds ml/ to sys.path automatically.

Usage:
  python ml/scripts/run_data_analysis.py
  python ml/scripts/run_data_analysis.py --annotation-csv ml/output/annotation/llm/llm_annotation.csv
  python ml/scripts/run_data_analysis.py --no-plots
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import config
from analysis.annotation_stats import run_analysis


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute annotation coverage statistics and write per-analysis + combined artifacts."
    )
    parser.add_argument(
        "--annotation-csv",
        default=config.LLM_ANNOTATION_CSV,
        help="Annotation CSV to analyse (default: LLM annotation).",
    )
    parser.add_argument(
        "--labels-csv",
        default=config.LABELS_CSV,
        help="Taxonomy labels CSV.",
    )
    parser.add_argument(
        "--out-dir",
        default=config.DATA_ANALYSIS_DIR,
        help="Directory to write CSV + PNG + combined text report.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip PNG plot generation; CSVs and text report still written.",
    )
    args = parser.parse_args()

    run_analysis(
        annotation_csv=args.annotation_csv,
        labels_csv=args.labels_csv,
        out_dir=args.out_dir,
        make_plots=not args.no_plots,
    )
    print(f"\nWrote artifacts to: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
