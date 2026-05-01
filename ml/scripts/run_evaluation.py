"""Score the latest predictions and record results to evaluation history.

No env PYTHONPATH needed — this script adds ml/ to sys.path automatically.

Auto-detects which prediction file to evaluate (contrastive-backbone predictions
preferred; falls back to binary-backbone predictions).

Usage:
  python ml/scripts/run_evaluation.py
  python ml/scripts/run_evaluation.py --label "after cycle 3"
  python ml/scripts/run_evaluation.py --prediction-csv ml/output/production/binary/petbert_predictions.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import config
from evaluation import evaluate, log_evaluation


def main() -> int:
    subdir = config.best_predictions_subdir()

    parser = argparse.ArgumentParser(
        description="Score predictions against verified labels and record results to history."
    )
    parser.add_argument(
        "--prediction-csv",
        default=f"{config.OUTPUT_PRODUCTION_DIR}/{subdir}/petbert_predictions.csv",
        help="Predictions file to evaluate.",
    )
    parser.add_argument(
        "--annotation-csv",
        default=config.ANNOTATION_CSV,
        help="Verified label annotations to score against.",
    )
    parser.add_argument(
        "--out-dir",
        default=f"{config.OUTPUT_EVALUATION_DIR}/{subdir}",
        help="Directory to write evaluation results.",
    )
    parser.add_argument(
        "--label", default="",
        help="Short description for this evaluation entry (e.g. 'manual check').",
    )
    parser.add_argument(
        "--test-cases",
        default="",
        help="Path to test_cases.txt. When provided, only held-out test cases are "
             "evaluated. Generate with ml/training/data/create_split.py.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    evaluate(Path(args.prediction_csv), Path(args.annotation_csv), out_dir,
             cases_txt=args.test_cases)
    log_evaluation(
        summary=str(out_dir / "evaluation_summary.csv"),
        history=str(out_dir / "evaluation_history.csv"),
        label=args.label,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
