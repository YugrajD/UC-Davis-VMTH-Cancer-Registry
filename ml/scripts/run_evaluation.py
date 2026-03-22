"""Run the evaluation pipeline standalone.

Scores existing production predictions against ground-truth labels
and logs results to evaluation history.

No env PYTHONPATH needed — this script adds ml/ to sys.path automatically.

Usage:
  python ml/scripts/run_evaluation.py
  python ml/scripts/run_evaluation.py --label "manual check"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.evaluate import evaluate
from evaluation.log_evaluation import log_evaluation


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate production predictions against ground truth.")
    parser.add_argument("--petbert-csv", default="ml/output/production/petbert_predictions.csv")
    parser.add_argument("--keyword-csv", default="ml/output/evaluation/keyword_predictions.csv")
    parser.add_argument("--out-dir", default="ml/output/evaluation")
    parser.add_argument("--label", default="", help="Label for evaluation history entry")
    args = parser.parse_args()

    evaluate(Path(args.petbert_csv), Path(args.keyword_csv), Path(args.out_dir))
    log_evaluation(label=args.label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
