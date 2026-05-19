"""Score the latest predictions and record results to evaluation history.

No env PYTHONPATH needed — this script adds ml/ to sys.path automatically.

Auto-detects which prediction file to evaluate (contrastive-backbone predictions
preferred; falls back to binary-backbone predictions).

Usage:
  python ml/scripts/run_evaluation.py
  python ml/scripts/run_evaluation.py --label "after cycle 3"
  python ml/scripts/run_evaluation.py --prediction-csv ml/output/production/binary/petbert_predictions.csv

Per-stage evaluation (4-stage pipeline):
  python ml/scripts/run_evaluation.py --stage case-presence --test-cases ml/output/splits/test_cases.txt
  python ml/scripts/run_evaluation.py --stage groups        --test-cases ml/output/splits/test_cases.txt
  python ml/scripts/run_evaluation.py --stage label-presence --test-cases ml/output/splits/test_cases.txt
  python ml/scripts/run_evaluation.py --stage all           --test-cases ml/output/splits/test_cases.txt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import config
from evaluation import (
    evaluate,
    evaluate_case_based,
    evaluate_case_presence,
    evaluate_common_labels,
    evaluate_groups,
    evaluate_label_presence,
    evaluate_top_n_verdicts,
    log_evaluation,
)


_STAGE_CHOICES = ["pipeline", "case-based", "common-labels", "top-n-verdicts", "case-presence", "groups", "label-presence", "all"]
_DEFAULT_THRESHOLDS = {
    "case-presence": 0.5,
    "groups": 0.85,
    "label-presence": 0.5,
}


def main() -> int:
    subdir = config.BEST_PREDICTIONS_SUBDIR

    parser = argparse.ArgumentParser(
        description="Score predictions against verified labels and record results to history. "
                    "Use --stage to evaluate one classifier in the 4-stage pipeline.",
    )
    parser.add_argument(
        "--stage", choices=_STAGE_CHOICES, default="pipeline",
        help="Which evaluation to run. 'pipeline' is the existing end-to-end "
             "evaluation. 'case-presence', 'groups', 'label-presence' isolate "
             "one classifier each. 'all' runs every evaluation.",
    )
    # Pipeline (existing) flags
    parser.add_argument(
        "--prediction-csv",
        default=f"{config.OUTPUT_PRODUCTION_DIR}/{subdir}/petbert_predictions.csv",
        help="Predictions file to evaluate (pipeline stage only).",
    )
    parser.add_argument(
        "--annotation-csv", default=config.ANNOTATION_CSV,
        help="Verified label annotations to score against.",
    )
    parser.add_argument(
        "--out-dir", default=f"{config.OUTPUT_EVALUATION_DIR}/{subdir}",
        help="Directory to write evaluation results.",
    )
    parser.add_argument(
        "--label", default="",
        help="Short description for this evaluation entry (e.g. 'manual check').",
    )
    parser.add_argument(
        "--test-cases", default="",
        help="Path to test_cases.txt. When provided, only held-out test cases are evaluated.",
    )
    parser.add_argument(
        "--uncommon-groups", default=config.UNCOMMON_GROUPS_TXT,
        help="Path to uncommon_groups.txt.",
    )
    # Per-stage flags
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Override default threshold for the selected stage "
             "(case-presence=0.5, groups=0.85, label-presence=0.5). Ignored for 'pipeline' and 'all'.",
    )
    parser.add_argument(
        "--case-presence-classifier", default=config.CASE_PRESENCE_CLASSIFIER_PT,
        help="CasePresenceClassifier checkpoint path (Stage 1).",
    )
    parser.add_argument(
        "--group-classifier",
        default=f"{config.CHECKPOINT_GROUP_DIR}/group_classifier_best.pt",
        help="GroupClassifier checkpoint path (Stage 2).",
    )
    parser.add_argument(
        "--label-presence-classifier-dir", default=config.CHECKPOINT_LABEL_PRESENCE_DIR,
        help="Directory of per-group LabelPresenceClassifier checkpoints (Stage 3).",
    )
    parser.add_argument(
        "--embedding-cache", default=config.EMBEDDING_CACHE_NPZ,
        help="Embedding cache NPZ used for per-stage scoring.",
    )
    parser.add_argument(
        "--labels-csv", default=config.LABELS_CSV,
        help="Vet-ICD-O labels CSV.",
    )
    parser.add_argument(
        "--top-n", type=int, default=20,
        help="Number of most-common labels to report (common-labels stage only).",
    )
    parser.add_argument(
        "--top-ns", default="25,50,100",
        help="Comma-separated Ns for top-n-verdicts stage (default: 25,50,100).",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    def _t(stage: str) -> float:
        return args.threshold if args.threshold is not None else _DEFAULT_THRESHOLDS[stage]

    def _run_pipeline() -> None:
        evaluate(
            Path(args.prediction_csv), Path(args.annotation_csv), out_dir,
            cases_txt=args.test_cases, uncommon_groups_file=args.uncommon_groups,
        )
        log_evaluation(
            summary=str(out_dir / "evaluation_summary.csv"),
            history=str(out_dir / "evaluation_history.csv"),
            label=args.label,
        )

    def _run_case_based() -> None:
        evaluate_case_based(
            Path(args.prediction_csv), Path(args.annotation_csv), out_dir,
            cases_txt=args.test_cases, uncommon_groups_file=args.uncommon_groups,
        )
        log_evaluation(
            summary=str(out_dir / "case_based_summary.csv"),
            history=str(out_dir / "case_based_history.csv"),
            label=args.label,
        )

    def _run_common_labels() -> None:
        evaluate_common_labels(
            Path(args.prediction_csv), Path(args.annotation_csv), out_dir,
            cases_txt=args.test_cases, uncommon_groups_file=args.uncommon_groups,
            top_n=args.top_n,
        )

    def _run_top_n_verdicts() -> None:
        top_ns = tuple(int(x) for x in args.top_ns.split(",") if x.strip())
        evaluate_top_n_verdicts(
            Path(args.prediction_csv), Path(args.annotation_csv), out_dir,
            cases_txt=args.test_cases, uncommon_groups_file=args.uncommon_groups,
            top_ns=top_ns,
        )

    def _run_case_presence() -> None:
        evaluate_case_presence(
            classifier_path=Path(args.case_presence_classifier),
            embedding_cache_path=Path(args.embedding_cache),
            annotation_csv=Path(args.annotation_csv),
            out_dir=out_dir,
            cases_txt=args.test_cases,
            threshold=_t("case-presence"),
            history_label=args.label,
        )

    def _run_groups() -> None:
        evaluate_groups(
            classifier_path=Path(args.group_classifier),
            embedding_cache_path=Path(args.embedding_cache),
            annotation_csv=Path(args.annotation_csv),
            out_dir=out_dir,
            cases_txt=args.test_cases,
            threshold=_t("groups"),
            history_label=args.label,
        )

    def _run_label_presence() -> None:
        evaluate_label_presence(
            classifier_dir=Path(args.label_presence_classifier_dir),
            embedding_cache_path=Path(args.embedding_cache),
            annotation_csv=Path(args.annotation_csv),
            labels_csv=args.labels_csv,
            out_dir=out_dir,
            cases_txt=args.test_cases,
            uncommon_groups_path=args.uncommon_groups,
            threshold=_t("label-presence"),
            history_label=args.label,
        )

    if args.stage == "pipeline":
        _run_pipeline()
    elif args.stage == "case-based":
        _run_case_based()
    elif args.stage == "common-labels":
        _run_common_labels()
    elif args.stage == "top-n-verdicts":
        _run_top_n_verdicts()
    elif args.stage == "case-presence":
        _run_case_presence()
    elif args.stage == "groups":
        _run_groups()
    elif args.stage == "label-presence":
        _run_label_presence()
    elif args.stage == "all":
        _run_pipeline()
        _run_case_based()
        _run_common_labels()
        _run_top_n_verdicts()
        _run_case_presence()
        _run_groups()
        _run_label_presence()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
