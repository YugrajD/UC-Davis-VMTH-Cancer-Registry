#!/usr/bin/env python3
"""Analyze PetBERT manual-review thresholds from labelled predictions.

The backend review gate auto-confirms a diagnosis only when confidence is high
enough and the top-1/top-2 margin is wide enough. This script measures labelled
prediction precision by confidence and margin so those settings can be chosen
from validation data instead of intuition.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


CORRECT_VERDICTS = {"good", "slightly_off", "true_positive", "correct"}
INCORRECT_VERDICTS = {
    "completely_off",
    "false_positive",
    "false_negative",
    "incorrect",
    "wrong",
}
DEFAULT_MARGIN_CANDIDATES = (0.05, 0.10, 0.15, 0.20)


@dataclass(frozen=True)
class ValidationRow:
    diagnosis_id: str
    confidence: float
    ground_truth_correct: bool
    top2_margin: float | None
    cancer_type: str
    verdict: str


@dataclass(frozen=True)
class PrecisionSummary:
    label: str
    total: int
    correct: int

    @property
    def precision(self) -> float | None:
        if self.total == 0:
            return None
        return self.correct / self.total


def parse_bool(raw: str) -> bool:
    value = (raw or "").strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {raw!r}")


def parse_optional_float(raw: str) -> float | None:
    value = (raw or "").strip()
    if not value:
        return None
    return float(value)


def row_value(row: dict[str, str], *names: str) -> str:
    for name in names:
        value = row.get(name)
        if value is not None and value != "":
            return value
    return ""


def correctness_from_row(row: dict[str, str]) -> bool:
    raw_correct = row_value(row, "ground_truth_correct", "correct", "is_correct")
    if raw_correct:
        return parse_bool(raw_correct)

    verdict = row_value(row, "verdict", "label").strip().lower()
    if verdict in CORRECT_VERDICTS:
        return True
    if verdict in INCORRECT_VERDICTS:
        return False
    raise ValueError(
        "Each validation row must include ground_truth_correct or a known verdict."
    )


def load_validation_rows(path: Path) -> list[ValidationRow]:
    rows: list[ValidationRow] = []
    with path.open(newline="", encoding="utf-8") as f:
        for line_number, raw in enumerate(csv.DictReader(f), start=2):
            confidence_raw = row_value(raw, "confidence", "score")
            if not confidence_raw:
                raise ValueError(f"Line {line_number}: missing confidence")
            confidence = float(confidence_raw)
            diagnosis_id = row_value(raw, "diagnosis_id")
            if not diagnosis_id:
                case_id = row_value(raw, "case_id")
                diagnosis_index = row_value(raw, "diagnosis_index")
                diagnosis_id = f"{case_id}:{diagnosis_index}" if case_id else str(line_number)
            rows.append(
                ValidationRow(
                    diagnosis_id=diagnosis_id,
                    confidence=confidence,
                    ground_truth_correct=correctness_from_row(raw),
                    top2_margin=parse_optional_float(
                        row_value(raw, "top2_margin", "margin")
                    ),
                    cancer_type=row_value(
                        raw, "cancer_type", "predicted_group", "predicted_cancer_type"
                    ),
                    verdict=row_value(raw, "verdict"),
                )
            )
    if not rows:
        raise ValueError(f"No validation rows found in {path}")
    return rows


def precision_for(rows: Iterable[ValidationRow], label: str) -> PrecisionSummary:
    materialized = list(rows)
    return PrecisionSummary(
        label=label,
        total=len(materialized),
        correct=sum(1 for row in materialized if row.ground_truth_correct),
    )


def bin_floor(value: float, width: float) -> float:
    return math.floor((value + 1e-12) / width) * width


def summarize_confidence_bins(
    rows: list[ValidationRow], bin_width: float
) -> list[PrecisionSummary]:
    bins: dict[float, list[ValidationRow]] = defaultdict(list)
    for row in rows:
        bins[bin_floor(row.confidence, bin_width)].append(row)
    summaries: list[PrecisionSummary] = []
    for lower in sorted(bins):
        upper = lower + bin_width
        summaries.append(precision_for(bins[lower], f"{lower:.2f}-{upper:.2f}"))
    return summaries


def summarize_cumulative_thresholds(
    rows: list[ValidationRow],
) -> list[tuple[float, PrecisionSummary]]:
    thresholds = sorted({round(row.confidence, 4) for row in rows})
    summaries: list[tuple[float, PrecisionSummary]] = []
    for threshold in thresholds:
        matching = [row for row in rows if row.confidence >= threshold]
        summaries.append((threshold, precision_for(matching, f">={threshold:.2f}")))
    return summaries


def summarize_margin_buckets(rows: list[ValidationRow]) -> list[PrecisionSummary]:
    with_margin = [row for row in rows if row.top2_margin is not None]
    return [
        precision_for(
            [row for row in with_margin if row.top2_margin is not None and row.top2_margin < 0.05],
            "<0.05",
        ),
        precision_for(
            [
                row
                for row in with_margin
                if row.top2_margin is not None and 0.05 <= row.top2_margin < 0.10
            ],
            "0.05-0.10",
        ),
        precision_for(
            [row for row in with_margin if row.top2_margin is not None and row.top2_margin >= 0.10],
            ">=0.10",
        ),
    ]


def recommend_confidence_threshold(
    rows: list[ValidationRow], target_precision: float, min_count: int
) -> float | None:
    for threshold, summary in summarize_cumulative_thresholds(rows):
        if summary.total < min_count:
            continue
        if summary.precision is not None and summary.precision >= target_precision:
            return threshold
    return None


def recommend_review_priority_threshold(
    rows: list[ValidationRow], review_precision: float, min_count: int
) -> float | None:
    candidate: float | None = None
    for threshold, summary in summarize_cumulative_thresholds(rows):
        if summary.total < min_count:
            continue
        if summary.precision is not None and summary.precision >= review_precision:
            candidate = threshold
            break
    return candidate


def recommend_margin_threshold(
    rows: list[ValidationRow],
    review_precision: float,
    min_count: int,
    candidates: Iterable[float] = DEFAULT_MARGIN_CANDIDATES,
) -> float | None:
    with_margin = [row for row in rows if row.top2_margin is not None]
    chosen: float | None = None
    for threshold in candidates:
        below = [row for row in with_margin if row.top2_margin is not None and row.top2_margin < threshold]
        above = [row for row in with_margin if row.top2_margin is not None and row.top2_margin >= threshold]
        if len(below) < min_count or len(above) < min_count:
            continue
        below_precision = precision_for(below, "below").precision
        above_precision = precision_for(above, "above").precision
        if (
            below_precision is not None
            and above_precision is not None
            and below_precision < review_precision
            and above_precision > below_precision
        ):
            chosen = threshold
    return chosen


def format_precision(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def print_table(title: str, rows: list[PrecisionSummary]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(f"{'bucket':<14} {'n':>6} {'correct':>8} {'precision':>10}")
    for row in rows:
        print(
            f"{row.label:<14} {row.total:>6} {row.correct:>8} "
            f"{format_precision(row.precision):>10}"
        )


def analyze(args: argparse.Namespace) -> int:
    rows = load_validation_rows(Path(args.validation_csv))
    print(f"Loaded {len(rows)} labelled prediction rows from {args.validation_csv}")

    print_table(
        "Precision by confidence bin",
        summarize_confidence_bins(rows, args.bin_width),
    )

    cumulative = [
        summary
        for _threshold, summary in summarize_cumulative_thresholds(rows)
        if summary.total >= args.min_report_count
    ]
    print_table("Cumulative precision at/above threshold", cumulative)

    margin_rows = [row for row in rows if row.top2_margin is not None]
    if margin_rows:
        print_table("Precision by top-1/top-2 margin bucket", summarize_margin_buckets(rows))
    else:
        print("\nNo top2_margin values found; margin gate cannot be validated.")

    confidence_threshold = recommend_confidence_threshold(
        rows, args.target_precision, args.min_auto_accept_count
    )
    review_threshold = recommend_review_priority_threshold(
        rows, args.review_precision, args.min_auto_accept_count
    )
    margin_threshold = recommend_margin_threshold(
        rows, args.review_precision, args.min_margin_count
    )

    print("\nRecommendation")
    print("--------------")
    if confidence_threshold is None:
        print(
            "REVIEW_AUTO_ACCEPT_CONFIDENCE: no threshold met "
            f"{args.target_precision:.0%} precision with n>={args.min_auto_accept_count}; "
            "do not auto-accept from this validation set."
        )
    else:
        print(f"REVIEW_AUTO_ACCEPT_CONFIDENCE={confidence_threshold:.2f}")
    if margin_threshold is None:
        print(
            "REVIEW_AUTO_ACCEPT_MARGIN: no validated margin threshold; "
            "keep the existing gate until more labelled top-2 data exists."
        )
    else:
        print(f"REVIEW_AUTO_ACCEPT_MARGIN={margin_threshold:.2f}")
    if review_threshold is not None:
        print(
            f"Review-priority confidence boundary: predictions below {review_threshold:.2f} "
            f"fall under the {args.review_precision:.0%} precision band."
        )
    return 0


def prediction_confidences_by_case(prediction_csv: Path) -> dict[str, list[tuple[int, float]]]:
    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    with prediction_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            diagnosis_index = row_value(row, "diagnosis_index")
            if not diagnosis_index:
                continue
            grouped[row["case_id"]].append(
                (int(diagnosis_index), float(row_value(row, "confidence") or 0.0))
            )
    for rows in grouped.values():
        rows.sort(key=lambda item: item[0])
    return grouped


def build_validation_sample(args: argparse.Namespace) -> int:
    evaluation_csv = Path(args.evaluation_csv)
    prediction_csv = Path(args.prediction_csv)
    output_csv = Path(args.output_validation_csv)

    grouped_predictions = prediction_confidences_by_case(prediction_csv)
    source_rows: list[dict[str, str]] = []
    with evaluation_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            diagnosis_index = row_value(row, "diagnosis_index")
            if not diagnosis_index:
                continue
            confidence = float(row_value(row, "confidence") or 0.0)
            case_id = row["case_id"]
            rank = int(diagnosis_index)
            case_predictions = grouped_predictions.get(case_id, [])
            top2_margin = ""
            if rank == 1 and len(case_predictions) > 1:
                top2_margin = f"{confidence - case_predictions[1][1]:.2f}"
            row_out = {
                "diagnosis_id": f"{case_id}:{rank}",
                "case_id": case_id,
                "diagnosis_index": str(rank),
                "predicted_term": row_value(row, "predicted_term"),
                "predicted_group": row_value(row, "predicted_group"),
                "predicted_code": row_value(row, "predicted_code"),
                "confidence": f"{confidence:.2f}",
                "top2_margin": top2_margin,
                "ground_truth_correct": str(correctness_from_row(row)).lower(),
                "verdict": row_value(row, "verdict"),
                "validation_source": str(evaluation_csv),
                "notes": "Seeded from existing evaluation verdict; replace with clinician review when available.",
            }
            source_rows.append(row_out)

    rng = random.Random(args.seed)
    sampled: list[dict[str, str]] = []
    bins: dict[float, list[dict[str, str]]] = defaultdict(list)
    for row in source_rows:
        confidence = float(row["confidence"])
        if confidence >= args.include_all_at_or_above:
            sampled.append(row)
        else:
            bins[bin_floor(confidence, args.bin_width)].append(row)

    for lower in sorted(bins):
        bucket = sorted(bins[lower], key=lambda row: row["diagnosis_id"])
        if len(bucket) <= args.per_bin_sample:
            sampled.extend(bucket)
        else:
            sampled.extend(rng.sample(bucket, args.per_bin_sample))

    sampled.sort(key=lambda row: (float(row["confidence"]), row["diagnosis_id"]))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "diagnosis_id",
        "case_id",
        "diagnosis_index",
        "predicted_term",
        "predicted_group",
        "predicted_code",
        "confidence",
        "top2_margin",
        "ground_truth_correct",
        "verdict",
        "validation_source",
        "notes",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sampled)

    print(f"Wrote {len(sampled)} validation rows to {output_csv}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze or build PetBERT review-threshold validation data."
    )
    subparsers = parser.add_subparsers(dest="command")

    analyze_parser = subparsers.add_parser("analyze", help="Analyze a validation CSV.")
    analyze_parser.add_argument(
        "--validation-csv",
        default="ml/data/validation/review_threshold_validation.csv",
        help="Labelled validation CSV with confidence and ground_truth_correct columns.",
    )
    analyze_parser.add_argument("--bin-width", type=float, default=0.05)
    analyze_parser.add_argument("--target-precision", type=float, default=0.95)
    analyze_parser.add_argument("--review-precision", type=float, default=0.85)
    analyze_parser.add_argument("--min-auto-accept-count", type=int, default=30)
    analyze_parser.add_argument("--min-margin-count", type=int, default=30)
    analyze_parser.add_argument("--min-report-count", type=int, default=10)
    analyze_parser.set_defaults(func=analyze)

    sample_parser = subparsers.add_parser(
        "build-sample",
        help="Build a deterministic validation seed from existing evaluation outputs.",
    )
    sample_parser.add_argument(
        "--evaluation-csv",
        default="ml/output/finetuned_eval/evaluation.csv",
        help="Existing evaluation.csv with verdict labels.",
    )
    sample_parser.add_argument(
        "--prediction-csv",
        default="ml/output/finetuned_eval/petbert_predictions.csv",
        help="Prediction CSV used to compute top-1/top-2 margins.",
    )
    sample_parser.add_argument(
        "--output-validation-csv",
        default="ml/data/validation/review_threshold_validation.csv",
    )
    sample_parser.add_argument("--bin-width", type=float, default=0.05)
    sample_parser.add_argument("--per-bin-sample", type=int, default=25)
    sample_parser.add_argument("--include-all-at-or-above", type=float, default=0.20)
    sample_parser.add_argument("--seed", type=int, default=193)
    sample_parser.set_defaults(func=build_validation_sample)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
