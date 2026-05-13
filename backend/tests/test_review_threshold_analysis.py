"""Unit tests for PetBERT review-threshold analysis helpers."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.analyze_review_thresholds import (  # noqa: E402
    load_validation_rows,
    recommend_confidence_threshold,
    recommend_margin_threshold,
    recommend_review_priority_threshold,
    summarize_confidence_bins,
    summarize_margin_buckets,
)


def _write_validation_csv(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "diagnosis_id,confidence,top2_margin,ground_truth_correct,verdict",
                "a,0.01,0.01,false,completely_off",
                "b,0.02,0.04,false,false_positive",
                "c,0.20,0.08,false,completely_off",
                "d,0.21,0.12,false,completely_off",
                "e,0.23,0.16,true,slightly_off",
                "f,0.24,0.20,true,good",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_confidence_and_margin_recommendations(tmp_path: Path):
    csv_path = tmp_path / "validation.csv"
    _write_validation_csv(csv_path)

    rows = load_validation_rows(csv_path)

    assert recommend_confidence_threshold(rows, target_precision=0.95, min_count=2) == 0.23
    assert recommend_review_priority_threshold(rows, review_precision=0.85, min_count=2) == 0.23
    assert recommend_margin_threshold(rows, review_precision=0.85, min_count=2) == 0.15


def test_precision_summaries_are_bucketed(tmp_path: Path):
    csv_path = tmp_path / "validation.csv"
    _write_validation_csv(csv_path)

    rows = load_validation_rows(csv_path)
    confidence_bins = summarize_confidence_bins(rows, bin_width=0.05)
    margin_bins = summarize_margin_buckets(rows)

    assert [(row.label, row.total, row.correct) for row in confidence_bins] == [
        ("0.00-0.05", 2, 0),
        ("0.20-0.25", 4, 2),
    ]
    assert [(row.label, row.total, row.correct) for row in margin_bins] == [
        ("<0.05", 2, 0),
        ("0.05-0.10", 1, 0),
        (">=0.10", 3, 2),
    ]
