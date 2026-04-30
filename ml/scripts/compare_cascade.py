"""Diff two evaluation_summary.csv files and print a per-scope delta table.

Usage:
  python ml/scripts/compare_cascade.py \\
      --baseline ml/output/evaluation/group_baseline/evaluation_summary.csv \\
      --cascade  ml/output/evaluation/group_cascade/evaluation_summary.csv
"""

import argparse
import csv
import sys
from pathlib import Path


def load_summary(path: Path) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row["scope"]] = {
                "total": int(row["total"]),
                "good_pct": float(row["good_pct"]),
                "slightly_off_pct": float(row["slightly_off_pct"]),
                "completely_off_pct": float(row["completely_off_pct"]),
                "false_positive_pct": float(row["false_positive_pct"]),
                "false_negative_pct": float(row["false_negative_pct"]),
            }
    return rows


def fmt_delta(d: float) -> str:
    sign = "+" if d > 0 else ""
    return f"{sign}{d:.1f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--cascade", required=True, type=Path)
    parser.add_argument(
        "--min-n", type=int, default=10,
        help="Hide groups with fewer than this many cases in the baseline.",
    )
    args = parser.parse_args()

    base = load_summary(args.baseline)
    casc = load_summary(args.cascade)

    scopes = ["OVERALL"] + sorted(
        s for s in base.keys() if s != "OVERALL"
    )

    metric_cols = [
        ("good_pct", "Good"),
        ("slightly_off_pct", "Slight"),
        ("completely_off_pct", "Off"),
        ("false_positive_pct", "FP"),
        ("false_negative_pct", "FN"),
    ]

    header = f"{'Scope':<45} {'n':>5}  " + "  ".join(
        f"{label:>6} -> {label:>6} (Δ)" for _, label in metric_cols
    )
    print(header)
    print("-" * len(header))

    for scope in scopes:
        if scope not in casc:
            continue
        b = base[scope]
        c = casc[scope]
        if scope != "OVERALL" and b["total"] < args.min_n:
            continue
        row = f"{scope:<45} {b['total']:>5}  "
        for key, _ in metric_cols:
            d = c[key] - b[key]
            row += f"{b[key]:>6.1f} -> {c[key]:>6.1f} ({fmt_delta(d):>5})  "
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
