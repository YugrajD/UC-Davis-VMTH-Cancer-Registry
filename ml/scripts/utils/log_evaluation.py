"""Append the current evaluation results to a persistent history log.

Run this after every evaluate_predictions.py call to record a snapshot.
The history is written to ml/output/evaluation/evaluation_history.csv and
a formatted trend table is printed to the console.

Usage:
  python ml/scripts/utils/log_evaluation.py
  python ml/scripts/utils/log_evaluation.py --label "after training v1"
  python ml/scripts/utils/log_evaluation.py --show          # print history without recording
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path


_SUMMARY_DEFAULT = "ml/output/evaluation/evaluation_summary.csv"
_HISTORY_DEFAULT = "ml/output/evaluation/evaluation_history.csv"

_HISTORY_FIELDS = [
    "timestamp",
    "label",
    "total",
    "good",
    "good_pct",
    "slightly_off",
    "slightly_off_pct",
    "completely_off",
    "completely_off_pct",
    "false_positive",
    "false_positive_pct",
    "false_negative",
    "false_negative_pct",
]


def _read_overall(summary_path: Path) -> dict:
    with open(summary_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["scope"] == "OVERALL":
                return row
    raise ValueError(f"No OVERALL row found in {summary_path}")


def _read_history(history_path: Path) -> list[dict]:
    if not history_path.exists():
        return []
    with open(history_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _append_history(history_path: Path, entry: dict) -> None:
    exists = history_path.exists()
    with open(history_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_HISTORY_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(entry)


def _delta(current: float, previous: float | None, higher_is_better: bool) -> str:
    if previous is None:
        return ""
    diff = current - previous
    if abs(diff) < 0.05:
        return " --"
    arrow = "^" if diff > 0 else "v"
    sign = "+" if diff > 0 else ""
    return f" {arrow}{sign}{diff:.1f}"


def _print_history(rows: list[dict]) -> None:
    if not rows:
        print("No history recorded yet.")
        return

    header = (
        f"{'#':>3}  {'Timestamp':<19}  {'Label':<25}  "
        f"{'Good%':>6}  {'Slight%':>7}  {'Off%':>6}  {'FP%':>6}  {'FN%':>6}  {'Total':>7}"
    )
    print()
    print("=== Evaluation History ===")
    print(header)
    print("-" * len(header))

    prev: dict | None = None
    for i, row in enumerate(rows, start=1):
        g   = float(row["good_pct"])
        sl  = float(row["slightly_off_pct"])
        off = float(row["completely_off_pct"])
        fp  = float(row["false_positive_pct"])
        fn  = float(row.get("false_negative_pct") or 0)

        pg   = float(prev["good_pct"])                      if prev else None
        psl  = float(prev["slightly_off_pct"])              if prev else None
        poff = float(prev["completely_off_pct"])            if prev else None
        pfp  = float(prev["false_positive_pct"])            if prev else None
        pfn  = float(prev.get("false_negative_pct") or 0)  if prev else None

        label = row["label"] or ""
        ts    = row["timestamp"][:19]  # trim microseconds

        print(
            f"{i:>3}  {ts:<19}  {label:<25}  "
            f"{g:>5.1f}%{_delta(g, pg, True)}  "
            f"{sl:>5.1f}%{_delta(sl, psl, True)}  "
            f"{off:>5.1f}%{_delta(off, poff, False)}  "
            f"{fp:>5.1f}%{_delta(fp, pfp, False)}  "
            f"{fn:>5.1f}%{_delta(fn, pfn, False)}  "
            f"{row['total']:>7}"
        )
        prev = row

    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Log evaluation results to history.")
    parser.add_argument("--summary", default=_SUMMARY_DEFAULT,
                        help="Path to evaluation_summary.csv")
    parser.add_argument("--history", default=_HISTORY_DEFAULT,
                        help="Path to the persistent history CSV")
    parser.add_argument("--label", default="",
                        help="Short description for this run (e.g. 'after training v1')")
    parser.add_argument("--show", action="store_true",
                        help="Print history without recording a new entry")
    args = parser.parse_args()

    summary_path = Path(args.summary)
    history_path = Path(args.history)

    if args.show:
        _print_history(_read_history(history_path))
        return 0

    if not summary_path.exists():
        print(f"Error: {summary_path} not found. Run evaluate_predictions.py first.")
        return 1

    overall = _read_overall(summary_path)
    entry = {
        "timestamp": datetime.now().isoformat(sep=" ", timespec="seconds"),
        "label":     args.label,
        "total":               overall["total"],
        "good":                overall["good"],
        "good_pct":            overall["good_pct"],
        "slightly_off":        overall["slightly_off"],
        "slightly_off_pct":    overall["slightly_off_pct"],
        "completely_off":      overall["completely_off"],
        "completely_off_pct":  overall["completely_off_pct"],
        "false_positive":      overall["false_positive"],
        "false_positive_pct":  overall["false_positive_pct"],
        "false_negative":      overall["false_negative"],
        "false_negative_pct":  overall["false_negative_pct"],
    }

    history_path.parent.mkdir(parents=True, exist_ok=True)
    _append_history(history_path, entry)

    history = _read_history(history_path)
    _print_history(history)
    print(f"Recorded to {history_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
