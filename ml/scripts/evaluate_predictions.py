"""Compare petbert_scan_predictions.csv against keyword_predictions.csv.

For every row in petbert_scan_predictions, assign one of four verdicts:
  good           — predicted_term exactly matches a matched_term for this case
  slightly_off   — no exact term match, but predicted_group matches a matched_group
  completely_off — neither term nor group matches any keyword label for this case
  false_positive — the case has no keyword labels at all (should have been Uncategorized)

Output (written to --out-dir):
  evaluation.csv         — petbert_scan_predictions columns + verdict
  evaluation_summary.csv — overall + per predicted-group counts and percentages
"""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def load_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def score_prediction(predicted_term: str, predicted_group: str,
                     matched_terms: set[str], matched_groups: set[str]) -> str:
    """Score one petbert prediction against the case's keyword label sets."""
    if not matched_terms:
        return "false_positive"
    if predicted_term in matched_terms:
        return "good"
    if predicted_group in matched_groups:
        return "slightly_off"
    return "completely_off"


def evaluate(petbert_csv: Path, keyword_csv: Path, out_dir: Path) -> None:
    pb_rows = load_csv(petbert_csv)
    kw_rows = load_csv(keyword_csv)

    # Build per-case label sets from keyword predictions
    case_terms: dict[str, set[str]] = defaultdict(set)
    case_groups: dict[str, set[str]] = defaultdict(set)
    for row in kw_rows:
        if row["matched_term"].strip():
            case_terms[row["case_id"]].add(row["matched_term"])
        if row["matched_group"].strip():
            case_groups[row["case_id"]].add(row["matched_group"])

    # Score every petbert prediction row
    out_rows: list[dict] = []
    for row in pb_rows:
        cid = row["case_id"]
        verdict = score_prediction(
            row["predicted_term"], row["predicted_group"],
            case_terms.get(cid, set()), case_groups.get(cid, set()),
        )
        out_rows.append({**row, "verdict": verdict})

    # ── Write output CSV ──────────────────────────────────────────────────────
    out_path = out_dir / "evaluation.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(pb_rows[0].keys()) + ["verdict"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    # ── Console summary ───────────────────────────────────────────────────────
    counts = Counter(r["verdict"] for r in out_rows)
    total = len(out_rows)

    print(f"\n=== Overall ({total} predictions) ===")
    for label, key in [
        ("Good", "good"),
        ("Slightly off", "slightly_off"),
        ("Completely off", "completely_off"),
        ("False positive", "false_positive"),
    ]:
        n = counts[key]
        print(f"  {label:<22} {n:>5}  ({n / total * 100:.1f}%)")

    # Per-group breakdown (excluding false positives — they have no matched group)
    group_counts: dict[str, Counter] = defaultdict(Counter)
    for row in out_rows:
        if row["verdict"] == "false_positive":
            continue
        # Use the predicted_group as the breakdown axis so every row has one
        group_counts[row["predicted_group"]][row["verdict"]] += 1

    print("\n=== Per predicted-group breakdown (excluding false positives) ===")
    header = f"{'Group':<45} {'Total':>5}  {'Good':>6}  {'Slight':>6}  {'Off':>6}"
    print(header)
    print("-" * len(header))
    for group, c in sorted(group_counts.items(), key=lambda x: -sum(x[1].values())):
        n = sum(c.values())
        print(
            f"{group:<45} {n:>5}  "
            f"{c['good']:>5} ({c['good']/n*100:>4.0f}%)  "
            f"{c['slightly_off']:>5} ({c['slightly_off']/n*100:>4.0f}%)  "
            f"{c['completely_off']:>5} ({c['completely_off']/n*100:>4.0f}%)"
        )

    # ── Write summary CSV ─────────────────────────────────────────────────────
    summary_fields = [
        "scope", "total",
        "good", "good_pct",
        "slightly_off", "slightly_off_pct",
        "completely_off", "completely_off_pct",
        "false_positive", "false_positive_pct",
    ]

    def make_summary_row(scope: str, c: Counter) -> dict:
        n = sum(c.values())
        return {
            "scope": scope,
            "total": n,
            "good": c["good"],
            "good_pct": round(c["good"] / n * 100, 1),
            "slightly_off": c["slightly_off"],
            "slightly_off_pct": round(c["slightly_off"] / n * 100, 1),
            "completely_off": c["completely_off"],
            "completely_off_pct": round(c["completely_off"] / n * 100, 1),
            "false_positive": c["false_positive"],
            "false_positive_pct": round(c["false_positive"] / n * 100, 1),
        }

    summary_rows = [make_summary_row("OVERALL", counts)]
    for group, c in sorted(group_counts.items(), key=lambda x: -sum(x[1].values())):
        # false_positives are excluded from group_counts; include them as 0
        summary_rows.append(make_summary_row(group, c))

    summary_path = out_dir / "evaluation_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nWrote:")
    print(f"  {out_path}")
    print(f"  {summary_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate PetBERT predictions against keyword ground truth.")
    parser.add_argument(
        "--petbert-csv",
        default="ml/output/report/petbert_scan_predictions.csv",
        help="Path to petbert_scan_predictions.csv",
    )
    parser.add_argument(
        "--keyword-csv",
        default="ml/output/diagnoses/keyword_predictions.csv",
        help="Path to keyword_predictions.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="ml/output/evaluation",
        help="Directory to write evaluation.csv (default: ml/output/evaluation)",
    )
    args = parser.parse_args()
    evaluate(Path(args.petbert_csv), Path(args.keyword_csv), Path(args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
