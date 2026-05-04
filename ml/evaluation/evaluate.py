"""Score cancer label predictions against verified annotations.

Each predicted label receives one of six verdicts:

  good           — predicted term exactly matches a verified label for this case
  slightly_off   — correct cancer group, wrong specific term
  completely_off — neither term nor group matches any verified label for this case
  false_positive — model made a positive prediction for a case with no verified labels
  false_negative — case has verified labels but model predicted "Uncategorized", or case has
                   no prediction row at all
  true_negative  — model correctly predicted "Uncategorized" for a non-cancer case
                   (excluded from evaluation.csv and from metrics)

Output files (written to --out-dir):
  evaluation.csv         — all predictions + verdicts, sorted by case_id
  evaluation_summary.csv — overall + per predicted-group counts and percentages
"""

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def load_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def score_prediction(predicted_term: str, predicted_group: str,
                     matched_terms: set[str], matched_groups: set[str],
                     uncommon_groups: frozenset[str] = frozenset()) -> str:
    """Score one petbert prediction against the case's keyword label sets."""
    if not matched_terms:
        # Model correctly abstained on a non-cancer case → true negative, not FP.
        if predicted_term == "Uncategorized":
            return "true_negative"
        return "false_positive"
    # Model abstained on a confirmed cancer case → false negative, not completely_off.
    if predicted_term == "Uncategorized":
        return "false_negative"
    if predicted_term in matched_terms:
        return "good"
    if predicted_group in matched_groups:
        return "slightly_off"
    # Predicting "Uncommon" for a case whose true group is genuinely uncommon counts
    # as slightly_off — the model got the right tier, just not the specific term.
    if predicted_group == "Uncommon" and uncommon_groups and matched_groups & uncommon_groups:
        return "slightly_off"
    return "completely_off"


def evaluate(prediction_csv: Path, expectation_csv: Path, out_dir: Path,
             cases_txt: str = "", uncommon_groups_file: str = "") -> None:
    pb_rows = load_csv(prediction_csv)
    kw_rows = load_csv(expectation_csv)

    uncommon_groups: frozenset[str] = frozenset()
    if uncommon_groups_file and Path(uncommon_groups_file).exists():
        with open(uncommon_groups_file, encoding="utf-8") as f:
            uncommon_groups = frozenset(line.strip() for line in f if line.strip())
        print(f"  Uncommon groups loaded: {len(uncommon_groups)} groups from {uncommon_groups_file}")

    if cases_txt and Path(cases_txt).exists():
        with open(cases_txt, encoding="utf-8") as f:
            filter_ids = {line.strip() for line in f if line.strip()}
        pb_rows = [r for r in pb_rows if r["case_id"] in filter_ids]
        kw_rows = [r for r in kw_rows if r["case_id"] in filter_ids]
        print(f"  Case filter active — evaluating {len(filter_ids)} cases.")

    # Build per-case label sets from keyword predictions
    case_terms: dict[str, set[str]] = defaultdict(set)
    case_groups: dict[str, set[str]] = defaultdict(set)
    term_to_group: dict[str, str] = {}
    for row in kw_rows:
        if row["matched_term"].strip():
            case_terms[row["case_id"]].add(row["matched_term"])
            if row["matched_group"].strip():
                term_to_group[row["matched_term"]] = row["matched_group"]
        if row["matched_group"].strip():
            case_groups[row["case_id"]].add(row["matched_group"])

    # All cases confirmed to have cancer ground truth
    cancer_case_ids: set[str] = set(case_terms.keys())

    # Score every petbert prediction row
    out_rows: list[dict] = []
    true_negative_rows: list[dict] = []
    uncategorized_cancer_case_ids: set[str] = set()
    for row in pb_rows:
        cid = row["case_id"]
        verdict = score_prediction(
            row["predicted_term"], row["predicted_group"],
            case_terms.get(cid, set()), case_groups.get(cid, set()),
            uncommon_groups,
        )
        scored_row = {
            **row,
            "verdict": verdict,
            "expected_term": " | ".join(sorted(case_terms.get(cid, set()))),
            "expected_group": " | ".join(sorted(case_groups.get(cid, set()))),
        }
        if verdict == "true_negative":
            true_negative_rows.append(scored_row)
        else:
            out_rows.append(scored_row)
            if row["predicted_term"] == "Uncategorized" and verdict == "false_negative":
                uncategorized_cancer_case_ids.add(cid)

    # Determine which expected terms are covered by good or slightly_off predictions.
    # A good prediction covers its exact term; a slightly_off prediction covers all
    # expected terms whose group matches the predicted group.
    covered_terms: dict[str, set[str]] = defaultdict(set)
    for row in out_rows:
        cid = row["case_id"]
        if row["verdict"] == "good":
            covered_terms[cid].add(row["predicted_term"])
        elif row["verdict"] == "slightly_off":
            pg = row["predicted_group"]
            for term in case_terms.get(cid, set()):
                if term_to_group.get(term) == pg:
                    covered_terms[cid].add(term)

    # ── Write output CSV ──────────────────────────────────────────────────────
    out_path = out_dir / "evaluation.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build fieldnames: drop 'method', interleave expected_* after predicted_*
    base_fields = [f for f in pb_rows[0].keys() if f not in ("method", "predicted_code")]
    fieldnames: list[str] = []
    for f in base_fields:
        fieldnames.append(f)
        if f == "predicted_term":
            fieldnames.append("expected_term")
        elif f == "predicted_group":
            fieldnames.append("expected_group")
    fieldnames.append("verdict")

    # One FN row per uncovered expected term. Cases that predicted "Uncategorized"
    # already have a FN prediction row and are excluded to avoid double-counting.
    blank_pred = {k: "" for k in pb_rows[0].keys()}
    fn_rows = [
        {
            **blank_pred,
            "case_id": cid,
            "verdict": "false_negative",
            "expected_term": term,
            "expected_group": term_to_group.get(term, ""),
        }
        for cid, terms in case_terms.items()
        if cid not in uncategorized_cancer_case_ids
        for term in terms
        if term not in covered_terms.get(cid, set())
    ]
    # true_negatives are excluded from evaluation.csv (correct abstentions add no signal)
    all_rows = sorted(out_rows + fn_rows, key=lambda r: int(r["case_id"].split("-")[-1]) if r["case_id"].split("-")[-1].isdigit() else r["case_id"])
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    # ── Console summary ───────────────────────────────────────────────────────
    counts = Counter(r["verdict"] for r in out_rows)
    # FN = Uncategorized prediction rows + per-term FN rows for uncovered expected terms.
    # True negatives (correct abstentions on non-cancer cases) are excluded from total.
    n_false_negatives = counts["false_negative"] + len(fn_rows)
    total = len(out_rows) + len(fn_rows)
    n_true_negatives = len(true_negative_rows)

    print(f"\n=== Overall ({total} prediction-cases, {n_true_negatives} correct abstentions excluded) ===")
    for label, key in [
        ("Good", "good"),
        ("Slightly off", "slightly_off"),
        ("Completely off", "completely_off"),
        ("False positive", "false_positive"),
        ("False negative", "false_negative"),
    ]:
        n = counts[key] if key != "false_negative" else n_false_negatives
        print(f"  {label:<22} {n:>5}  ({n / total * 100:.1f}%)")


    # Per-group breakdown (excluding verdicts with no meaningful predicted group)
    group_counts: dict[str, Counter] = defaultdict(Counter)
    for row in out_rows:
        if row["verdict"] in ("false_positive", "false_negative"):
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
        "false_negative", "false_negative_pct",
    ]

    def make_summary_row(scope: str, c: Counter, fn: int = 0) -> dict:
        # Exclude false_negative from counter sum; fn is the authoritative total FN count.
        n = sum(v for k, v in c.items() if k != "false_negative") + fn
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
            "false_negative": fn,
            "false_negative_pct": round(fn / n * 100, 1),
        }

    summary_rows = [make_summary_row("OVERALL", counts, fn=n_false_negatives)]
    for group, c in sorted(group_counts.items(), key=lambda x: -sum(x[1].values())):
        # false_positives and false_negatives are excluded from group_counts; include as 0
        summary_rows.append(make_summary_row(group, c))

    summary_path = out_dir / "evaluation_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nWrote:")
    print(f"  {out_path}  ({n_false_negatives} false negatives included)")
    print(f"  {summary_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score cancer label predictions against verified annotations."
    )
    parser.add_argument(
        "--prediction-csv",
        default=f"{config.OUTPUT_PRODUCTION_DIR}/binary/petbert_predictions.csv",
        help="Path to the predictions CSV to evaluate.",
    )
    parser.add_argument(
        "--expectation-csv",
        default=config.ANNOTATION_CSV,
        help="Path to the verified annotation CSV.",
    )
    parser.add_argument(
        "--out-dir",
        default=f"{config.OUTPUT_EVALUATION_DIR}/binary",
        help="Directory to write evaluation results.",
    )
    parser.add_argument(
        "--test-cases",
        default="",
        help="Path to test_cases.txt (one case_id per line). When provided, only held-out "
             "test cases are evaluated. Generate with create_split.py.",
    )
    parser.add_argument(
        "--uncommon-groups",
        default=config.UNCOMMON_GROUPS_TXT,
        help="Path to uncommon_groups.txt (one group name per line). Groups listed here "
             "are treated as correctly gated when the model predicts 'Uncommon', scoring "
             "as slightly_off rather than completely_off.",
    )
    args = parser.parse_args()
    evaluate(Path(args.prediction_csv), Path(args.expectation_csv), Path(args.out_dir),
             cases_txt=args.test_cases, uncommon_groups_file=args.uncommon_groups)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
