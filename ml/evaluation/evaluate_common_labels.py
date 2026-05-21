"""How often do we catch the most-common ICD-O labels.

Where evaluate_case_based.py rolls every label up to one verdict per case and
hides which specific terms the model is good or bad at, this report keys off
the *expected* label so missed labels stay visible.

For each of the top-N labels (by occurrence in the annotation CSV) we report:

  cases          — number of cases the annotator marked with this label
  correct        — of those, how many got this exact label predicted
                   (anywhere in the case's top-5 predictions)
  pct_correct    — correct / cases
  same_group     — of those, how many got at least the right cancer group
                   predicted (broader than 'correct'; matches the
                   slightly_off bucket in evaluate_case_based, including the
                   Uncommon-tier rule)
  pct_same_group — same_group / cases

Output (in --out-dir):
  common_labels_evaluation.csv — one row per label, sorted by `cases` desc
"""

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from evaluation.evaluate import load_csv
from evaluation.common import load_filter_ids, load_uncommon_groups


def _covers_group(predicted_group: str, expected_group: str,
                  uncommon_groups: frozenset[str]) -> bool:
    """Mirror the slightly_off rule from evaluate.score_prediction()."""
    if predicted_group == expected_group:
        return True
    if uncommon_groups and expected_group in uncommon_groups and \
            (predicted_group == "Uncommon" or predicted_group in uncommon_groups):
        return True
    return False


def evaluate_common_labels(prediction_csv: Path, expectation_csv: Path, out_dir: Path,
                           cases_txt: str = "", uncommon_groups_file: str = "",
                           top_n: int = 20) -> None:
    pb_rows = load_csv(prediction_csv)
    kw_rows = load_csv(expectation_csv)

    uncommon_groups = load_uncommon_groups(uncommon_groups_file) if uncommon_groups_file else frozenset()
    if uncommon_groups:
        print(f"  Uncommon groups loaded: {len(uncommon_groups)} groups from {uncommon_groups_file}")

    filter_ids = load_filter_ids(cases_txt)
    if filter_ids is not None:
        pb_rows = [r for r in pb_rows if r["case_id"] in filter_ids]
        kw_rows = [r for r in kw_rows if r["case_id"] in filter_ids]
        print(f"  Case filter active — evaluating {len(filter_ids)} cases.")

    # Per-case expected term sets and the canonical group for each term.
    case_terms: dict[str, set[str]] = defaultdict(set)
    term_to_group: dict[str, str] = {}
    term_counts: Counter = Counter()
    for row in kw_rows:
        term = row["matched_term"].strip()
        if not term:
            continue
        case_terms[row["case_id"]].add(term)
        term_counts[term] += 1
        if row["matched_group"].strip() and term not in term_to_group:
            term_to_group[term] = row["matched_group"].strip()

    # Per-case predicted term and group sets (top-5 already in the CSV).
    case_pred_terms: dict[str, set[str]] = defaultdict(set)
    case_pred_groups: dict[str, set[str]] = defaultdict(set)
    for row in pb_rows:
        cid = row["case_id"]
        if row["predicted_term"]:
            case_pred_terms[cid].add(row["predicted_term"])
        if row["predicted_group"]:
            case_pred_groups[cid].add(row["predicted_group"])

    top_terms = [t for t, _ in term_counts.most_common(top_n)]

    rows_out: list[dict] = []
    for rank, term in enumerate(top_terms, start=1):
        canonical_group = term_to_group.get(term, "")
        cases_with_term = [cid for cid, terms in case_terms.items() if term in terms]
        cases = len(cases_with_term)
        correct = sum(1 for cid in cases_with_term if term in case_pred_terms.get(cid, set()))
        same_group = sum(
            1 for cid in cases_with_term
            if any(_covers_group(pg, canonical_group, uncommon_groups)
                   for pg in case_pred_groups.get(cid, set()))
        )
        pct_correct = (correct / cases * 100) if cases else 0.0
        pct_same_group = (same_group / cases * 100) if cases else 0.0
        rows_out.append({
            "rank": rank,
            "label": term,
            "group": canonical_group,
            "cases": cases,
            "correct": correct,
            "pct_correct": round(pct_correct, 1),
            "same_group": same_group,
            "pct_same_group": round(pct_same_group, 1),
        })

    # ── Write CSV ─────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "common_labels_evaluation.csv"
    fieldnames = ["rank", "label", "group", "cases",
                  "correct", "pct_correct", "same_group", "pct_same_group"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    # ── Console table ─────────────────────────────────────────────────────────
    n_cases = len(filter_ids) if filter_ids is not None else len({r["case_id"] for r in kw_rows})
    print(f"\n=== Common ICD labels (top {top_n} by annotation frequency) — N={n_cases} cases ===")
    header = (
        f"{'Rank':>4}  {'Label':<50}  {'Group':<40}  "
        f"{'Cases':>5}  {'Correct':>7}  {'% Correct':>9}  "
        f"{'Same Group':>10}  {'% Same Group':>12}"
    )
    print(header)
    print("-" * len(header))
    for r in rows_out:
        print(
            f"{r['rank']:>4}  "
            f"{r['label'][:50]:<50}  "
            f"{r['group'][:40]:<40}  "
            f"{r['cases']:>5}  "
            f"{r['correct']:>7}  "
            f"{r['pct_correct']:>8.1f}%  "
            f"{r['same_group']:>10}  "
            f"{r['pct_same_group']:>11.1f}%"
        )

    print(f"\nWrote: {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Per-label recall report for the most common ICD-O labels."
    )
    parser.add_argument(
        "--prediction-csv",
        default=f"{config.OUTPUT_PRODUCTION_DIR}/{config.BEST_PREDICTIONS_SUBDIR}/petbert_predictions.csv",
    )
    parser.add_argument("--expectation-csv", default=config.ANNOTATION_CSV)
    parser.add_argument(
        "--out-dir",
        default=f"{config.OUTPUT_EVALUATION_DIR}/{config.BEST_PREDICTIONS_SUBDIR}",
    )
    parser.add_argument("--test-cases", default="")
    parser.add_argument("--uncommon-groups", default=config.UNCOMMON_GROUPS_TXT)
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()
    evaluate_common_labels(
        Path(args.prediction_csv), Path(args.expectation_csv), Path(args.out_dir),
        cases_txt=args.test_cases, uncommon_groups_file=args.uncommon_groups,
        top_n=args.top_n,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
