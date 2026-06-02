"""How often do we catch the most-common ICD-O labels.

Where evaluate_case_based.py rolls every label up to one verdict per case and
hides which specific terms the model is good or bad at, this report keys off
the *expected* label so missed labels stay visible.

For each of the top-N labels (by occurrence in the annotation CSV) we report:

  cases             — number of cases the annotator marked with this label
  correct           — of those, how many got this exact label predicted
                      (anywhere in the case's top-5 predictions); == good
  pct_correct       — correct / cases
  same_group        — of those, how many got at least the right cancer group
                      predicted (broader than 'correct'; matches the
                      slightly_off bucket in evaluate_case_based, including
                      the Uncommon-tier rule); == good + slightly_off
  pct_same_group    — same_group / cases

Plus a per-label verdict breakdown matching evaluate.py's semantics
(good + slightly_off + completely_off + false_negative = cases):

  good              — exact label was in the case's top-5 predictions
  slightly_off      — exact label missed, but some top-5 group covered it
  completely_off    — no top-5 prediction covered the label's group
  false_negative    — case was abstained on by Stage 1 (predicted Uncategorized)
  false_positive    — cases where the model predicted this label in top-5 but
                      the annotation does NOT contain it. Broader than
                      evaluate.py's FP definition (which only fires on
                      non-cancer cases).
  pct_false_positive — false_positive / cases (noise ratio relative to base
                       rate; blank when cases == 0)

Tail block: predicted-but-never-annotated labels are appended after the
annotation-frequency-sorted block, sorted by false_positive desc. Their
`cases == 0`, so all percentage columns are written as blank.

To get a complete list (not just top-20), pass `--top-n` above the taxonomy
size (e.g. `--top-n 1000`).

Output (in --out-dir):
  common_labels_evaluation.csv — one row per label
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
    # Also flag cases where Stage 1 abstained ("Uncategorized") — those count as
    # false_negative when an expected label is uncovered, matching evaluate.py.
    # Build a reverse index term -> set(case_ids) for the per-label FP count.
    case_pred_terms: dict[str, set[str]] = defaultdict(set)
    case_pred_groups: dict[str, set[str]] = defaultdict(set)
    case_is_uncat: set[str] = set()
    term_to_pred_cases: dict[str, set[str]] = defaultdict(set)
    term_to_pred_group: dict[str, str] = {}
    for row in pb_rows:
        cid = row["case_id"]
        if row["predicted_term"] == "Uncategorized":
            case_is_uncat.add(cid)
            continue
        if row["predicted_term"]:
            case_pred_terms[cid].add(row["predicted_term"])
            term_to_pred_cases[row["predicted_term"]].add(cid)
            if row["predicted_group"] and row["predicted_term"] not in term_to_pred_group:
                term_to_pred_group[row["predicted_term"]] = row["predicted_group"]
        if row["predicted_group"]:
            case_pred_groups[cid].add(row["predicted_group"])

    top_terms = [t for t, _ in term_counts.most_common(top_n)]

    rows_out: list[dict] = []
    for rank, term in enumerate(top_terms, start=1):
        canonical_group = term_to_group.get(term, "")
        cases_with_term = [cid for cid, terms in case_terms.items() if term in terms]
        cases = len(cases_with_term)
        good = slight = co = fn = 0
        for cid in cases_with_term:
            if term in case_pred_terms.get(cid, set()):
                good += 1
            elif any(_covers_group(pg, canonical_group, uncommon_groups)
                     for pg in case_pred_groups.get(cid, set())):
                slight += 1
            elif cid in case_is_uncat:
                fn += 1
            else:
                co += 1
        correct = good
        same_group = good + slight
        false_positive = len(term_to_pred_cases.get(term, set()) - set(cases_with_term))
        pct = lambda n: round((n / cases * 100) if cases else 0.0, 1)
        rows_out.append({
            "rank": rank,
            "label": term,
            "group": canonical_group,
            "cases": cases,
            "correct": correct,
            "pct_correct": pct(correct),
            "same_group": same_group,
            "pct_same_group": pct(same_group),
            "good": good,
            "pct_good": pct(good),
            "slightly_off": slight,
            "pct_slightly_off": pct(slight),
            "completely_off": co,
            "pct_completely_off": pct(co),
            "false_negative": fn,
            "pct_false_negative": pct(fn),
            "false_positive": false_positive,
            "pct_false_positive": pct(false_positive),
        })

    # Predicted-but-never-annotated terms — emit a tail block sorted by FP desc.
    # cases=0 here, so percentages are undefined; written as blank.
    annotated = set(top_terms)
    extra_terms = sorted(
        (t for t in term_to_pred_cases.keys() if t not in annotated),
        key=lambda t: -len(term_to_pred_cases[t]),
    )
    next_rank = len(rows_out) + 1
    for offset, term in enumerate(extra_terms):
        fp = len(term_to_pred_cases[term])
        rows_out.append({
            "rank": next_rank + offset,
            "label": term,
            "group": term_to_pred_group.get(term, ""),
            "cases": 0,
            "correct": 0, "pct_correct": "",
            "same_group": 0, "pct_same_group": "",
            "good": 0, "pct_good": "",
            "slightly_off": 0, "pct_slightly_off": "",
            "completely_off": 0, "pct_completely_off": "",
            "false_negative": 0, "pct_false_negative": "",
            "false_positive": fp, "pct_false_positive": "",
        })

    # ── Write CSV ─────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "common_labels_evaluation.csv"
    fieldnames = ["rank", "label", "group", "cases",
                  "correct", "pct_correct", "same_group", "pct_same_group",
                  "good", "pct_good",
                  "slightly_off", "pct_slightly_off",
                  "completely_off", "pct_completely_off",
                  "false_negative", "pct_false_negative",
                  "false_positive", "pct_false_positive"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    # ── Console table ─────────────────────────────────────────────────────────
    n_cases = len(filter_ids) if filter_ids is not None else len({r["case_id"] for r in kw_rows})
    print(f"\n=== Common ICD labels (top {top_n} by annotation frequency) — N={n_cases} cases ===")
    header = (
        f"{'Rank':>4}  {'Label':<50}  {'Group':<40}  "
        f"{'Cases':>5}  {'G%':>6}  {'S%':>6}  {'CO%':>6}  {'FN%':>6}  "
        f"{'FP':>5}  {'FP%':>6}"
    )
    print(header)
    print("-" * len(header))
    def _fmt_pct(v) -> str:
        return f"{v:>5.1f}%" if isinstance(v, (int, float)) else f"{'  -  ':>6}"
    for r in rows_out:
        print(
            f"{r['rank']:>4}  "
            f"{r['label'][:50]:<50}  "
            f"{r['group'][:40]:<40}  "
            f"{r['cases']:>5}  "
            f"{_fmt_pct(r['pct_good'])}  "
            f"{_fmt_pct(r['pct_slightly_off'])}  "
            f"{_fmt_pct(r['pct_completely_off'])}  "
            f"{_fmt_pct(r['pct_false_negative'])}  "
            f"{r['false_positive']:>5}  "
            f"{_fmt_pct(r['pct_false_positive'])}"
        )

    print(f"\nWrote: {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Per-label recall report for the most common ICD-O labels."
    )
    parser.add_argument(
        "--prediction-csv",
        default=f"{config.OUTPUT_PRODUCTION_DIR}/petbert_predictions.csv",
    )
    parser.add_argument("--expectation-csv", default=config.ANNOTATION_CSV)
    parser.add_argument(
        "--out-dir",
        default=config.OUTPUT_EVALUATION_DIR,
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
