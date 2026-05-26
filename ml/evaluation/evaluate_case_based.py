"""Case-level lenient scoring: one verdict per case, not per label.

Where evaluate.py emits one row per expected term (so a case with three labels
can produce three Good rows or partial coverage), this module rolls all of a
case's predictions up to a single verdict using the rule "any expected label
hit ⇒ Good." It's the optimistic counterpart to the per-label eval.

Verdict precedence per case (mirrors evaluate.py's six buckets):

  good           — at least one prediction's term matches an expected term
  slightly_off   — no exact-term hit, but at least one prediction's group
                   matches an expected group (or the uncommon-tier rule fires)
  completely_off — at least one non-Uncategorized prediction, but none hit
                   term or group
  false_negative — every prediction is Uncategorized (or no prediction row),
                   and the case has expected labels
  false_positive — non-cancer case (no expected labels) with at least one
                   non-Uncategorized prediction
  true_negative  — non-cancer case with all-Uncategorized predictions
                   (excluded from totals, matching evaluate.py)

Outputs (in --out-dir):
  case_based_evaluation.csv — one row per case (TNs excluded)
  case_based_summary.csv    — same 12-column shape as evaluation_summary.csv
                              so log_evaluation reads it unchanged
"""

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from evaluation.evaluate import load_csv, score_prediction
from evaluation.common import load_filter_ids, load_uncommon_groups


_PRECEDENCE = ("good", "slightly_off", "completely_off", "false_negative")


def _aggregate_case_verdict(per_pred_verdicts: list[str], is_cancer_case: bool) -> str:
    if not is_cancer_case:
        return "true_negative" if all(v == "true_negative" for v in per_pred_verdicts) else "false_positive"
    if not per_pred_verdicts:
        return "false_negative"
    for bucket in _PRECEDENCE:
        if bucket in per_pred_verdicts:
            return bucket
    # Cancer case where every prediction was true_negative shouldn't happen
    # (score_prediction only returns true_negative when matched_terms is empty),
    # but fall through safely to FN if it does.
    return "false_negative"


def evaluate_case_based(prediction_csv: Path, expectation_csv: Path, out_dir: Path,
                        cases_txt: str = "", uncommon_groups_file: str = "") -> None:
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

    case_terms: dict[str, set[str]] = defaultdict(set)
    case_groups: dict[str, set[str]] = defaultdict(set)
    for row in kw_rows:
        if row["matched_term"].strip():
            case_terms[row["case_id"]].add(row["matched_term"])
        if row["matched_group"].strip():
            case_groups[row["case_id"]].add(row["matched_group"])

    predictions_by_case: dict[str, list[dict]] = defaultdict(list)
    for row in pb_rows:
        predictions_by_case[row["case_id"]].append(row)

    # Cases that exist in either side. Cases with no prediction row but with
    # expected labels still count as FN; cases in kw_rows with empty matched_*
    # (non-cancer) appear via predictions_by_case if the model emitted anything.
    all_case_ids = set(case_terms) | set(predictions_by_case) | {r["case_id"] for r in kw_rows}

    out_rows: list[dict] = []
    true_negative_count = 0
    counts: Counter = Counter()
    group_counts: dict[str, Counter] = defaultdict(Counter)

    for cid in all_case_ids:
        preds = predictions_by_case.get(cid, [])
        exp_terms = case_terms.get(cid, set())
        exp_groups = case_groups.get(cid, set())
        is_cancer = bool(exp_terms)

        per_pred_verdicts = [
            score_prediction(p["predicted_term"], p["predicted_group"],
                             exp_terms, exp_groups, uncommon_groups)
            for p in preds
        ]
        verdict = _aggregate_case_verdict(per_pred_verdicts, is_cancer)

        if verdict == "true_negative":
            true_negative_count += 1
            continue

        counts[verdict] += 1
        # Per-group breakdown axis: first expected group (alphabetical) for cancer cases,
        # "(non-cancer)" for FPs.
        if is_cancer:
            axis = sorted(exp_groups)[0] if exp_groups else "(no expected group)"
        else:
            axis = "(non-cancer)"
        group_counts[axis][verdict] += 1

        predicted_terms = " | ".join(sorted({p["predicted_term"] for p in preds if p["predicted_term"]}))
        predicted_groups = " | ".join(sorted({p["predicted_group"] for p in preds if p["predicted_group"]}))
        out_rows.append({
            "case_id": cid,
            "verdict": verdict,
            "expected_terms": " | ".join(sorted(exp_terms)),
            "expected_groups": " | ".join(sorted(exp_groups)),
            "predicted_terms": predicted_terms,
            "predicted_groups": predicted_groups,
            "n_predictions": len(preds),
        })

    # ── Write per-case CSV ────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "case_based_evaluation.csv"
    fieldnames = ["case_id", "verdict", "expected_terms", "expected_groups",
                  "predicted_terms", "predicted_groups", "n_predictions"]
    sorted_rows = sorted(
        out_rows,
        key=lambda r: int(r["case_id"].split("-")[-1]) if r["case_id"].split("-")[-1].isdigit() else r["case_id"],
    )
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted_rows)

    # ── Console summary ───────────────────────────────────────────────────────
    total = sum(counts.values())
    print(f"\n=== Case-based ({total} cases, {true_negative_count} correct abstentions excluded) ===")
    for label, key in [
        ("Good", "good"),
        ("Slightly off", "slightly_off"),
        ("Completely off", "completely_off"),
        ("False positive", "false_positive"),
        ("False negative", "false_negative"),
    ]:
        n = counts[key]
        pct = (n / total * 100) if total else 0.0
        print(f"  {label:<22} {n:>5}  ({pct:.1f}%)")
    g_plus_s = counts["good"] + counts["slightly_off"]
    g_plus_s_pct = (g_plus_s / total * 100) if total else 0.0
    print(f"  {'G+S (lenient)':<22} {g_plus_s:>5}  ({g_plus_s_pct:.1f}%)")

    # Per-expected-group breakdown
    print("\n=== Per expected-group breakdown ===")
    header = f"{'Group':<45} {'Total':>5}  {'Good':>11}  {'Slight':>11}  {'Off':>11}"
    print(header)
    print("-" * len(header))
    for group, c in sorted(group_counts.items(), key=lambda x: -sum(x[1].values())):
        n = sum(c.values())
        if n == 0:
            continue
        print(
            f"{group:<45} {n:>5}  "
            f"{c['good']:>5} ({c['good']/n*100:>4.0f}%)  "
            f"{c['slightly_off']:>5} ({c['slightly_off']/n*100:>4.0f}%)  "
            f"{c['completely_off']:>5} ({c['completely_off']/n*100:>4.0f}%)"
        )

    # ── Write summary CSV (same shape as evaluation_summary.csv) ──────────────
    summary_fields = [
        "scope", "total",
        "good", "good_pct",
        "slightly_off", "slightly_off_pct",
        "completely_off", "completely_off_pct",
        "false_positive", "false_positive_pct",
        "false_negative", "false_negative_pct",
    ]

    def make_summary_row(scope: str, c: Counter) -> dict:
        n = sum(c.values())
        if n == 0:
            return {"scope": scope, "total": 0, **{k: 0 for k in summary_fields if k not in ("scope", "total")}}
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
            "false_negative": c["false_negative"],
            "false_negative_pct": round(c["false_negative"] / n * 100, 1),
        }

    summary_rows = [make_summary_row("OVERALL", counts)]
    for group, c in sorted(group_counts.items(), key=lambda x: -sum(x[1].values())):
        summary_rows.append(make_summary_row(group, c))

    summary_path = out_dir / "case_based_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nWrote:")
    print(f"  {out_path}")
    print(f"  {summary_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score predictions per-case (lenient: any expected hit ⇒ Good)."
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
    args = parser.parse_args()
    evaluate_case_based(
        Path(args.prediction_csv), Path(args.expectation_csv), Path(args.out_dir),
        cases_txt=args.test_cases, uncommon_groups_file=args.uncommon_groups,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
