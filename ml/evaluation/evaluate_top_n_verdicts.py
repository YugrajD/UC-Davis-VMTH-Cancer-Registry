"""Per expected-term verdict breakdown for the most frequent ICD-O labels.

For the top-N most common terms in the annotation (one CSV per N, e.g. 25, 50,
100), report the same Good / Slightly Off / Completely Off / FN buckets used in
``evaluate.py`` plus a per-term FP count (= pipeline predicted this term on a
case that did NOT have it expected). Each row also carries the term's
annotation frequency, precision, recall, and F1.

A MACRO row at the bottom averages the per-term percentages and the per-term
precision/recall/F1 with equal weight (no frequency weighting).

Output (in --out-dir): top_n_verdicts_{N}.csv, one per requested N.
"""

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Some Vet-ICD-O label names contain non-cp1252 Unicode (e.g. U+2010 HYPHEN).
# Windows console defaults to cp1252; reconfigure stdout to UTF-8 with a
# fallback so the per-N tables print cleanly. CSV output is already utf-8.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except (AttributeError, OSError):
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from evaluation.common import load_filter_ids, load_uncommon_groups, safe_div
from evaluation.evaluate import load_csv
from evaluation.evaluate_common_labels import _covers_group


_FIELDNAMES = [
    "rank", "label", "group", "frequency",
    "good", "good_pct",
    "slight", "slight_pct",
    "completely_off", "completely_off_pct",
    "fn", "fn_pct",
    "fp",
    "precision", "recall", "f1",
]


def _f1(p: float, r: float) -> float:
    return safe_div(2 * p * r, p + r)


def _round(x: float | str, ndigits: int = 1) -> float | str:
    if x == "" or x is None:
        return ""
    return round(float(x), ndigits)


def _per_term_row(rank: int, term: str, group: str, frequency: int,
                  good: int, slight: int, co: int, fn: int, fp: int) -> dict:
    precision = safe_div(good, good + fp)
    recall = safe_div(good, frequency)
    f1 = _f1(precision, recall)
    return {
        "rank": rank,
        "label": term,
        "group": group,
        "frequency": frequency,
        "good": good, "good_pct": round(safe_div(good, frequency) * 100, 1),
        "slight": slight, "slight_pct": round(safe_div(slight, frequency) * 100, 1),
        "completely_off": co, "completely_off_pct": round(safe_div(co, frequency) * 100, 1),
        "fn": fn, "fn_pct": round(safe_div(fn, frequency) * 100, 1),
        "fp": fp,
        "precision": "" if good + fp == 0 else round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _macro_row(per_term: list[dict]) -> dict:
    n = len(per_term)
    if n == 0:
        return {k: "" for k in _FIELDNAMES} | {"label": "MACRO"}

    def _mean(key: str) -> float:
        return sum(float(r[key]) for r in per_term if r[key] != "") / max(
            1, sum(1 for r in per_term if r[key] != "")
        )
    return {
        "rank": "", "label": "MACRO", "group": "", "frequency": "",
        "good": "", "good_pct": round(_mean("good_pct"), 1),
        "slight": "", "slight_pct": round(_mean("slight_pct"), 1),
        "completely_off": "", "completely_off_pct": round(_mean("completely_off_pct"), 1),
        "fn": "", "fn_pct": round(_mean("fn_pct"), 1),
        "fp": "",
        "precision": round(_mean("precision"), 4),
        "recall": round(_mean("recall"), 4),
        "f1": round(_mean("f1"), 4),
    }


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _print_table(top_n: int, rows: list[dict], macro: dict) -> None:
    print(f"\n=== Top-{top_n} verdict breakdown (per expected term) ===")
    header = (
        f"{'#':>3}  {'Label':<48}  {'Freq':>5}  "
        f"{'G%':>5}  {'S%':>5}  {'CO%':>5}  {'FN%':>5}  "
        f"{'FP':>5}  {'P':>6}  {'R':>6}  {'F1':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        prec = r["precision"] if r["precision"] != "" else "  n/a"
        prec = f"{float(prec):.3f}" if prec != "  n/a" else "  n/a"
        print(
            f"{r['rank']:>3}  {r['label'][:48]:<48}  {r['frequency']:>5}  "
            f"{r['good_pct']:>5.1f}  {r['slight_pct']:>5.1f}  "
            f"{r['completely_off_pct']:>5.1f}  {r['fn_pct']:>5.1f}  "
            f"{r['fp']:>5}  {prec:>6}  {r['recall']:>6.3f}  {r['f1']:>6.3f}"
        )
    mp = macro["precision"] if macro["precision"] != "" else "  n/a"
    mp = f"{float(mp):.3f}" if mp != "  n/a" else "  n/a"
    print(
        f"{'':>3}  {'MACRO (unweighted mean of per-term metrics)':<48}  {'':>5}  "
        f"{macro['good_pct']:>5.1f}  {macro['slight_pct']:>5.1f}  "
        f"{macro['completely_off_pct']:>5.1f}  {macro['fn_pct']:>5.1f}  "
        f"{'':>5}  {mp:>6}  {macro['recall']:>6.3f}  {macro['f1']:>6.3f}"
    )


def evaluate_top_n_verdicts(
    prediction_csv: Path,
    expectation_csv: Path,
    out_dir: Path,
    cases_txt: str = "",
    uncommon_groups_file: str = "",
    top_ns: tuple[int, ...] = (25, 50, 100),
) -> None:
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

    # Per-case expected term sets and canonical group lookup.
    case_terms: dict[str, set[str]] = defaultdict(set)
    term_to_group: dict[str, str] = {}
    for row in kw_rows:
        term = row["matched_term"].strip()
        if not term:
            continue
        case_terms[row["case_id"]].add(term)
        grp = row["matched_group"].strip()
        if grp and term not in term_to_group:
            term_to_group[term] = grp

    # Frequency = number of unique cases expecting each term (matches the per-case
    # G/S/CO/FN counts below — so good_pct = G / frequency reads as recall).
    term_counts: Counter = Counter()
    for terms in case_terms.values():
        for t in terms:
            term_counts[t] += 1

    # Per-case predicted (term, group) sets — dedupes K=2 tail-gate duplicates.
    # "Uncategorized" predictions are dropped so a case with only abstentions
    # has an empty prediction set (→ FN bucket for expected terms, no FP).
    case_preds: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in pb_rows:
        pt = row["predicted_term"]
        if pt and pt != "Uncategorized":
            case_preds[row["case_id"]].add((pt, row["predicted_group"]))

    # Cases that received at least one real prediction — used to separate
    # Completely-off from FN for expected-term cases.
    case_has_real_pred: dict[str, bool] = {cid: len(preds) > 0 for cid, preds in case_preds.items()}

    max_n = max(top_ns)
    top_terms_full = [t for t, _ in term_counts.most_common(max_n)]

    for n in top_ns:
        top_terms = top_terms_full[:n]
        top_set = set(top_terms)
        rows_out: list[dict] = []

        # Pre-aggregate predicted terms per case (set of term strings) for fast FP scan.
        case_pred_term_set: dict[str, set[str]] = {
            cid: {pt for pt, _ in preds} for cid, preds in case_preds.items()
        }
        case_pred_group_set: dict[str, set[str]] = {
            cid: {pg for _, pg in preds} for cid, preds in case_preds.items()
        }

        for rank, term in enumerate(top_terms, start=1):
            canonical_group = term_to_group.get(term, "")
            good = slight = co = fn = fp = 0

            for cid, terms in case_terms.items():
                if term in terms:
                    pred_terms = case_pred_term_set.get(cid, set())
                    pred_groups = case_pred_group_set.get(cid, set())
                    has_real_pred = case_has_real_pred.get(cid, False)
                    if term in pred_terms:
                        good += 1
                    elif canonical_group and any(
                        _covers_group(pg, canonical_group, uncommon_groups)
                        for pg in pred_groups
                    ):
                        slight += 1
                    elif has_real_pred:
                        co += 1
                    else:
                        fn += 1
                else:
                    # Term not expected for this case — counts toward FP if predicted.
                    if term in case_pred_term_set.get(cid, set()):
                        fp += 1

            # Also include cases that have no annotation row at all (non-cancer cases).
            # case_terms only contains cancer cases; non-cancer cases live solely in
            # case_preds. Their predictions still count as per-term FPs.
            for cid, pred_terms in case_pred_term_set.items():
                if cid in case_terms:
                    continue
                if term in pred_terms:
                    fp += 1

            frequency = term_counts[term]
            rows_out.append(_per_term_row(
                rank, term, canonical_group, frequency, good, slight, co, fn, fp,
            ))

        macro = _macro_row(rows_out)
        out_path = out_dir / f"top_n_verdicts_{n}.csv"
        _write_csv(rows_out + [macro], out_path)

        _print_table(n, rows_out, macro)
        print(f"\nWrote: {out_path}")
        del top_terms, top_set


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Top-N expected-term verdict breakdown (per-term G/S/CO/FN + per-term FP)."
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
    parser.add_argument(
        "--top-ns", default="25,50,100",
        help="Comma-separated Ns (default: 25,50,100).",
    )
    args = parser.parse_args()
    top_ns = tuple(int(x) for x in args.top_ns.split(",") if x.strip())
    evaluate_top_n_verdicts(
        Path(args.prediction_csv), Path(args.expectation_csv), Path(args.out_dir),
        cases_txt=args.test_cases, uncommon_groups_file=args.uncommon_groups,
        top_ns=top_ns,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
