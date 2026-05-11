"""Sweep per-LP thresholds for the Stage 3 LabelPresenceClassifier.

Reads the per-(case, label) evaluation CSV produced by
`run_evaluation.py --stage label-presence`, splits cases 50/50 by case-ID
hash (deterministic), selects per-LP threshold maximizing F1 on the sweep
half, then reports unbiased F1 on the eval half at the baseline threshold
vs the tuned threshold.

The 50/50 split keeps the LP-trained-on-train-cases constraint clean: both
halves are held-out from training, so neither half's score distribution is
saturated; threshold curves are realistic.

Usage:
  python ml/scripts/sweep_lp_thresholds.py
  python ml/scripts/sweep_lp_thresholds.py --baseline-threshold 0.5
  python ml/scripts/sweep_lp_thresholds.py --grid 0.05,0.95,0.01
"""

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

DEFAULT_CSV = (
    Path(config.OUTPUT_EVALUATION_DIR)
    / config.BEST_PREDICTIONS_SUBDIR
    / "label_presence_evaluation.csv"
)


def sweep_half(case_id: str) -> bool:
    """True → case goes to sweep half; False → eval half. Deterministic."""
    return int(hashlib.md5(case_id.encode()).hexdigest(), 16) % 2 == 0


def metrics_at(rows: list[tuple[float, int]], threshold: float) -> tuple[int, int, int, float, float, float]:
    """(tp, fp, fn, p, r, f1) for one LP's (prob, true) rows at a threshold."""
    tp = fp = fn = 0
    for prob, true in rows:
        pred = prob >= threshold
        if pred and true:
            tp += 1
        elif pred:
            fp += 1
        elif true:
            fn += 1
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return tp, fp, fn, p, r, f1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-csv", default=str(DEFAULT_CSV))
    parser.add_argument("--baseline-threshold", type=float, default=0.5)
    parser.add_argument(
        "--grid", default="0.05,0.95,0.01",
        help="start,stop,step for threshold sweep (inclusive of start; stop exclusive).",
    )
    parser.add_argument(
        "--out-csv", default="",
        help="Optional output CSV path for per-LP results.",
    )
    parser.add_argument(
        "--out-json", default=config.LABEL_PRESENCE_THRESHOLDS_JSON,
        help=(
            "Output JSON path for {group_name: threshold} — consumed by the "
            "production pipeline. Pass an empty string to skip writing."
        ),
    )
    args = parser.parse_args()

    g_start, g_stop, g_step = (float(x) for x in args.grid.split(","))
    thresholds = []
    t = g_start
    while t <= g_stop + 1e-9:
        thresholds.append(round(t, 4))
        t += g_step

    eval_path = Path(args.eval_csv)
    if not eval_path.exists():
        print(f"ERROR: {eval_path} not found. Run `run_evaluation.py --stage label-presence` first.")
        return 1

    print(f"Loading {eval_path}...")
    per_lp_sweep: dict[str, list[tuple[float, int]]] = defaultdict(list)
    per_lp_eval: dict[str, list[tuple[float, int]]] = defaultdict(list)
    sweep_cases: set[str] = set()
    eval_cases: set[str] = set()

    with open(eval_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = row["case_id"]
            lp = row["group"]
            prob = float(row["prob"])
            true = int(row["true_label"])
            if sweep_half(cid):
                per_lp_sweep[lp].append((prob, true))
                sweep_cases.add(cid)
            else:
                per_lp_eval[lp].append((prob, true))
                eval_cases.add(cid)

    print(
        f"  Split: {len(sweep_cases)} sweep cases / {len(eval_cases)} eval cases. "
        f"{len(per_lp_sweep)} LPs.\n"
    )

    results: list[dict] = []
    baseline_t = args.baseline_threshold
    for lp in sorted(per_lp_sweep.keys()):
        sweep_rows = per_lp_sweep[lp]
        eval_rows = per_lp_eval[lp]

        # Pick threshold on sweep half.
        best_t, best_f1 = baseline_t, -1.0
        for t in thresholds:
            _, _, _, _, _, f1 = metrics_at(sweep_rows, t)
            if f1 > best_f1:
                best_f1, best_t = f1, t

        # Measure on eval half.
        _, _, _, b_p, b_r, b_f1 = metrics_at(eval_rows, baseline_t)
        tp1, fp1, fn1, t_p, t_r, t_f1 = metrics_at(eval_rows, best_t)
        support = tp1 + fn1

        results.append({
            "lp": lp,
            "support_eval": support,
            "optimal_t": best_t,
            "baseline_p": b_p, "baseline_r": b_r, "baseline_f1": b_f1,
            "tuned_p": t_p, "tuned_r": t_r, "tuned_f1": t_f1,
            "delta_f1": t_f1 - b_f1,
            "tuned_tp": tp1, "tuned_fp": fp1, "tuned_fn": fn1,
        })

    # Print table sorted by support desc.
    results.sort(key=lambda r: -r["support_eval"])
    print(
        f"=== Per-LP threshold sweep (baseline t={baseline_t:.2f}, "
        f"grid={args.grid}) ===\n"
    )
    header = (
        f"{'LP':<54} {'sup':>5} {'t*':>5} | "
        f"{'P0':>5} {'R0':>5} {'F0':>5} | "
        f"{'P*':>5} {'R*':>5} {'F*':>5} | {'dF1':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['lp'][:54]:<54} {r['support_eval']:>5} {r['optimal_t']:>5.2f} | "
            f"{r['baseline_p']:>5.3f} {r['baseline_r']:>5.3f} {r['baseline_f1']:>5.3f} | "
            f"{r['tuned_p']:>5.3f} {r['tuned_r']:>5.3f} {r['tuned_f1']:>5.3f} | "
            f"{r['delta_f1']:>+6.3f}"
        )

    # Macro/micro on eval half across all LPs (baseline vs tuned).
    macro_b_f1 = sum(r["baseline_f1"] for r in results if r["support_eval"] > 0) / max(
        1, sum(1 for r in results if r["support_eval"] > 0)
    )
    macro_t_f1 = sum(r["tuned_f1"] for r in results if r["support_eval"] > 0) / max(
        1, sum(1 for r in results if r["support_eval"] > 0)
    )

    micro_tp_b = micro_fp_b = micro_fn_b = 0
    micro_tp_t = micro_fp_t = micro_fn_t = 0
    for lp, eval_rows in per_lp_eval.items():
        b_tp, b_fp, b_fn, *_ = metrics_at(eval_rows, baseline_t)
        micro_tp_b += b_tp
        micro_fp_b += b_fp
        micro_fn_b += b_fn
        t_star = next(r["optimal_t"] for r in results if r["lp"] == lp)
        t_tp, t_fp, t_fn, *_ = metrics_at(eval_rows, t_star)
        micro_tp_t += t_tp
        micro_fp_t += t_fp
        micro_fn_t += t_fn

    def _prf(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return p, r, f

    bp, br, bf1 = _prf(micro_tp_b, micro_fp_b, micro_fn_b)
    tp_, tr, tf1 = _prf(micro_tp_t, micro_fp_t, micro_fn_t)

    print(
        f"\n=== Overall on eval half (held-out from sweep) ===\n"
        f"  Macro F1: {macro_b_f1:.4f}  ->  {macro_t_f1:.4f}  "
        f"(d={macro_t_f1 - macro_b_f1:+.4f})\n"
        f"  Micro F1: {bf1:.4f}  ->  {tf1:.4f}  "
        f"(d={tf1 - bf1:+.4f})\n"
        f"  Micro P : {bp:.4f}  ->  {tp_:.4f}\n"
        f"  Micro R : {br:.4f}  ->  {tr:.4f}"
    )

    if args.out_csv:
        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"\nWrote per-LP results to {out}")

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        mapping = {r["lp"]: r["optimal_t"] for r in results}
        out_json.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote per-LP threshold JSON to {out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
