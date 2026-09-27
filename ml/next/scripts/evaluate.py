"""Thin entry point: silver-eval, gold-eval (four results) and Tier-3 audit rates.

Replaces ``ml/scripts/run_evaluation.py``. Prints counts and percentages only.

  ml/.venv/bin/python ml/next/scripts/evaluate.py silver --predictions ml/output/predictions/gen-0-legacy_predictions.csv \\
      --labels silver-0-legacy --split three-way-v1 --partition test
  ml/.venv/bin/python ml/next/scripts/evaluate.py silver ... --split legacy-80-20 --partition test --half eval
  ml/.venv/bin/python ml/next/scripts/evaluate.py gold --predictions PATH --silver silver-0-legacy --split three-way-v1
  ml/.venv/bin/python ml/next/scripts/evaluate.py audit-rates
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from evaluation import audit_rates, gold_eval, silver_eval, verdicts
from generations.guards import GuardViolation


def _pct(x: float) -> str:
    return "   —" if pd.isna(x) else f"{100 * x:5.1f}%"


def _ci(row, name: str) -> str:
    return f"{_pct(row[name])} [{_pct(row[f'{name}_lo'])}, {_pct(row[f'{name}_hi'])}]"


def _cmd_silver(args: argparse.Namespace) -> int:
    result = silver_eval.run(args.predictions, args.labels, args.split, args.partition,
                             generation=args.generation, half=args.half, history_csv=args.history)
    s = result["summary"]
    print(f"Silver-eval: {result['generation_id']} vs {args.labels} on {args.split} {result['partition']} "
          f"({result['n_cases']} cases, {s['total']} code rows)")
    print(f"  G+S {s['good_plus_slight']} ({100 * s['good_plus_slight_share']:.2f}%)")
    for verdict in verdicts.VERDICTS:
        print(f"  {verdict:<15} {s[verdict]:>6}  ({100 * s[f'{verdict}_share']:.2f}%)")
    print("\nPer expected group:")
    for group, row in result["by_group"].iterrows():
        print(f"  {group:<45} n={row['n']:>5}  good {_pct(row['good_share'])}  G+S {_pct(row['gs_share'])}")
    print(f"\nRecorded to {result['history_csv']}")
    return 0


def _print_cells(title: str, table: pd.DataFrame, key: str) -> None:
    print(f"\n{title}")
    for _, row in table.iterrows():
        label = " × ".join(str(row[k]) for k in ([key] if isinstance(key, str) else key))
        print(f"  {label:<34} cases={row['cases']:>4} codes={row['codes']:>4} tn={row['tn_cases']:>4}  "
              f"good {_ci(row, 'good')} boot [{_pct(row['good_boot_lo'])}, {_pct(row['good_boot_hi'])}]  "
              f"G+S {_ci(row, 'gs')}  FN {_pct(row['false_negative'])}  FP {_pct(row['false_positive'])}")


def _cmd_gold(args: argparse.Namespace) -> int:
    result = gold_eval.run(args.predictions, args.silver, args.split, generation=args.generation,
                           n_boot=args.n_boot, seed=args.seed)
    for note in result["notes"]:
        print(f"NOTE: {note}")
    print(f"Gold-eval on {args.split}: " + ", ".join(f"{k}={v}" for k, v in result["counts"].items()))
    _print_cells("1. Silver vs gold, by decision_stage", result["silver_vs_gold"], "stage")
    _print_cells("2. Bronze vs gold, by confidence band", result["bronze_vs_gold"], "band")
    print("\n3. Bronze vs silver on the same cases (pre-gold-correction)")
    for _, row in result["bronze_vs_silver"].iterrows():
        if row["ruler"] in ("gold", "silver"):
            print(f"  vs {row['ruler']:<7} codes={int(row['codes']):>4}  good {_ci(row, 'good')}  G+S {_ci(row, 'gs')}")
        else:
            name = "good" if pd.notna(row.get("good")) else "gs"
            print(f"  {row['ruler']:<22} {100 * row[name]:+.1f} pp "
                  f"[{100 * row[f'{name}_boot_lo']:+.1f}, {100 * row[f'{name}_boot_hi']:+.1f}]")
    print("\n4. Who was right on disagreements, by decision_stage × bronze band")
    for _, row in result["disagreements"].iterrows():
        print(f"  {row['stage'] + ' × ' + row['band']:<44} cases={row['cases']:>4} disagree={row['disagreements']:>4}  "
              f"silver {_ci(row, 'silver_right')}  bronze {_ci(row, 'bronze_right')}  "
              f"neither {_pct(row['neither_right'])}")
    print("\nCause pass on misses (method error = input supports the gold code)")
    for _, row in result["cause_split"].iterrows():
        print(f"  {row['method']:<7} misses={row['misses']:>4} reviewed={row['reviewed']:>4} "
              f"method_error={row['method_error']} input_gap={row['input_gap']}  "
              f"method-error share {_pct(row['method_error_share'])} "
              f"[{_pct(row['method_error_lo'])}, {_pct(row['method_error_hi'])}]")
    rep = result["representativeness"]
    print(f"\nRepresentativeness ({'PASS' if rep['pass'] else 'not yet'}; major groups from {result['group_basis']})")
    ci = rep["ci_half_width"]
    print(f"  CI half-width {_pct(ci['value'])} <= {_pct(ci['threshold'])}: {ci['pass']}")
    groups = rep["major_groups"]
    print(f"  every major group >= {gold_eval.MIN_GROUP_CODES} gold codes: {groups['pass']}")
    for group, row in groups["table"].iterrows():
        print(f"    {group:<45} share {_pct(row['share'])}  gold {row['gold_codes']:>3}  test {row['test_codes']:>4}")
    if groups["unreachable_from_test"]:
        print(f"  cannot reach {gold_eval.MIN_GROUP_CODES} codes from test alone: {groups['unreachable_from_test']}")
    print(f"  random-slice cases {rep['random_slice']['cases']}: {rep['random_slice']['pass']}")
    if args.misses_out:
        result["misses"].to_csv(args.misses_out, index=False, encoding="utf-8", lineterminator="\n")
        print(f"\nWrote {len(result['misses'])} miss(es) to {args.misses_out} (input to audit.py cause-sheet)")
    return 0


def _print_rate(label: str, r: dict) -> None:
    print(f"  {label:<40} sample {r['hits']}/{r['n']} {_pct(r['sample_share'])}   "
          f"weighted {_pct(r['weighted_share'])} [{_pct(r['lo'])}, {_pct(r['hi'])}]   represents {r['represents']:,.0f}")


def _cmd_audit_rates(args: argparse.Namespace) -> int:
    result = audit_rates.audit_rates()
    print(f"Tier-3 audit rates — {result['rows']} reviewed row(s), {result['cases']} case(s)")
    for _, row in result["by_stratum"].iterrows():
        _print_rate(f"{row['stratum']} {row['verdict']}{' ←' if row['headline'] else ''}", row)
    print()
    _print_rate("false-negative reservoir (wrong)", result["reservoir"])
    _print_rate("declined LLM rows that are missed cancers", result["declined_missed_cancer"])
    _print_rate("declined LLM rows judged uncertain", result["declined_uncertain"])
    _print_rate("tier2_fuzzy error rate (wrong)", result["tier2_fuzzy_error"])
    _print_rate("tier2_fuzzy judged uncertain", result["tier2_fuzzy_uncertain"])
    for warning in result["warnings"]:
        print(f"WARNING: {warning}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("silver", help="Bronze vs a labels table on one partition; appends to the history CSV.")
    p.add_argument("--predictions", required=True)
    p.add_argument("--labels", required=True, help="A silver_id, or a labels CSV path.")
    p.add_argument("--split", required=True)
    p.add_argument("--partition", required=True, choices=["test", "calibration"])
    p.add_argument("--half", choices=silver_eval.HALVES, default=None,
                   help="Keep only the md5 eval/sweep half of the partition.")
    p.add_argument("--generation", default="current", help='The generation that made the predictions.')
    p.add_argument("--history", default=None, help="History CSV (default: config.SILVER_EVAL_HISTORY_CSV).")
    p.set_defaults(func=_cmd_silver)

    p = sub.add_parser("gold", help="The four results on gold-eval, with CIs and representativeness.")
    p.add_argument("--predictions", required=True)
    p.add_argument("--silver", required=True, help="Silver generation id.")
    p.add_argument("--split", required=True)
    p.add_argument("--generation", default="current", help='The generation that made the predictions.')
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--misses-out", default=None, help="Write the misses table for audit.py cause-sheet.")
    p.set_defaults(func=_cmd_gold)

    p = sub.add_parser("audit-rates", help="Weighted Tier-3 audit rates per stratum.")
    p.set_defaults(func=_cmd_audit_rates)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return args.func(args)
    except (silver_eval.SilverEvalError, gold_eval.GoldEvalError, audit_rates.AuditRatesError,
            GuardViolation) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
