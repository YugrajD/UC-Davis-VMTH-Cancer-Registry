"""Thin entry point: the promotion recommendation (default) or carrying it out (``--apply``).

All the work lives in ``generations.promote``. Prints counts and percentages only.

  ml/.venv/bin/python ml/next/scripts/promote.py --candidate-predictions PATH --incumbent-predictions PATH
  ml/.venv/bin/python ml/next/scripts/promote.py ... --apply [--description short-desc]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.gold_eval import GoldEvalError
from evaluation.silver_eval import SilverEvalError
from generations import promote
from generations.guards import GuardViolation
from generations.manifest import ManifestError
from report_mapping.model.generation import GenerationError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidate-predictions", required=True, help="Predictions CSV made by candidate/.")
    parser.add_argument("--incumbent-predictions", required=True, help="Predictions CSV made by current/.")
    parser.add_argument("--split", default=None, help="split_id whose test side holds gold-eval (default: config).")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--apply", action="store_true",
                        help="Promote a winning candidate (archive current/ first) or delete a losing one.")
    parser.add_argument("--description", default=None,
                        help="Archive folder suffix (default: the incumbent's generation_id).")
    return parser


def _pp(x: float) -> str:
    return f"{100 * x:+.2f} pp"


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = promote.recommend(args.candidate_predictions, args.incumbent_predictions, split_id=args.split,
                                   n_boot=args.n_boot, seed=args.seed)
    except (promote.PromotionError, GoldEvalError, SilverEvalError, GuardViolation, ManifestError,
            GenerationError) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 1

    c = result["comparison"]
    print(f"Challenger {result['challenger_id']} vs incumbent {result['incumbent_id']} on gold-eval of "
          f"{result['split_id']} ({c['cases']} cases; {c['challenger_codes']} / {c['incumbent_codes']} code rows)")
    for name, label in (("good", "good (primary)"), ("gs", "G+S")):
        print(f"  {label:<15} {100 * c[f'challenger_{name}']:.2f}% vs {100 * c[f'incumbent_{name}']:.2f}%  "
              f"diff {_pp(c[f'{name}_diff'])} [{_pp(c[f'{name}_lo'])}, {_pp(c[f'{name}_hi'])}]")
    print(f"  rule: lower bound of good diff >= {_pp(result['margin'])}: {c['good_lo'] >= result['margin']}")
    for trigger in result["triggers"]:
        numbers = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in trigger.numbers.items())
        print(f"  trigger {trigger.name:<20} {'MET' if trigger.met else 'not met'}  ({numbers})")
    print(f"Recommendation: {'PROMOTE' if result['promote'] else 'DO NOT PROMOTE (candidate would be deleted)'}")

    if args.apply:
        outcome = promote.apply(result, args.description)
        if outcome["action"] == "promoted":
            print(f"Promoted {outcome['generation_id']}; incumbent archived to {outcome['archive']}")
        else:
            print(f"Deleted losing candidate {outcome['generation_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
