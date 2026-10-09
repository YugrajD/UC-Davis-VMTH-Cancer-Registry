"""Thin entry point: promote a retrained candidate (gold rule), or adopt an architectural change with a reason.

All the work lives in ``generations.promote``. Prints counts and percentages only.

Retraining (same code, new weights; judged on gold-eval only):
  ml/.venv/Scripts/python.exe ml/scripts/promote.py --candidate-predictions PATH --incumbent-predictions PATH
  ml/.venv/Scripts/python.exe ml/scripts/promote.py ... --apply [--description short-desc] [--publish]

--publish (with --apply) then publishes the promoted generation to S3 (sync.py publish-model; needs AWS credentials).

Architectural change (new code; no gold rule, no triggers; the reason is a file under ml/documentation/history/):
  ml/.venv/Scripts/python.exe ml/scripts/promote.py --adopt --reason ml/documentation/history/decisions/NNNN-title.md
      [--description short-desc] [--publish]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from evaluation.gold_eval import GoldEvalError
from evaluation.silver_eval import SilverEvalError
from generations import promote
from generations.guards import GuardViolation
from generations.manifest import ManifestError
from report_mapping.generation import GenerationError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidate-predictions", help="Predictions CSV made by candidate/ (not with --adopt).")
    parser.add_argument("--incumbent-predictions", help="Predictions CSV made by current/ (not with --adopt).")
    parser.add_argument("--split", default=None, help="split_id whose test side holds gold-eval (default: config).")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--apply", action="store_true",
                        help="Promote a winning candidate (archive current/ first) or delete a losing one.")
    parser.add_argument("--adopt", action="store_true",
                        help="Adopt candidate/ as an architectural change (archive current/ first); needs --reason.")
    parser.add_argument("--reason", default=None,
                        help="With --adopt: an existing file under ml/documentation/history/ explaining the change.")
    parser.add_argument("--description", default=None,
                        help="Archive folder suffix (default: the incumbent's generation_id).")
    parser.add_argument("--publish", action="store_true",
                        help="After a promotion or adoption, publish the new current/ to S3 (requires --apply or --adopt).")
    return parser


def _publish() -> int:
    from s3sync import hooks  # only when --publish is given: no boto3 or credentials otherwise

    return hooks.publish_current()


def _pp(x: float) -> str:
    return f"{100 * x:+.2f} pp"


def _adopt(args: argparse.Namespace) -> int:
    try:
        outcome = promote.adopt_architecture(config.REPORT_MAPPING_CANDIDATE_DIR, config.REPORT_MAPPING_CURRENT_DIR,
                                             args.reason, args.description)
    except (promote.PromotionError, ManifestError, GenerationError) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 1
    print(f"Adopted {outcome['generation_id']} (reason {args.reason}); "
          f"incumbent archived to {outcome['archive'] or 'nowhere (no current/)'}")
    return _publish() if args.publish else 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.adopt and args.apply:
        parser.error("--adopt and --apply are mutually exclusive")
    if args.adopt and not args.reason:
        parser.error("--adopt requires --reason PATH")
    if args.reason and not args.adopt:
        parser.error("--reason is only for --adopt")
    if args.publish and not (args.apply or args.adopt):
        parser.error("--publish requires --apply or --adopt")
    if args.adopt:
        return _adopt(args)
    if not (args.candidate_predictions and args.incumbent_predictions):
        parser.error("--candidate-predictions and --incumbent-predictions are required (unless --adopt)")
    try:
        result = promote.recommend(args.candidate_predictions, args.incumbent_predictions, split_id=args.split,
                                   n_boot=args.n_boot, seed=args.seed)
    except (promote.PromotionError, GoldEvalError, SilverEvalError, GuardViolation, ManifestError,
            GenerationError) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 1

    c = result["comparison"]
    print(f"Challenger {result['challenger_id']} vs incumbent {result['incumbent_id']} on gold-eval of "
          f"{result['split_id']} ({c['cases']} cases; {c['challenger_codes']} / {c['incumbent_codes']} code rows; "
          f"{c['challenger_unpredicted']} / {c['incumbent_unpredicted']} cases with no prediction row)")
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
            if args.publish:
                return _publish()
        else:
            print(f"Deleted losing candidate {outcome['generation_id']}")
            if args.publish:
                print("(--publish ignored: nothing promoted)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
