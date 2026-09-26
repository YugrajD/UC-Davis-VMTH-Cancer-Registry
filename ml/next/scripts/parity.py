"""Parity harness: compare the rewrite's outputs with the frozen legacy reference.

Temporary — deleted at cutover. Compares files only; it never trains or embeds.
Prints counts, percentages, case IDs and group names only. Exit code 1 on FAIL.

  parity.py freeze                                   build the reference pack (once, Mac, old code)
  parity.py l1 [--scope eval_half|test|both]         new scorer vs legacy verdict tables
  parity.py l2 --predictions NEW.csv                 same checkpoints + embeddings: identical rows
  parity.py l2 --reembedded --predictions NEW.csv --embeddings NEW.npz [--embeddings-key K]
               [--embeddings-ids IDS.txt]            re-embedded from report.csv
  parity.py l3 --predictions SEED1.csv SEED2.csv SEED3.csv   retrained heads (gated)
  parity.py l4 --predictions RUN.csv [...]           cold start (report only)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from parity import levels, reference


def _print(report: levels.Report) -> int:
    print("\n".join(report.lines))
    return 1 if report.passed is False else 0


def _l1(args) -> int:
    reference.load_manifest()
    scopes = ["eval_half", "test"] if args.scope == "both" else [args.scope]
    predictions = reference.reference_predictions()
    status = 0
    for scope in scopes:
        print(f"[{scope}]")
        status |= _print(levels.l1(reference.score_scope(predictions, scope), reference.reference_verdicts(scope)))
    return status


def _l2(args) -> int:
    manifest = reference.load_manifest()
    new_preds = reference.read_table(args.predictions)
    ref_preds = reference.reference_predictions()
    if not args.reembedded:
        return _print(levels.l2(new_preds, ref_preds, reference.reference_uncommon_groups()))
    return _print(levels.l2_reembedded(
        reference.load_embeddings(args.embeddings, args.embeddings_key, args.embeddings_ids),
        reference.reference_embeddings(manifest),
        new_preds, ref_preds,
        reference.score_scope(new_preds, "eval_half"), reference.reference_verdicts("eval_half"),
        reference.reference_uncommon_groups(),
    ))


def _seeds(args, gate: bool, name: str) -> int:
    reference.load_manifest()
    tables = [reference.score_scope(reference.read_table(p), "eval_half") for p in args.predictions]
    return _print(levels.l3(tables, reference.reference_verdicts("eval_half"), gate=gate, name=name))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="level", required=True)
    sub.add_parser("freeze", help="Build the legacy reference pack with the old evaluator.")
    p1 = sub.add_parser("l1", help="Scorer: new verdicts vs legacy, row for row.")
    p1.add_argument("--scope", choices=["eval_half", "test", "both"], default="both")
    p2 = sub.add_parser("l2", help="Inference: new predictions CSV vs legacy.")
    p2.add_argument("--predictions", required=True)
    p2.add_argument("--reembedded", action="store_true",
                    help="Embeddings were recomputed from report.csv: cosine / row-share / G+S checks.")
    p2.add_argument("--embeddings", help="New embeddings .npz (or .npy with --embeddings-ids).")
    p2.add_argument("--embeddings-key", help="Matrix key in the .npz (default: col_concat_3, else embeddings).")
    p2.add_argument("--embeddings-ids", help="Case-id list (one per line) aligned with an .npy matrix.")
    for name, text in (("l3", "Retrained heads: N seed runs (gated)."), ("l4", "Cold start (report only).")):
        p = sub.add_parser(name, help=text)
        p.add_argument("--predictions", nargs="+", required=True)
    args = parser.parse_args()

    if args.level == "freeze":
        return 0 if reference.freeze() else 1
    if args.level == "l1":
        return _l1(args)
    if args.level == "l2":
        if args.reembedded and not args.embeddings:
            parser.error("--reembedded needs --embeddings")
        return _l2(args)
    return _seeds(args, gate=args.level == "l3", name=args.level.upper())


if __name__ == "__main__":
    raise SystemExit(main())
