"""Thin entry point for the retrieve-and-rerank Phase 1 steps (ml/bronze_rr/). Prints aggregates only.

  ml/.venv/Scripts/python.exe ml/scripts/rr_phase1.py ruler
  ml/.venv/Scripts/python.exe ml/scripts/rr_phase1.py train-gate
  ml/.venv/Scripts/python.exe ml/scripts/rr_phase1.py train-retriever --seed 2
  ml/.venv/Scripts/python.exe ml/scripts/rr_phase1.py score --retriever-dir ml/output/rr_phase0/bce --partition calibration
  ml/.venv/Scripts/python.exe ml/scripts/rr_phase1.py bakeoff
  ml/.venv/Scripts/python.exe ml/scripts/rr_phase1.py freeze --gate old --retriever-dir ml/output/rr_phase0/bce
  ml/.venv/Scripts/python.exe ml/scripts/rr_phase1.py train-shared-gate --retriever-dir ml/output/rr_phase0/bce
  ml/.venv/Scripts/python.exe ml/scripts/rr_phase1.py time-cpu --n 50

Outputs go under config.RR_PHASE1_DIR. ``--limit N`` is a smoke: N cases, and everything goes under RR_PHASE1_DIR/_smoke.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: F401  must load before scipy/sklearn on Windows (libomp DLL clash, WinError 1114)

import config
from bronze_rr import bakeoff, freeze, ruler, scoring, shared_gate, timing, train
from bronze_rr.data import SILVER_ID, load_data
from evaluation import silver_eval
from report_mapping.training.recipe import PRODUCTION_SEED


def _root(limit: int) -> Path:
    return config.RR_PHASE1_DIR / "_smoke" if limit else config.RR_PHASE1_DIR


def _cmd_ruler(args: argparse.Namespace) -> int:
    print(ruler.SUMMARY_HEADER)
    for partition in ("test", "calibration"):
        summary = ruler.score_partition(args.predictions, args.split, partition)
        print(ruler.summary_row(f"{args.predictions.name} on {partition}", summary))
        if partition == "test":
            print(f"expected WP14 test (rows {ruler.WP14_TEST['total']}, Good {ruler.WP14_TEST['good']}, ...): "
                  f"{'MATCH' if ruler.matches_wp14_test(summary) else 'MISMATCH'}")
    return 0


def _recipe(args: argparse.Namespace) -> train.Recipe:
    return train.Recipe(epochs=args.epochs, max_length=args.max_length, token_budget=args.token_budget, lr=args.lr,
                        seed=args.seed, limit_train=args.limit, max_steps=args.max_steps)


def _cmd_train_gate(args: argparse.Namespace) -> int:
    out = args.out or _root(args.limit) / "gate"
    print(json.dumps(train.train_gate(load_data(), out, _recipe(args)), indent=1))
    return 0


def _cmd_train_retriever(args: argparse.Namespace) -> int:
    out = args.out or _root(args.limit) / f"retriever_seed{args.seed}"
    print(json.dumps(train.train_retriever(load_data(), out, _recipe(args)), indent=1))
    return 0


def _cmd_train_shared_gate(args: argparse.Namespace) -> int:
    root = _root(args.limit)
    out = args.out or shared_gate.gate_dir_for(root, args.retriever_dir)
    print(json.dumps(shared_gate.train_shared_gate(load_data(), args.retriever_dir, out, scores_dir=root / "scores",
                                                   limit=args.limit, seed=args.seed, epochs=args.epochs), indent=1))
    return 0


def _cmd_time_cpu(args: argparse.Namespace) -> int:
    root = config.RR_PHASE1_DIR
    shared = args.shared_gate_dir or shared_gate.gate_dir_for(root, args.retriever_dir)
    result = timing.time_cpu(load_data(), args.retriever_dir, args.gate_dir, shared if shared.is_dir() else None, args.n)
    print(json.dumps(result, indent=1))
    if args.save:
        print(f"saved {timing.save(result)}")
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    scoring.check_partition_allowed(args.partition, args.i_understand_this_spends_the_test_set)
    if not (args.retriever_dir or args.gate_dir):
        raise SystemExit("give --retriever-dir and/or --gate-dir")
    data, root = load_data(), _root(args.limit)
    if args.retriever_dir:
        scores = scoring.retriever_scores(args.retriever_dir, data, args.partition, out_dir=root / "scores", limit=args.limit)
        print(f"retriever {scoring.model_tag(args.retriever_dir)}: {len(scores.case_ids)} cases scored/cached")
    if args.gate_dir:
        probs = scoring.gate_scores(args.gate_dir, data, args.partition, out_dir=root / "scores", limit=args.limit)
        print(f"gate {scoring.model_tag(args.gate_dir)}: {len(probs)} cases scored/cached")
    return 0


def _cmd_bakeoff(args: argparse.Namespace) -> int:
    root = _root(args.limit)
    bakeoff.run_bakeoff(load_data(), silver_eval.load_labels(SILVER_ID), root=root, limit=args.limit)
    print((root / "bakeoff" / "summary.md").read_text(encoding="utf-8"))
    return 0


def _cmd_freeze(args: argparse.Namespace) -> int:
    result = freeze.run_freeze(load_data(), silver_eval.load_labels(SILVER_ID), gate=args.gate,
                               retriever_dir=args.retriever_dir, root=_root(args.limit), limit=args.limit)
    print(json.dumps(result, indent=1))
    return 0


def _add_training_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=PRODUCTION_SEED)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5, help="encoder learning rate")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--token-budget", type=int, default=16384, help="padded tokens per micro-batch")
    parser.add_argument("--limit", type=int, default=0, help="smoke: first N training cases; output under _smoke")
    parser.add_argument("--max-steps", type=int, default=0, help="smoke: stop after N optimizer steps")
    parser.add_argument("--out", type=Path, default=None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ruler", help="score the WP14 baseline predictions with WP14's uncommon list (test and calibration)")
    p.add_argument("--predictions", type=Path, default=config.PREDICTIONS_DIR / "gen-20260927T025939Z_predictions.csv")
    p.add_argument("--split", default=config.DEFAULT_SPLIT_ID)
    p.set_defaults(func=_cmd_ruler)

    p = sub.add_parser("train-gate", help="train the ModernBERT case-presence gate into rr_phase1/gate")
    _add_training_flags(p)
    p.set_defaults(func=_cmd_train_gate)

    p = sub.add_parser("train-retriever", help="train a BCE term retriever into rr_phase1/retriever_seed<S>")
    _add_training_flags(p)
    p.set_defaults(func=_cmd_train_retriever)

    p = sub.add_parser("train-shared-gate", help="train a gate head on a retriever's pooled vectors into rr_phase1/gate_shared_<tag>")
    p.add_argument("--retriever-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=PRODUCTION_SEED)
    p.add_argument("--epochs", type=int, default=50, help="head epochs; the best held-out-AUC epoch is kept")
    p.add_argument("--limit", type=int, default=0, help="smoke: first N train cases; output under _smoke")
    p.add_argument("--out", type=Path, default=None)
    p.set_defaults(func=_cmd_train_shared_gate)

    p = sub.add_parser("time-cpu", help="CPU seconds per report: retriever + separate gate vs retriever + shared head")
    p.add_argument("--n", type=int, default=50, help="first N calibration reports")
    p.add_argument("--retriever-dir", type=Path, default=config.RR_PHASE0_DIR / "bce")
    p.add_argument("--gate-dir", type=Path, default=config.RR_PHASE1_DIR / "gate")
    p.add_argument("--shared-gate-dir", type=Path, default=None,
                   help="default rr_phase1/gate_shared_<retriever tag>; an untrained stand-in head if it does not exist")
    p.add_argument("--save", action="store_true", help="also write rr_phase1/timing/cpu_n<N>.json")
    p.set_defaults(func=_cmd_time_cpu)

    p = sub.add_parser("score", help="score a retriever and/or gate over a partition; cached as numbers")
    p.add_argument("--retriever-dir", type=Path, default=None)
    p.add_argument("--gate-dir", type=Path, default=None)
    p.add_argument("--partition", choices=("train", "calibration", "test", "all"), default="calibration")
    p.add_argument("--i-understand-this-spends-the-test-set", action="store_true",
                   help="required for --partition test or all")
    p.add_argument("--limit", type=int, default=0, help="smoke: first N cases per partition; output under _smoke")
    p.set_defaults(func=_cmd_score)

    p = sub.add_parser("bakeoff", help="gate bake-off on the calibration partition (fit half -> eval half)")
    p.add_argument("--limit", type=int, default=0, help="smoke: first N calibration cases; output under _smoke")
    p.set_defaults(func=_cmd_bakeoff)

    p = sub.add_parser("freeze", help="fit thresholds on the full calibration partition, write a generation and predictions")
    p.add_argument("--gate", choices=("old", "new", "shared"), required=True)
    p.add_argument("--retriever-dir", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0, help="smoke: first N cases per partition; output under _smoke")
    p.set_defaults(func=_cmd_freeze)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
