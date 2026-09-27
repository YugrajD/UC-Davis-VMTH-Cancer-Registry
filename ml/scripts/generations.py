"""Report-mapping generation management.

  ml/.venv/bin/python ml/scripts/generations.py status --silver SID [--split ID] [--incumbent-predictions PATH]
  ml/.venv/bin/python ml/scripts/generations.py fork --split ID [--from current] [--to candidate]

``status`` (WP10) shows current/ and candidate/ and whether a challenger
trained on ``--silver`` would meet a retraining trigger; promotion itself is
``promote.py``. ``fork`` (WP14) copies a generation as a new, uncalibrated one on
another split with the same train partition; calibrate.py then refits it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from generations.manifest import read_manifest
from generations.promote import trigger_status
from report_mapping.model.generation import fork_generation, resolve_generation_dir


def _status(args: argparse.Namespace) -> None:
    for name, directory in (("current", config.REPORT_MAPPING_CURRENT_DIR),
                            ("candidate", config.REPORT_MAPPING_CANDIDATE_DIR)):
        if not directory.is_dir():
            print(f"{name}: none")
            continue
        m = read_manifest(directory)
        parents = m.get("parents") or {}
        print(f"{name}: {m['generation_id']}  status={m.get('status')}  "
              f"calibration={(m.get('calibration') or {}).get('status')}  silver={parents.get('silver_id')}  "
              f"split={parents.get('split_id')}  gold_train_codes={parents.get('gold_train_codes')}")
    current_id = read_manifest(config.REPORT_MAPPING_CURRENT_DIR)["generation_id"]
    predictions = args.incumbent_predictions or config.PREDICTIONS_DIR / f"{current_id}_predictions.csv"
    fired = trigger_status(args.silver, predictions, split_id=args.split)
    for trigger in fired:
        numbers = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in trigger.numbers.items())
        print(f"trigger {trigger.name:<20} {'MET' if trigger.met else 'not met'}  ({numbers})")
    print(f"Retrain: {'yes' if any(t.met for t in fired) else 'no trigger met'}")


def _fork(args: argparse.Namespace) -> None:
    manifest = fork_generation(resolve_generation_dir(args.source), resolve_generation_dir(args.to),
                               split_id=args.split)
    print(f"forked {manifest['parents']['forked_from']} -> {manifest['generation_id']} on {args.split} "
          f"(calibration pending; run calibrate.py)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_status = sub.add_parser("status", help="current/ + candidate/ and the retraining triggers.")
    p_status.add_argument("--silver", required=True, help="The silver_id a challenger would train on.")
    p_status.add_argument("--split", default=None, help="split_id whose test side holds gold-eval (default: config).")
    p_status.add_argument("--incumbent-predictions", default=None,
                          help="current/'s predictions CSV (default: config.PREDICTIONS_DIR/<id>_predictions.csv); "
                               "read only when gold-eval exists.")
    p_status.set_defaults(func=_status)

    p_fork = sub.add_parser("fork", help="Copy a generation as a new, uncalibrated one on another split.")
    p_fork.add_argument("--split", required=True, help="The split to calibrate on; its train must match.")
    p_fork.add_argument("--from", dest="source", default="current")
    p_fork.add_argument("--to", default="candidate")
    p_fork.set_defaults(func=_fork)

    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
