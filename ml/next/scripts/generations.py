"""Report-mapping generation management.

  ml/.venv/bin/python ml/next/scripts/generations.py import-gen0
  ml/.venv/bin/python ml/next/scripts/generations.py import-cache [--generation current|candidate|<dir>]
  ml/.venv/bin/python ml/next/scripts/generations.py status --silver SID [--split ID] [--incumbent-predictions PATH]

``import-gen0`` and ``import-cache`` are WP4's contribution (one-time legacy
imports for gen-0). ``status`` (WP10) shows current/ and candidate/ and whether a
challenger trained on ``--silver`` would meet a retraining trigger; promotion
itself is ``promote.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from generations.manifest import read_manifest
from generations.promote import trigger_status
from report_mapping.inference.embedding_cache import import_legacy_cache
from report_mapping.model.generation import import_legacy_gen0, resolve_generation_dir


def _import_gen0(_args: argparse.Namespace) -> None:
    manifest = import_legacy_gen0()
    print(f"Imported gen-0: generation_id={manifest['generation_id']!r}")
    print(f"  files: {len(manifest['files'])}")
    print(f"  embedding_fingerprint: {manifest['embedding_fingerprint']}")


def _import_cache(args: argparse.Namespace) -> None:
    key = import_legacy_cache(resolve_generation_dir(args.generation))
    print(f"Imported legacy embedding cache -> key {key}")


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen0 = sub.add_parser("import-gen0", help="Copy the legacy checkpoints into config.REPORT_MAPPING_CURRENT_DIR.")
    p_gen0.set_defaults(func=_import_gen0)

    p_cache = sub.add_parser("import-cache", help="Copy the legacy embedding cache under the new content-hash key.")
    p_cache.add_argument("--generation", default="current")
    p_cache.set_defaults(func=_import_cache)

    p_status = sub.add_parser("status", help="current/ + candidate/ and the retraining triggers.")
    p_status.add_argument("--silver", required=True, help="The silver_id a challenger would train on.")
    p_status.add_argument("--split", default=None, help="split_id whose test side holds gold-eval (default: config).")
    p_status.add_argument("--incumbent-predictions", default=None,
                          help="current/'s predictions CSV (default: config.PREDICTIONS_DIR/<id>_predictions.csv); "
                               "read only when gold-eval exists.")
    p_status.set_defaults(func=_status)

    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
