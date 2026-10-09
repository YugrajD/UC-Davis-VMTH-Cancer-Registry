"""Report-mapping generation management.

  ml/.venv/Scripts/python.exe ml/scripts/generations.py status [--silver SID [--split ID] [--incumbent-predictions PATH]]
  ml/.venv/Scripts/python.exe ml/scripts/generations.py recalibrate --labels SID [--split ID] [--from current]
      [--to candidate] [--max-fp-share X] [--device cuda]

``status`` lists current/, candidate/ and the archive (generation id, architecture, calibration status, adoption
reason); with ``--silver`` it also shows whether a challenger trained on that silver would meet a retraining
trigger. Promotion itself is ``promote.py``. ``recalibrate`` copies a generation's weights into a new, uncalibrated
generation (new id, same weights) and fits new thresholds on it: new thresholds for the same weights are a new
generation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from generations.manifest import read_manifest
from generations.promote import trigger_status
from report_mapping.generation import copy_for_recalibration, resolve_generation_dir
from report_mapping.model import resolve_device
from report_mapping.training.calibrate import calibrate


def _describe(name: str, directory: Path) -> str:
    m = read_manifest(directory)
    parents = m.get("parents") or {}
    adoption = (m.get("adoption") or {}).get("reason")
    return (f"{name}: {m['generation_id']}  status={m.get('status')}  "
            f"architecture={(m.get('architecture') or {}).get('name')}  "
            f"calibration={(m.get('calibration') or {}).get('status')}  silver={parents.get('silver_id')}  "
            f"split={parents.get('split_id')}  gold_train_codes={parents.get('gold_train_codes')}"
            + (f"  adopted_for={adoption}" if adoption else ""))


def _status(args: argparse.Namespace) -> int:
    for name, directory in (("current", config.REPORT_MAPPING_CURRENT_DIR),
                            ("candidate", config.REPORT_MAPPING_CANDIDATE_DIR)):
        print(_describe(name, directory) if directory.is_dir() else f"{name}: none")
    archived = sorted(p for p in config.ARCHIVE_ROOT.glob("*") if (p / "manifest.json").is_file())
    print(f"archive: {len(archived)} generation(s)")
    for directory in archived:
        print("  " + _describe(directory.name, directory))
    if args.silver:
        current_id = read_manifest(config.REPORT_MAPPING_CURRENT_DIR)["generation_id"]
        predictions = args.incumbent_predictions or config.PREDICTIONS_DIR / f"{current_id}_predictions.csv"
        fired = trigger_status(args.silver, predictions, split_id=args.split)
        for trigger in fired:
            numbers = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                for k, v in trigger.numbers.items())
            print(f"trigger {trigger.name:<20} {'MET' if trigger.met else 'not met'}  ({numbers})")
        print(f"Retrain: {'yes' if any(t.met for t in fired) else 'no trigger met'}")
    return 0


def _recalibrate(args: argparse.Namespace) -> int:
    dest = resolve_generation_dir(args.to)
    manifest = copy_for_recalibration(resolve_generation_dir(args.source), dest)
    print(f"copied {manifest['recalibrated_from']} -> {manifest['generation_id']} at {dest} (calibration pending)")
    split_id = args.split if args.split is not None else config.DEFAULT_SPLIT_ID
    max_fp_share = args.max_fp_share if args.max_fp_share is not None else config.FP_CAP
    block = calibrate(dest, args.labels, split_id, max_fp_share, config.REVIEW_TARGET_SHARE,
                      resolve_device(args.device))
    print("calibrated: " + ", ".join(f"{k}={v}" for k, v in block["values"].items())
          + f"; in-sample G+S {100 * block['partition_gs_share']:.2f}%")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_status = sub.add_parser("status", help="current/, candidate/, the archive, and optionally the retraining triggers.")
    p_status.add_argument("--silver", default=None, help="The silver_id a challenger would train on (shows the triggers).")
    p_status.add_argument("--split", default=None, help="split_id whose test side holds gold-eval (default: config).")
    p_status.add_argument("--incumbent-predictions", default=None,
                          help="current/'s predictions CSV (default: config.PREDICTIONS_DIR/<id>_predictions.csv); "
                               "read only when gold-eval exists.")
    p_status.set_defaults(func=_status)

    p_re = sub.add_parser("recalibrate", help="Copy a generation's weights as a new generation and refit its thresholds.")
    p_re.add_argument("--labels", required=True, help="silver_id or labels-table CSV path for the calibration partition.")
    p_re.add_argument("--split", default=None, help=f"split_id holding the calibration partition (default: {config.DEFAULT_SPLIT_ID}).")
    p_re.add_argument("--from", dest="source", default="current")
    p_re.add_argument("--to", default="candidate")
    p_re.add_argument("--max-fp-share", type=float, default=None, help=f"FP cap (default: config.FP_CAP = {config.FP_CAP}).")
    p_re.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p_re.set_defaults(func=_recalibrate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
