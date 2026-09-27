"""The local lane of the full cycle in one go (icd-mapping-strategy.md, "The full cycle"). Recommend-only.

  ml/.venv/bin/python ml/scripts/retrain_cycle.py --silver SID [--split ID] [--device cuda] [--local-only]
      [--gold-csv PATH --export-id ID --reviewer NAME [--upload-period YYYY-MM] [--slice-rate R]]
      [--backbone] [--force]

Steps, each the existing entry point run as its own process (the parity runbook's commands, and a fresh
process frees GPU memory between steps):

1. ``handoff.py import-gold`` when ``--gold-csv`` is given.
2. Stop unless gold-eval exists: promotion cannot be decided without it.
3. ``predict.py`` for current/ if its predictions are missing (the random-slice trigger scores them), then
   stop unless a retraining trigger is met for a challenger trained on ``--silver`` (``--force`` trains
   anyway; ``promote.py`` still needs a trigger).
4. ``code_cases.py corrected``: latest silver + gold-train, the table the challenger trains on.
5. ``train.py`` heads on the current backbone, after ``--stage backbone`` when ``--backbone`` is given.
6. ``calibrate.py``, then ``predict.py`` for the candidate.
7. ``promote.py`` without ``--apply``. Applying the recommendation is left to the Admin.

It never runs the LLM cascade: a new silver generation is its own step (``map_diagnoses.py``).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from generations.manifest import read_manifest
from generations.promote import trigger_status
from manual_audit import gold

SCRIPTS = Path(__file__).resolve().parent


def _run(script: str, *args: str) -> None:
    print(f"\n== {script} {' '.join(args)}", flush=True)
    returncode = subprocess.run([sys.executable, str(SCRIPTS / script), *args]).returncode
    if returncode:
        raise SystemExit(returncode)  # the step printed its own error; stop the cycle with its exit code


def _predictions_path(generation_dir: Path) -> Path:
    return config.PREDICTIONS_DIR / f"{read_manifest(generation_dir)['generation_id']}_predictions.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--silver", required=True, help="The silver_id the challenger trains on.")
    parser.add_argument("--split", default=None, help=f"split_id (default: {config.DEFAULT_SPLIT_ID})")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps", "xpu"])
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--backbone", action="store_true", help="Retrain the backbone before the heads (cold start).")
    parser.add_argument("--force", action="store_true", help="Train even when no retraining trigger is met.")
    parser.add_argument("--gold-csv", help="A gold_<export>.csv from the cloud to ingest first.")
    parser.add_argument("--export-id")
    parser.add_argument("--reviewer")
    parser.add_argument("--upload-period", default="")
    parser.add_argument("--slice-rate", default="")
    parser.add_argument("--n-boot", default="1000")
    args = parser.parse_args()
    if args.gold_csv and not (args.export_id and args.reviewer):
        parser.error("--gold-csv needs --export-id and --reviewer")
    split_id = args.split if args.split is not None else config.DEFAULT_SPLIT_ID
    candidate, current = config.REPORT_MAPPING_CANDIDATE_DIR, config.REPORT_MAPPING_CURRENT_DIR
    if candidate.exists():
        print(f"REFUSED: {candidate} exists; promote or delete it before a new cycle", file=sys.stderr)
        return 1

    if args.gold_csv:
        _run("handoff.py", "import-gold", "--csv", args.gold_csv, "--export-id", args.export_id,
             "--reviewer", args.reviewer, "--upload-period", args.upload_period, "--slice-rate", args.slice_rate)
    if gold.gold_eval(split_id=split_id).empty:
        print(f"STOP: no gold-eval rows for split {split_id!r}, so promotion cannot be decided")
        return 0
    incumbent_predictions = _predictions_path(current)
    if not incumbent_predictions.is_file():
        _run("predict.py", "--generation", "current", "--device", args.device, "--out", str(incumbent_predictions))
    fired = trigger_status(args.silver, incumbent_predictions, split_id=split_id)
    for trigger in fired:
        print(f"trigger {trigger.name:<20} {'MET' if trigger.met else 'not met'}  {trigger.numbers}")
    if not any(t.met for t in fired) and not args.force:
        print("STOP: no retraining trigger met (--force trains anyway)")
        return 0

    _run("code_cases.py", "corrected", "--silver", args.silver, "--split", split_id)
    common = ["--labels", str(config.CORRECTED_ANNOTATIONS_CSV), "--split", split_id]
    local = ["--local-only"] if args.local_only else []
    for stage in (["backbone"] if args.backbone else []) + ["heads"]:
        _run("train.py", "--stage", stage, *common, "--device", args.device, "--out", "candidate", *local)
    _run("calibrate.py", "--generation", "candidate", *common,
         "--device", "cpu" if args.device == "auto" else args.device)
    candidate_predictions = _predictions_path(candidate)
    _run("predict.py", "--generation", "candidate", "--device", args.device, "--out", str(candidate_predictions))
    _run("promote.py", "--candidate-predictions", str(candidate_predictions),
         "--incumbent-predictions", str(incumbent_predictions), "--split", split_id, "--n-boot", args.n_boot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
