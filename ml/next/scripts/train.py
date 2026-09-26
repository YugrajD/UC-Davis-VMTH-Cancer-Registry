"""Train a report-mapping candidate generation.

Thin entry point: all training logic lives in ``report_mapping.training.*``.
This script only parses args, resolves the labels table (a silver_id or a CSV
path), makes sure a heads-only stage has a backbone to embed with, dispatches
to the right trainer(s), and writes/refreshes the candidate's manifest with a
placeholder ``thresholds.json`` copied from the parent (current) generation —
flagged ``calibration.status: "pending"`` until WP5b's ``calibrate.py`` fills it in.

No env PYTHONPATH needed — this script adds ml/next/ to sys.path itself.

Usage
-----
  python ml/next/scripts/train.py --stage heads --labels silver-0-legacy \\
      --split three-way-v1 --seed 42 --device cuda
  python ml/next/scripts/train.py --stage backbone --labels silver-0-legacy --device cuda
  python ml/next/scripts/train.py --stage oof --oof-stage case-presence --labels silver-0-legacy
"""

from __future__ import annotations

import argparse
import platform
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

import config
from generations import guards
from generations.manifest import write_manifest
from report_mapping.model import generation as generation_mod
from report_mapping.training import backbone as backbone_mod
from report_mapping.training import case_presence as case_presence_mod
from report_mapping.training import group as group_mod
from report_mapping.training import label_presence as label_presence_mod
from report_mapping.training import labels as labels_mod
from report_mapping.training import oof as oof_mod
from report_mapping.training import recipe


def _ensure_backbone(out_dir: Path, backbone_override: str | None) -> str:
    """Heads-only stages need ``<out_dir>/petbert/`` populated before they can
    build/read the embedding cache. Copies the chosen backbone in (default:
    the current generation's — ml-rewrite-plan.md WP5's "frozen old backbone"
    case) unless the candidate already has one from an earlier stage this
    cycle (so running case-presence then group then label-presence in
    sequence reuses the same backbone, and thus the same embedding cache)."""
    paths = generation_mod.generation_paths(out_dir)
    resolved = backbone_override if backbone_override is not None else case_presence_mod.default_backbone_dir()
    if not paths.petbert_dir.exists():
        shutil.copytree(resolved, paths.petbert_dir)
    return str(paths.petbert_dir)


def _write_candidate_manifest(out_dir: Path, *, labels_source: str, split_id: str, seed: int,
                               device: str, stage: str) -> dict:
    paths = generation_mod.generation_paths(out_dir)
    parent_paths = generation_mod.generation_paths(config.REPORT_MAPPING_CURRENT_DIR)

    # thresholds.json: copied from the parent generation as a placeholder --
    # WP5b's calibrate.py fits real values on the calibration partition.
    if parent_paths.thresholds_json.is_file() and not paths.thresholds_json.is_file():
        paths.thresholds_json.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(parent_paths.thresholds_json, paths.thresholds_json)
    if not paths.labels_csv.is_file():
        paths.labels_csv.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(config.LABELS_CSV, paths.labels_csv)

    fingerprint = (
        generation_mod.compute_embedding_fingerprint(paths.petbert_dir) if paths.petbert_dir.is_dir() else None
    )
    fields = {
        "kind": "report_mapping_generation",
        "generation_id": out_dir.name,
        "parents": {"labels_source": labels_source, "split_id": split_id, "gold_train_snapshot": None},
        "recipe": {
            "gate": recipe.GATE.__dict__,
            "group": recipe.GROUP.__dict__,
            "label_presence": recipe.LABEL_PRESENCE.__dict__,
            "backbone": recipe.BACKBONE.__dict__,
        },
        "seed": seed,
        "device": device,
        "library_versions": {"torch": torch.__version__, "python": platform.python_version()},
        "embedding_fingerprint": fingerprint,
        "status": "candidate",
        "calibration": {"status": "pending", "partition": None, "objective": None, "values": None},
        "last_stage": stage,
    }
    return write_manifest(out_dir, fields)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a report-mapping candidate generation.")
    parser.add_argument("--stage", required=True,
                         choices=["backbone", "case-presence", "group", "label-presence", "heads", "oof"])
    parser.add_argument("--labels", required=True, help="A silver_id (diagnosis_mapping.silver) or a labels CSV path")
    parser.add_argument("--split", default=None, help=f"Split id (default: {config.DEFAULT_SPLIT_ID})")
    parser.add_argument("--seed", type=int, default=recipe.PRODUCTION_SEED)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps", "xpu"])
    parser.add_argument("--out", default="candidate", help="'current', 'candidate', or a literal directory")
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--model", default=None,
                         help="[heads-only / backbone] backbone dir/name override "
                              "(default: the current generation's petbert/)")
    parser.add_argument("--oof-stage", default="case-presence", choices=["case-presence", "group"],
                         help="[--stage oof] which head to run k-fold out-of-fold predictions for")
    parser.add_argument("--k", type=int, default=oof_mod.DEFAULT_K, help="[--stage oof] number of folds")
    args = parser.parse_args(argv)

    split_id = args.split if args.split is not None else config.DEFAULT_SPLIT_ID

    # ml-rewrite-plan.md: "split.py check runs inside train, calibrate, evaluate
    # gold and promote and refuses on any violation." Run before any data
    # loading, and only for the two stages that actually (re)build a
    # generation end to end (heads, backbone) -- guards.check_all's other
    # checks (disjointness, gold origins, ...) are independent of which head
    # a lower-level sub-stage happens to be training.
    if args.stage in ("heads", "backbone"):
        guards.check_all(split_id)

    out_dir = generation_mod.resolve_generation_dir(args.out)
    labels = labels_mod.load_labels_table(args.labels)

    if args.stage == "oof":
        run = oof_mod.run_case_presence_oof if args.oof_stage == "case-presence" else oof_mod.run_group_oof
        result = run(labels, split_id, args.seed, args.device, k=args.k,
                      backbone_dir=args.model, local_only=args.local_only)
        print(f"oof[{args.oof_stage}]: {len(result.case_ids)} train cases, k={args.k}")
        return 0  # a diagnostic run, not a training stage -- no candidate manifest to write

    if args.stage == "backbone":
        result = backbone_mod.train(labels, split_id, args.seed, args.device,
                                     model_name=args.model, local_only=args.local_only, out_dir=out_dir)
        print(f"backbone: {result}")
    else:
        backbone_dir = _ensure_backbone(out_dir, args.model)
        if args.stage == "case-presence":
            print(f"gate: {case_presence_mod.train(labels, split_id, args.seed, args.device, backbone_dir=backbone_dir, local_only=args.local_only, out_dir=out_dir)}")
        elif args.stage == "group":
            print(f"group: {group_mod.train(labels, split_id, args.seed, args.device, backbone_dir=backbone_dir, local_only=args.local_only, out_dir=out_dir)}")
        elif args.stage == "label-presence":
            print(f"label-presence: {label_presence_mod.train(labels, split_id, args.seed, args.device, backbone_dir=backbone_dir, local_only=args.local_only, out_dir=out_dir)}")
        elif args.stage == "heads":
            print(f"gate: {case_presence_mod.train(labels, split_id, args.seed, args.device, backbone_dir=backbone_dir, local_only=args.local_only, out_dir=out_dir)}")
            print(f"group: {group_mod.train(labels, split_id, args.seed, args.device, backbone_dir=backbone_dir, local_only=args.local_only, out_dir=out_dir)}")
            print(f"label-presence: {label_presence_mod.train(labels, split_id, args.seed, args.device, backbone_dir=backbone_dir, local_only=args.local_only, out_dir=out_dir)}")

    manifest = _write_candidate_manifest(out_dir, labels_source=args.labels, split_id=split_id,
                                          seed=args.seed, device=args.device, stage=args.stage)
    print(f"candidate manifest: {out_dir / 'manifest.json'} (generation_id={manifest['generation_id']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
