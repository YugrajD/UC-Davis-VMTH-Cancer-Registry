"""Run the PetBERT production pipeline.

No env PYTHONPATH needed — this script adds ml/ to sys.path automatically.

Usage:
  python ml/scripts/run_production.py
  python ml/scripts/run_production.py --max-rows 100 --local-only
  python ml/scripts/run_production.py --presence-classifier ml/model/checkpoints/presence_classifier_best.pt
  python ml/scripts/run_production.py --group-classifier ml/model/checkpoints/group_classifier_best.pt

All petbert_pipeline CLI flags are supported — run with --help to see all options.
"""

import sys
from pathlib import Path

# Add ml/ to the path so all packages are importable without env PYTHONPATH=ml
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from production.petbert_pipeline.cli import build_config, build_parser
from production.petbert_pipeline.pipeline import run_scan


def _best_classifier() -> str | None:
    p = Path("ml/model/checkpoints/presence_classifier_best.pt")
    return str(p) if p.exists() else None


def main() -> int:
    parser = build_parser()
    parser.set_defaults(
        embedding_cache="ml/data/embedding_cache.npz",
        presence_classifier=_best_classifier(),
        out_dir="ml/output/production",
        local_only=True,
    )
    run_scan(build_config(parser.parse_args()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
