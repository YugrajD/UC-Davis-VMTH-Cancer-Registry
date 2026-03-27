"""Run production inference: score all reports against the cancer label taxonomy.

No env PYTHONPATH needed — this script adds ml/ to sys.path automatically.

Auto-detects the best available classifier checkpoint (contrastive-backbone
preferred; falls back to binary-backbone). Pass --presence-classifier explicitly
to override.

Usage:
  python ml/scripts/run_production.py --local-only
  python ml/scripts/run_production.py --max-rows 100 --local-only
  python ml/scripts/run_production.py --presence-classifier ml/output/checkpoints/contrastive/presence_classifier_best.pt

All scoring pipeline flags are supported — run with --help to see all options.
"""

import sys
from pathlib import Path

# Add ml/ to sys.path so all packages are importable without setting PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from production.petbert_pipeline import build_config, build_parser, run_scan


def main() -> int:
    classifier, subdir = config.best_checkpoint_info()
    # When the contrastive backbone produced the best classifier, use it for
    # embedding too — otherwise the cache would miss and embeddings would be
    # produced by plain PetBERT, mismatching what the classifier was trained on.
    embedding_model = (
        config.CHECKPOINT_CONTRASTIVE_DIR if subdir == "contrastive" else "SAVSNET/PetBERT"
    )

    parser = build_parser()
    parser.set_defaults(
        model=embedding_model,
        embedding_cache=config.EMBEDDING_CACHE_NPZ,
        presence_classifier=classifier,
        out_dir=f"{config.OUTPUT_PRODUCTION_DIR}/{subdir}",
        local_only=True,
    )
    run_scan(build_config(parser.parse_args()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
