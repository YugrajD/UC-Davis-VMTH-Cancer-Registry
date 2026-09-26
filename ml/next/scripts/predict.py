"""Thin entry point: stamped predictions from a report-mapping generation.

Replaces ``ml/scripts/run_production.py``. All the work lives in
``report_mapping.inference.predict.run_predict``; this script only parses
args and calls it.

  ml/.venv/bin/python ml/next/scripts/predict.py --generation current --local-only
  ml/.venv/bin/python ml/next/scripts/predict.py --generation current --embed-only --local-only
  ml/.venv/bin/python ml/next/scripts/predict.py --generation candidate --model SAVSNET/PetBERT --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from report_mapping.inference.predict import run_predict
from report_mapping.model.generation import resolve_generation_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--generation", required=True,
                        help='"current", "candidate", or a literal generation directory path.')
    parser.add_argument("--model", default=None,
                        help="HF checkpoint dir or HF name. Defaults to the generation's own petbert/.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps", "xpu"])
    parser.add_argument("--embed-only", action="store_true",
                        help="Populate the embedding cache and stop before classification.")
    parser.add_argument("--out", default=None, help="Predictions CSV path (default: config.PREDICTIONS_DIR).")
    parser.add_argument("--local-only", action="store_true", help="Disable HuggingFace download.")
    parser.add_argument("--cache-dir", default=None,
                        help="Embedding cache directory (default: config.EMBEDDING_CACHE_DIR). Point this at an "
                             "empty scratch directory to force a re-embed without touching the real cache — "
                             "e.g. the L2b parity re-embed, which must not read or overwrite the imported "
                             "legacy cache that's synced between machines.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_predict(
        generation_dir=resolve_generation_dir(args.generation),
        model_override=args.model,
        device_arg=args.device,
        local_only=args.local_only,
        embed_only=args.embed_only,
        out_path=args.out,
        cache_dir=args.cache_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
