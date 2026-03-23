"""Build and save a KnnGroupSelector from the embedding cache.

No env PYTHONPATH needed — this script adds ml/ to sys.path automatically.

Usage:
  python ml/scripts/build_knn_selector.py
  python ml/scripts/build_knn_selector.py --k 15 --min-group-cases 20
  python ml/scripts/build_knn_selector.py --mean-only
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.knn_group_selector import KnnGroupSelector


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Build a KnnGroupSelector from the embedding cache and LLM predictions."
    )
    parser.add_argument(
        "--cache",
        default="ml/data/embedding_cache.npz",
        help="Path to embedding cache npz (default: ml/data/embedding_cache.npz)",
    )
    parser.add_argument(
        "--labels-csv",
        default="ml/output/annotation/llm/llm_annotation.csv",
        help="Path to predictions CSV with case_id and matched_group columns "
             "(default: ml/output/annotation/llm/llm_annotation.csv)",
    )
    parser.add_argument(
        "--out",
        default="ml/model/checkpoints/knn_group_selector.npz",
        help="Output path (default: ml/model/checkpoints/knn_group_selector.npz)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of nearest neighbours to vote with (default: 10)",
    )
    parser.add_argument(
        "--min-group-cases",
        type=int,
        default=10,
        help="Drop groups with fewer confirmed unique cases (default: 10)",
    )
    parser.add_argument(
        "--mean-only",
        action="store_true",
        help="Use mean embeddings (768-dim) instead of per-column concat (2304-dim).",
    )
    args = parser.parse_args()

    selector = KnnGroupSelector.build(
        cache_path=args.cache,
        labels_csv_path=args.labels_csv,
        k=args.k,
        per_column=not args.mean_only,
        min_group_cases=args.min_group_cases,
    )
    selector.save(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
