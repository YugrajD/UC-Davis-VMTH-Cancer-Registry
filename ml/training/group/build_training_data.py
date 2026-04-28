"""Build training data for the GroupClassifier.

Reads:
  - ml/output/training/embedding_cache.npz                         -- embeddings and case_ids
  - ml/output/annotation/llm/llm_annotation.csv  -- ground-truth (case_id, matched_group)

Produces:
  - ml/output/training/group/group_training_data.npz with:
      embeddings    (N, D)  float32  -- report embedding per case (D=2304 per-column, or 768 mean)
      targets       (N, G)  float32  -- multi-hot group labels (0.0 = non-cancer, 1.0 = present)
      case_ids      (N,)    object   -- case_id strings
      group_names   (G,)    object   -- group name strings (index = column in targets)
      class_weights (G,)    float32  -- inverse-frequency weights for BCEWithLogitsLoss

Ground-truth assumption: cases in the cache that do not appear in the labels CSV
are treated as non-cancer (all-zeros target). This is valid for a general veterinary
clinic population where ~18% cancer prevalence is expected.

Usage:
  python ml/training/group/build_training_data.py
  python ml/training/group/build_training_data.py \\
      --embedding-cache ml/output/training/embedding_cache.npz \\
      --expectation-csv ml/output/annotation/llm/llm_annotation.csv \\
      --out ml/output/training/group/group_training_data.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config


def build_training_data(
    cache_path: str,
    expectation_csv_path: str,
    out_path: str,
    per_column: bool = True,
) -> None:
    # --- Load embedding cache ------------------------------------------------
    print(f"Loading embedding cache: {cache_path}")
    if not Path(cache_path).exists():
        print(f"ERROR: cache not found at {cache_path}")
        print("Run the PetBERT scan first to generate the embedding cache:")
        print("  ml/.venv/Scripts/python.exe -m petbert_pipeline --embedding-cache ml/output/training/embedding_cache.npz --local-only")
        sys.exit(1)

    cache = np.load(cache_path, allow_pickle=True)
    case_ids: list[str] = list(cache["case_ids"])
    N = len(case_ids)
    case_idx = {cid: i for i, cid in enumerate(case_ids)}

    if per_column:
        col_keys = ["col_FINAL_COMMENT", "col_HISTOPATHOLOGICAL_SUMMARY", "col_ANCILLARY_TESTS"]
        embeddings = np.concatenate([cache[k] for k in col_keys], axis=1)  # (N, 2304)
        print(f"Using per-column embeddings: {embeddings.shape[1]}-dim ({len(col_keys)} cols × 768)")
    else:
        embeddings = cache["mean_embeddings"]  # (N, 768)
        print(f"Using mean embeddings: {embeddings.shape[1]}-dim")

    # --- Load keyword predictions --------------------------------------------
    print(f"Loading keyword predictions: {expectation_csv_path}")
    if not Path(expectation_csv_path).exists():
        print(f"ERROR: keyword predictions not found at {expectation_csv_path}")
        print("Run the keyword scan first:")
        print("  ml/.venv/bin/python3 -m keyword_scan")
        sys.exit(1)

    kw = pd.read_csv(expectation_csv_path)
    kw = kw[kw["matched_group"].notna()].copy()

    # --- Build sorted group list (deterministic ordering) --------------------
    all_groups: list[str] = sorted(kw["matched_group"].unique())
    G = len(all_groups)
    group_idx = {g: i for i, g in enumerate(all_groups)}
    print(f"Found {G} cancer groups across {kw['case_id'].nunique()} keyword-confirmed cases")

    # --- Build multi-hot targets (N, G) — default all zeros (non-cancer) ----
    targets = np.zeros((N, G), dtype=np.float32)
    skipped = 0
    for _, row in kw.iterrows():
        cid = str(row["case_id"])
        group = row["matched_group"]
        if cid not in case_idx:
            skipped += 1
            continue
        if group in group_idx:
            targets[case_idx[cid], group_idx[group]] = 1.0

    if skipped > 0:
        print(f"  Note: {skipped} keyword rows had case_ids not found in the embedding cache")

    # --- Class weights for BCEWithLogitsLoss pos_weight ----------------------
    # pos_weight[g] = negatives / positives for group g.
    # This balances the BCE loss so positive examples contribute equally to
    # negative examples. e.g. group with 200 positives out of 2845 cases:
    #   weight = (2845 - 200) / 200 ≈ 13.2
    # Without this, the model learns to output near-zero probabilities for
    # everything since most cases are non-cancer for any given group.
    positive_counts = targets.sum(axis=0)  # (G,) — positives per group
    negative_counts = N - positive_counts
    # Guard against zero positives (groups absent from the cache)
    positive_counts_safe = np.maximum(positive_counts, 1.0)
    class_weights = (negative_counts / positive_counts_safe).astype(np.float32)

    # --- Summary stats -------------------------------------------------------
    cancer_cases = int((targets.sum(axis=1) > 0).sum())
    non_cancer_cases = N - cancer_cases
    print(f"Cancer cases (keyword-matched in cache): {cancer_cases}")
    print(f"Non-cancer cases (assumed):              {non_cancer_cases}")
    print(f"Class weight range: {class_weights.min():.2f} – {class_weights.max():.2f}")
    print()
    print(f"{'Group':<50} {'Cases':>6} {'Weight':>7}")
    print("-" * 65)
    for g, group_name in enumerate(all_groups):
        print(f"{group_name:<50} {int(positive_counts[g]):>6} {class_weights[g]:>7.2f}")

    # --- Save ----------------------------------------------------------------
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        embeddings=embeddings,
        targets=targets,
        case_ids=np.array(case_ids, dtype=object),
        group_names=np.array(all_groups, dtype=object),
        class_weights=class_weights,
    )
    print(f"\nSaved: {out_path} ({N} cases, {G} groups)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build GroupClassifier training data from embedding cache and ground-truth predictions."
    )
    parser.add_argument(
        "--embedding-cache",
        default=config.EMBEDDING_CACHE_NPZ,
        help=f"Path to embedding cache npz (default: {config.EMBEDDING_CACHE_NPZ})",
    )
    parser.add_argument(
        "--expectation-csv",
        default=config.LLM_ANNOTATION_CSV,
        help="Path to predictions CSV with case_id and matched_group columns",
    )
    parser.add_argument(
        "--out",
        default=config.GROUP_TRAINING_DATA_NPZ,
        help=f"Output npz path (default: {config.GROUP_TRAINING_DATA_NPZ})",
    )
    parser.add_argument(
        "--mean-only",
        action="store_true",
        help="Use mean embeddings (768-dim) instead of per-column concat (2304-dim).",
    )
    args = parser.parse_args()
    build_training_data(args.embedding_cache, args.expectation_csv, args.out, per_column=not args.mean_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
