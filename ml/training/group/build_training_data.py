"""Build training data for the GroupClassifier.

Reads:
  - ml/data/embedding_cache.npz                      -- mean_embeddings and case_ids
  - ml/output/diagnoses/keyword_predictions.csv      -- ground-truth (case_id, matched_group)

Produces:
  - ml/output/group_training_data.npz with:
      embeddings    (N, 768)  float32  -- mean report embedding per case
      targets       (N, G)    float32  -- multi-hot group labels (0.0 = non-cancer, 1.0 = present)
      case_ids      (N,)      object   -- case_id strings
      group_names   (G,)      object   -- group name strings (index = column in targets)
      class_weights (G,)      float32  -- inverse-frequency weights for BCEWithLogitsLoss

Ground-truth assumption: cases in the cache that do not appear in keyword_predictions.csv
are treated as non-cancer (all-zeros target). This is valid for a general veterinary
clinic population where ~18% cancer prevalence is expected.

Usage:
  python ml/training/build_group_training_data.py
  python ml/training/build_group_training_data.py \\
      --embedding-cache ml/data/embedding_cache.npz \\
      --keyword-csv ml/output/diagnoses/keyword_predictions.csv \\
      --out ml/output/group_training_data.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def build_training_data(
    cache_path: str,
    keyword_csv_path: str,
    out_path: str,
) -> None:
    # --- Load embedding cache ------------------------------------------------
    print(f"Loading embedding cache: {cache_path}")
    if not Path(cache_path).exists():
        print(f"ERROR: cache not found at {cache_path}")
        print("Run the PetBERT scan first to generate the embedding cache:")
        print("  ml/.venv/bin/python3 -m petbert_pipeline --embedding-cache ml/data/embedding_cache.npz --local-only")
        sys.exit(1)

    cache = np.load(cache_path, allow_pickle=True)
    case_ids: list[str] = list(cache["case_ids"])
    embeddings: np.ndarray = cache["mean_embeddings"]  # (N, 768)
    N = len(case_ids)
    case_idx = {cid: i for i, cid in enumerate(case_ids)}

    # --- Load keyword predictions --------------------------------------------
    print(f"Loading keyword predictions: {keyword_csv_path}")
    if not Path(keyword_csv_path).exists():
        print(f"ERROR: keyword predictions not found at {keyword_csv_path}")
        print("Run the keyword scan first:")
        print("  ml/.venv/bin/python3 -m keyword_scan")
        sys.exit(1)

    kw = pd.read_csv(keyword_csv_path)
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
        description="Build GroupClassifier training data from embedding cache and keyword predictions."
    )
    parser.add_argument(
        "--embedding-cache",
        default="ml/data/embedding_cache.npz",
        help="Path to embedding cache npz (default: ml/data/embedding_cache.npz)",
    )
    parser.add_argument(
        "--keyword-csv",
        default="ml/output/diagnoses/keyword_predictions.csv",
        help="Path to keyword_predictions.csv (default: ml/output/diagnoses/keyword_predictions.csv)",
    )
    parser.add_argument(
        "--out",
        default="ml/output/group_training_data.npz",
        help="Output npz path (default: ml/output/group_training_data.npz)",
    )
    args = parser.parse_args()
    build_training_data(args.embedding_cache, args.keyword_csv, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
