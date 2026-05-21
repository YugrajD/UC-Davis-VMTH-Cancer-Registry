"""Build training data for the GroupClassifier.

Reads:
  - ml/output/training/embedding_cache.npz       -- concat-3 report embeddings and case_ids
  - ml/output/annotation/annotation.csv          -- ground-truth (case_id, matched_group)

Produces:
  - ml/output/training/group/group_training_data.npz with:
      embeddings    (N, 2304) float32 -- concat-3 report embedding per case
      targets       (N, G)    float32 -- multi-hot group labels (0.0 = non-cancer, 1.0 = present)
      case_ids      (N,)     object   -- case_id strings
      group_names   (G,)     object   -- group name strings (index = column in targets)
      class_weights (G,)     float32  -- inverse-frequency weights for BCEWithLogitsLoss

Ground-truth assumption: cases in the cache that do not appear in the labels CSV
are treated as non-cancer (all-zeros target). This is valid for a general veterinary
clinic population where ~18% cancer prevalence is expected.

Usage:
  python ml/training/group/build_training_data.py
  python ml/training/group/build_training_data.py \\
      --embedding-cache ml/output/training/embedding_cache.npz \\
      --expectation-csv ml/output/annotation/annotation.csv \\
      --train-cases ml/output/splits/train_cases.txt \\
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
    train_cases_txt: str = "",
    uncommon_threshold: int = 200,
    uncommon_groups_out: str = "",
    excluded_groups: list[str] | None = None,
) -> None:
    # --- Load embedding cache ------------------------------------------------
    print(f"Loading embedding cache: {cache_path}")
    if not Path(cache_path).exists():
        print(f"ERROR: cache not found at {cache_path}")
        print("Run the PetBERT scan first to generate the embedding cache:")
        print("  ml/.venv/Scripts/python.exe ml/scripts/run_production.py --local-only")
        sys.exit(1)

    cache = np.load(cache_path, allow_pickle=True)
    all_case_ids: list[str] = list(cache["case_ids"])
    all_embeddings: np.ndarray = cache["col_concat_3"]  # (N, 2304)

    # --- Filter to train cases if a split is active --------------------------
    if train_cases_txt:
        with open(train_cases_txt, encoding="utf-8") as f:
            train_ids = {line.strip() for line in f if line.strip()}
        keep = np.array([cid in train_ids for cid in all_case_ids])
        case_ids = [cid for cid, k in zip(all_case_ids, keep) if k]
        embeddings = all_embeddings[keep]
        print(f"Train-cases filter: {len(case_ids)}/{len(all_case_ids)} cases kept")
    else:
        case_ids = all_case_ids
        embeddings = all_embeddings

    N = len(case_ids)
    case_idx = {cid: i for i, cid in enumerate(case_ids)}
    print(f"Embeddings: {embeddings.shape[1]}-dim concat-3")

    # --- Load keyword predictions --------------------------------------------
    print(f"Loading keyword predictions: {expectation_csv_path}")
    if not Path(expectation_csv_path).exists():
        print(f"ERROR: keyword predictions not found at {expectation_csv_path}")
        print("Run the keyword scan first:")
        print("  ml/.venv/bin/python3 -m keyword_scan")
        sys.exit(1)

    kw = pd.read_csv(expectation_csv_path)
    kw = kw[kw["matched_group"].notna()].copy()

    # --- Build sorted group list, split into common and uncommon -------------
    all_groups: list[str] = sorted(kw["matched_group"].unique())
    print(f"Found {len(all_groups)} cancer groups across {kw['case_id'].nunique()} keyword-confirmed cases")

    force_uncommon: set[str] = set(excluded_groups) if excluded_groups else set()

    if uncommon_threshold > 0:
        in_cache = kw[kw["case_id"].astype(str).isin(case_idx)]
        per_group = in_cache["matched_group"].value_counts()
        common_groups = sorted(
            g for g in all_groups
            if per_group.get(g, 0) >= uncommon_threshold and g not in force_uncommon
        )
        uncommon_group_names = sorted(
            g for g in all_groups
            if per_group.get(g, 0) < uncommon_threshold or g in force_uncommon
        )
    else:
        in_cache = kw[kw["case_id"].astype(str).isin(case_idx)]
        per_group = in_cache["matched_group"].value_counts()
        common_groups = sorted(g for g in all_groups if g not in force_uncommon)
        uncommon_group_names = sorted(force_uncommon & set(all_groups))

    has_uncommon = len(uncommon_group_names) > 0
    final_groups: list[str] = common_groups + (["Uncommon"] if has_uncommon else [])
    G = len(final_groups)
    group_idx = {g: i for i, g in enumerate(final_groups)}

    if has_uncommon:
        threshold_moved = [g for g in uncommon_group_names if per_group.get(g, 0) < uncommon_threshold]
        force_moved = [g for g in uncommon_group_names if g in force_uncommon]
        print(f"  Common groups  (>= {uncommon_threshold} cases): {len(common_groups)}")
        if threshold_moved:
            print(f"  Below threshold (< {uncommon_threshold} cases): {len(threshold_moved)} -> merged into 'Uncommon'")
            for g in threshold_moved:
                print(f"    - {g} ({per_group.get(g, 0)} cases)")
        if force_moved:
            print(f"  Excluded groups (forced into 'Uncommon'): {len(force_moved)}")
            for g in force_moved:
                print(f"    - {g} ({per_group.get(g, 0)} cases)")

    # --- Build multi-hot targets (N, G) — default all zeros (non-cancer) ----
    targets = np.zeros((N, G), dtype=np.float32)
    uncommon_idx = group_idx.get("Uncommon")
    skipped = 0
    for _, row in kw.iterrows():
        cid = str(row["case_id"])
        group = row["matched_group"]
        if cid not in case_idx:
            skipped += 1
            continue
        if group in group_idx:
            targets[case_idx[cid], group_idx[group]] = 1.0
        elif uncommon_idx is not None:
            targets[case_idx[cid], uncommon_idx] = 1.0

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
    for g, group_name in enumerate(final_groups):
        print(f"{group_name:<50} {int(positive_counts[g]):>6} {class_weights[g]:>7.2f}")

    # --- Write uncommon group list -------------------------------------------
    if has_uncommon and uncommon_groups_out:
        Path(uncommon_groups_out).parent.mkdir(parents=True, exist_ok=True)
        with open(uncommon_groups_out, "w", encoding="utf-8") as f:
            f.write("\n".join(uncommon_group_names) + "\n")
        print(f"\nUncommon groups list: {uncommon_groups_out} ({len(uncommon_group_names)} groups)")

    # --- Save ----------------------------------------------------------------
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        embeddings=embeddings,
        targets=targets,
        case_ids=np.array(case_ids, dtype=object),
        group_names=np.array(final_groups, dtype=object),
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
        default=config.ANNOTATION_CSV,
        help="Path to predictions CSV with case_id and matched_group columns",
    )
    parser.add_argument(
        "--out",
        default=config.GROUP_TRAINING_DATA_NPZ,
        help=f"Output npz path (default: {config.GROUP_TRAINING_DATA_NPZ})",
    )
    parser.add_argument(
        "--train-cases",
        default="",
        help="Path to train_cases.txt. When provided, only those case IDs are included. "
             "Generate with ml/training/data/create_split.py.",
    )
    parser.add_argument(
        "--uncommon-threshold",
        type=int,
        default=200,
        help="Groups with fewer cases than this are merged into a single 'Uncommon' output "
             "class. Set to 0 to disable and keep all groups separate (default: 200).",
    )
    parser.add_argument(
        "--uncommon-groups-out",
        default=config.UNCOMMON_GROUPS_TXT,
        help=f"Path to write the uncommon group names list (default: {config.UNCOMMON_GROUPS_TXT})",
    )
    parser.add_argument(
        "--excluded-groups",
        default="Neoplasms, NOS",
        help="Pipe-separated group names to force into the 'Uncommon' bucket regardless of "
             "case count (use | not comma, as group names contain commas). "
             "Default: 'Neoplasms, NOS'.",
    )
    args = parser.parse_args()
    excluded = [g.strip() for g in args.excluded_groups.split("|")] if args.excluded_groups else []
    build_training_data(
        args.embedding_cache, args.expectation_csv, args.out, args.train_cases,
        uncommon_threshold=args.uncommon_threshold,
        uncommon_groups_out=args.uncommon_groups_out,
        excluded_groups=excluded,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
