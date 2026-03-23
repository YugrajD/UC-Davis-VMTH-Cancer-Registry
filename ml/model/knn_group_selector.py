"""K-nearest-neighbor group selector for the PetBERT pipeline.

Replaces the GroupClassifier with a retrieval-based approach: instead of
training a multi-label MLP, find the K most similar confirmed cancer cases
in a reference set and vote on their groups.

No training required — the reference embeddings and labels come directly from
the embedding cache and LLM (or keyword) predictions.  The reference set
automatically improves as more cases are confirmed, with zero retraining.

Interface mirrors GroupClassifier.predict_proba() so the pipeline can use
either interchangeably.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class KnnGroupSelector:
    """K-NN group selector using cosine similarity on report embeddings.

    Each query report finds its K nearest neighbours among confirmed cancer
    cases and returns the fraction of those neighbours belonging to each group
    as a "probability" array of shape (num_groups,).  These vote fractions are
    compatible with the threshold-based logic in run_categorization_group().

    Attributes:
        ref_embeddings:    (R, D) float32, L2-normalised reference embeddings.
        ref_group_indices: (R,)   int32, group index for each reference entry.
        group_names:       list of G group name strings.
        k:                 number of neighbours to use for voting.
    """

    def __init__(
        self,
        ref_embeddings: np.ndarray,
        ref_group_indices: np.ndarray,
        group_names: list[str],
        k: int = 10,
    ) -> None:
        self.ref_embeddings = ref_embeddings.astype(np.float32)
        self.ref_group_indices = ref_group_indices.astype(np.int32)
        self.group_names = group_names
        self.num_groups = len(group_names)
        self.k = k

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(
        self,
        query_embeddings: np.ndarray,
        batch_size: int = 512,
    ) -> np.ndarray:
        """Return (N, G) group vote-fraction matrix.

        Each row sums to 1.0 across groups (i.e. it is a proper distribution
        over the K neighbours, not independent per-group probabilities).

        Args:
            query_embeddings: (N, D) float32 report embeddings (need not be
                normalised; they are normalised internally).
            batch_size: number of query rows to process at once.

        Returns:
            (N, G) float32 array of vote fractions.
        """
        norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        query_norm = query_embeddings / np.maximum(norms, 1e-8)

        N = query_norm.shape[0]
        k = min(self.k, self.ref_embeddings.shape[0])
        probs = np.zeros((N, self.num_groups), dtype=np.float32)

        for start in range(0, N, batch_size):
            batch = query_norm[start : start + batch_size]           # (B, D)
            sims = batch @ self.ref_embeddings.T                     # (B, R)

            # argpartition(-sims, k) gives indices of the k largest sims
            if sims.shape[1] <= k:
                top_k_idx = np.tile(np.arange(sims.shape[1]), (sims.shape[0], 1))
            else:
                top_k_idx = np.argpartition(sims, -k, axis=1)[:, -k:]  # (B, k)

            for b_i in range(len(batch)):
                neighbour_groups = self.ref_group_indices[top_k_idx[b_i]]
                for g in neighbour_groups:
                    probs[start + b_i, g] += 1.0

        probs /= k
        return probs

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialise reference data to an npz file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            ref_embeddings=self.ref_embeddings,
            ref_group_indices=self.ref_group_indices,
            group_names=np.array(self.group_names, dtype=object),
            k=np.array(self.k),
        )
        print(f"Saved KnnGroupSelector: {path}")

    @classmethod
    def load(cls, path: str | Path) -> "KnnGroupSelector":
        """Load a saved KnnGroupSelector from an npz file."""
        data = np.load(path, allow_pickle=True)
        return cls(
            ref_embeddings=data["ref_embeddings"],
            ref_group_indices=data["ref_group_indices"],
            group_names=list(data["group_names"]),
            k=int(data["k"]),
        )

    # ------------------------------------------------------------------
    # Construction from embedding cache
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        cache_path: str | Path,
        labels_csv_path: str | Path,
        k: int = 10,
        per_column: bool = True,
        min_group_cases: int = 10,
    ) -> "KnnGroupSelector":
        """Build a KnnGroupSelector from the embedding cache and predictions CSV.

        Args:
            cache_path:       Path to embedding_cache.npz.
            labels_csv_path:  Path to a predictions CSV with ``case_id`` and
                              ``matched_group`` columns (LLM or keyword output).
            k:                Number of nearest neighbours to vote with.
            per_column:       If True (default), use per-column concat embeddings
                              (3 × 768 = 2304-dim); otherwise use mean (768-dim).
                              Must match what will be passed at inference time.
            min_group_cases:  Drop groups with fewer confirmed unique cases.

        Returns:
            A ready-to-use KnnGroupSelector instance.
        """
        import pandas as pd

        print(f"Loading embedding cache: {cache_path}")
        cache = np.load(cache_path, allow_pickle=True)
        case_ids = list(cache["case_ids"])
        case_idx = {cid: i for i, cid in enumerate(case_ids)}

        if per_column:
            col_keys = [
                "col_FINAL_COMMENT",
                "col_HISTOPATHOLOGICAL_SUMMARY",
                "col_ANCILLARY_TESTS",
            ]
            all_embeddings = np.concatenate(
                [cache[ck] for ck in col_keys], axis=1
            ).astype(np.float32)
            print(f"Using per-column embeddings: {all_embeddings.shape[1]}-dim")
        else:
            all_embeddings = cache["mean_embeddings"].astype(np.float32)
            print(f"Using mean embeddings: {all_embeddings.shape[1]}-dim")

        print(f"Loading labels: {labels_csv_path}")
        df = pd.read_csv(labels_csv_path)
        df = df[df["matched_group"].notna() & (df["matched_group"] != "")].copy()

        # Drop sparse groups
        group_counts = df.groupby("matched_group")["case_id"].nunique()
        valid_groups = sorted(
            group_counts[group_counts >= min_group_cases].index.tolist()
        )
        dropped = sorted(set(df["matched_group"].unique()) - set(valid_groups))
        if dropped:
            print(f"Dropping {len(dropped)} sparse group(s) (< {min_group_cases} cases): {dropped}")
        group_idx_map = {g: i for i, g in enumerate(valid_groups)}

        # Build reference set — one entry per unique (case_id, group) pair
        ref_embs: list[np.ndarray] = []
        ref_groups: list[int] = []
        seen: set[tuple[str, str]] = set()

        for _, row in df.iterrows():
            cid = str(row["case_id"])
            group = row["matched_group"]
            if cid not in case_idx or group not in group_idx_map:
                continue
            key = (cid, group)
            if key in seen:
                continue
            seen.add(key)
            ref_embs.append(all_embeddings[case_idx[cid]])
            ref_groups.append(group_idx_map[group])

        ref_embs_arr = np.stack(ref_embs, axis=0)                    # (R, D)
        ref_groups_arr = np.array(ref_groups, dtype=np.int32)

        # L2-normalise reference embeddings once at build time
        norms = np.linalg.norm(ref_embs_arr, axis=1, keepdims=True)
        ref_embs_arr /= np.maximum(norms, 1e-8)

        print(
            f"Reference set: {len(ref_embs_arr)} entries, "
            f"{len(valid_groups)} groups, k={k}, dim={ref_embs_arr.shape[1]}"
        )
        return cls(
            ref_embeddings=ref_embs_arr,
            ref_group_indices=ref_groups_arr,
            group_names=valid_groups,
            k=k,
        )
