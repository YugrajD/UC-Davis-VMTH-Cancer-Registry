"""Build within-group training pairs for LabelPresenceClassifier.

For a given ICD group, creates (case, label) pairs where:
  - Positives: annotation-confirmed (case, label) within the group
  - Negatives: same cases paired with other labels in the group (within-group negatives)

Negatives can be mined by cosine similarity to the positive label embedding
(hard-neg mining, QW1) when ``label_embeddings`` is supplied and
``hard_neg_fraction > 0``. Otherwise (or as the remaining fraction) negatives
are drawn uniformly at random from the in-group pool, matching the legacy
behavior.

To train the "Uncommon" model, pass all uncommon group names via
``uncommon_group_names``. The function will treat the union of all their labels
as the candidate pool and all annotations matching any of those groups as positives.

Output CSV columns: case_id, label_term, label_group, target
"""

import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from ICD_labels import load_labels_taxonomy
from production.petbert_pipeline.embedding import cosine_similarity_matrix


def build_label_presence_pairs(
    *,
    annotation_csv: str,
    labels_csv: str,
    out_csv: str,
    group_name: str,
    uncommon_group_names: list[str] | None = None,
    train_cases_txt: str = "",
    within_group_negs_per_pos: int = 5,
    hard_neg_fraction: float = 0.0,
    label_embeddings: dict[str, np.ndarray] | None = None,
    seed: int = 42,
) -> int:
    """Build within-group (case, label) training pairs for one group.

    Args:
        group_name: The ICD group to build pairs for (e.g. "Blood vessel tumors").
                    Pass "Uncommon" to build pairs covering all uncommon groups.
        uncommon_group_names: Required when group_name == "Uncommon". List of group
                              names that are merged into the Uncommon bucket.
        within_group_negs_per_pos: Number of other-label negatives per positive pair.
        hard_neg_fraction: Fraction of negatives mined by cosine similarity to the
                           positive label embedding (top-k). 0.0 = pure random
                           (legacy), 1.0 = pure hard. Ignored when
                           ``label_embeddings`` is None.
        label_embeddings: Mapping ``"{term} {group}"`` -> embedding (np.ndarray).
                          Source: ``embedding_cache.load_cache()``. When provided
                          alongside ``hard_neg_fraction > 0``, negatives are drawn
                          first from the cosine-ranked top-k other labels in the
                          group (excluding the positive), with the remainder
                          drawn at random from what's left.

    Returns:
        Total number of rows written.
    """
    rng = random.Random(seed)
    taxonomy = load_labels_taxonomy(labels_csv)

    if group_name == "Uncommon":
        if not uncommon_group_names:
            raise ValueError("uncommon_group_names must be provided when group_name='Uncommon'")
        target_groups = set(uncommon_group_names)
        labels_in_group = [t for t in taxonomy if t.group in target_groups]
    else:
        labels_in_group = [t for t in taxonomy if t.group == group_name]

    if len(labels_in_group) < 2:
        print(f"  Skipping {group_name!r}: only {len(labels_in_group)} labels (need >= 2 for negatives)")
        return 0

    label_terms_in_group = [t.term for t in labels_in_group]
    label_groups_in_group = [t.group for t in labels_in_group]

    # Load annotation CSV
    ann = pd.read_csv(annotation_csv)
    if group_name == "Uncommon":
        ann_group = ann[ann["matched_group"].isin(target_groups)].copy()
    else:
        ann_group = ann[ann["matched_group"] == group_name].copy()

    if ann_group.empty:
        print(f"  Skipping {group_name!r}: no annotation examples")
        return 0

    # Filter to train cases if provided
    if train_cases_txt:
        with open(train_cases_txt, encoding="utf-8") as f:
            train_ids = {line.strip() for line in f if line.strip()}
        ann_group = ann_group[ann_group["case_id"].astype(str).isin(train_ids)]
    if ann_group.empty:
        print(f"  Skipping {group_name!r}: no annotation examples after train-cases filter")
        return 0

    # Precompute hard-negative rankings once per positive label.
    # hard_rank[(term, group)] = list of (other_term, other_group) ordered by
    # descending cosine similarity to (term, group)'s embedding, self excluded.
    hard_rank: dict[tuple[str, str], list[tuple[str, str]]] = {}
    use_hard_neg = hard_neg_fraction > 0.0 and label_embeddings is not None
    if use_hard_neg:
        in_group_with_emb: list[tuple[str, str]] = []
        emb_rows: list[np.ndarray] = []
        for term, group in zip(label_terms_in_group, label_groups_in_group):
            emb = label_embeddings.get(f"{term} {group}")
            if emb is not None:
                in_group_with_emb.append((term, group))
                emb_rows.append(emb)
        if len(in_group_with_emb) >= 2:
            emb_matrix = np.stack(emb_rows, axis=0)
            sim_matrix = cosine_similarity_matrix(emb_matrix, emb_matrix)
            for i, (term, group) in enumerate(in_group_with_emb):
                order = np.argsort(-sim_matrix[i])
                hard_rank[(term, group)] = [
                    in_group_with_emb[j] for j in order
                    if in_group_with_emb[j][0] != term
                ]
        else:
            use_hard_neg = False  # not enough labels with embeddings; fall back

    rows: list[dict] = []
    n_hard_total = 0
    n_random_total = 0
    for _, row in ann_group.iterrows():
        pos_term = str(row["matched_term"])
        pos_group = str(row["matched_group"])
        case_id = str(row["case_id"])

        rows.append({"case_id": case_id, "label_term": pos_term, "label_group": pos_group, "target": 1})

        # Within-group negatives: other labels in the group
        neg_pool = [
            (t, g) for t, g in zip(label_terms_in_group, label_groups_in_group)
            if t != pos_term
        ]
        if not neg_pool:
            continue

        target_n = min(within_group_negs_per_pos, len(neg_pool))
        chosen: list[tuple[str, str]] = []

        ranked = hard_rank.get((pos_term, pos_group)) if use_hard_neg else None
        if ranked:
            n_hard = min(int(round(target_n * hard_neg_fraction)), len(ranked))
            chosen.extend(ranked[:n_hard])
            n_hard_total += len(chosen)

        chosen_set = set(chosen)
        remaining = [p for p in neg_pool if p not in chosen_set]
        n_random = min(target_n - len(chosen), len(remaining))
        if n_random > 0:
            random_negs = rng.sample(remaining, n_random)
            chosen.extend(random_negs)
            n_random_total += n_random

        for neg_term, neg_group in chosen:
            rows.append({"case_id": case_id, "label_term": neg_term, "label_group": neg_group, "target": 0})

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "label_term", "label_group", "target"])
        writer.writeheader()
        writer.writerows(rows)

    n_pos = sum(1 for r in rows if r["target"] == 1)
    n_neg = len(rows) - n_pos
    if use_hard_neg:
        print(
            f"  {group_name!r}: {n_pos} positives, "
            f"{n_neg} negatives ({n_hard_total} hard / {n_random_total} random) "
            f"-> {out_csv}"
        )
    else:
        print(f"  {group_name!r}: {n_pos} positives, {n_neg} negatives -> {out_csv}")
    return len(rows)
