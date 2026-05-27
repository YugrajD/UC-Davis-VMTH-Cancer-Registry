"""Build within-group training pairs for LabelPresenceClassifier.

For a given ICD group, creates (case, label) pairs where:
  - Positives: annotation-confirmed (case, label) within the group
  - Negatives: same cases paired with other labels in the group, drawn
    uniformly at random from the in-group pool.

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

import pandas as pd

from ICD_labels import load_labels_taxonomy


def build_label_presence_pairs(
    *,
    annotation_csv: str,
    labels_csv: str,
    out_csv: str,
    group_name: str,
    uncommon_group_names: list[str] | None = None,
    train_cases_txt: str = "",
    within_group_negs_per_pos: int = 5,
    seed: int = 42,
) -> int:
    """Build within-group (case, label) training pairs for one group.

    Args:
        group_name: The ICD group to build pairs for (e.g. "Blood vessel tumors").
                    Pass "Uncommon" to build pairs covering all uncommon groups.
        uncommon_group_names: Required when group_name == "Uncommon". List of group
                              names that are merged into the Uncommon bucket.
        within_group_negs_per_pos: Number of other-label negatives per positive pair.

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

    rows: list[dict] = []
    for _, row in ann_group.iterrows():
        pos_term = str(row["matched_term"])
        pos_group = str(row["matched_group"])
        case_id = str(row["case_id"])

        rows.append({"case_id": case_id, "label_term": pos_term, "label_group": pos_group, "target": 1})

        # Within-group negatives: other labels in the group, sampled at random
        neg_pool = [
            (t, g) for t, g in zip(label_terms_in_group, label_groups_in_group)
            if t != pos_term
        ]
        if not neg_pool:
            continue

        n_neg = min(within_group_negs_per_pos, len(neg_pool))
        for neg_term, neg_group in rng.sample(neg_pool, n_neg):
            rows.append({"case_id": case_id, "label_term": neg_term, "label_group": neg_group, "target": 0})

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "label_term", "label_group", "target"])
        writer.writeheader()
        writer.writerows(rows)

    n_pos = sum(1 for r in rows if r["target"] == 1)
    n_neg = len(rows) - n_pos
    print(f"  {group_name!r}: {n_pos} positives, {n_neg} negatives -> {out_csv}")
    return len(rows)
