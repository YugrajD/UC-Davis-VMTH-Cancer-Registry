"""k-fold heads-only out-of-fold (OOF) predictions for train cases, on the
fixed embedding cache.

**Caveat**: the backbone already saw every train case during its own
contrastive adaptation (it is not retrained per fold), so a case's
"held-out" status here only applies to the *head* trained on top of
embeddings that already reflect that case's own (report, label) pairs. OOF
error rates are therefore optimistic relative to a genuinely unseen case's
error rate — useful as calibration input (WP5b), not as a substitute for
calibration/test scores.

Reuses each trainer's ``train_on_case_ids`` (``case_presence.py``,
``group.py``): the same code path as full training, called once per fold
against a case-id subset instead of the whole train partition, sharing one
embedding cache across folds so the backbone is loaded/embedded only once.

Only the gate and group heads have OOF support here; per-group
LabelPresenceClassifier OOF would follow the identical pattern (fold ->
``label_presence.train_one_group`` on the fold's case subset) but is not
implemented in this WP — flag it to WP5b if per-LP OOF scores turn out to be
needed for calibration.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold

import config
from generations.splits import load_split
from report_mapping import sections
from report_mapping.model import backbone as backbone_mod
from report_mapping.model.heads import CasePresenceClassifier, GroupClassifier
from report_mapping.training import case_presence as case_presence_mod
from report_mapping.training import embeddings as embeddings_mod
from report_mapping.training import group as group_mod
from report_mapping.training import labels as labels_mod
from report_mapping.training import recipe
from report_mapping.training.case_presence import default_backbone_dir
from taxonomy.taxonomy import load_labels_taxonomy

DEFAULT_K = 5


@dataclass(frozen=True)
class CasePresenceOOF:
    case_ids: list[str]
    probs: np.ndarray     # (N,) out-of-fold case_presence probability
    targets: np.ndarray   # (N,) gate target (1.0 cancer, 0.0 non-cancer)


@dataclass(frozen=True)
class GroupOOF:
    case_ids: list[str]
    group_names: list[str]
    probs: np.ndarray     # (N, G) out-of-fold per-group probability
    targets: np.ndarray   # (N, G) multi-hot target


def _fold_indices(n: int, k: int, seed: int):
    return KFold(n_splits=k, shuffle=True, random_state=seed).split(np.arange(n))


def _prepare(labels: pd.DataFrame, split_id: str, backbone_dir: str | None, local_only: bool, device: str):
    backbone_dir = backbone_dir if backbone_dir is not None else default_backbone_dir()
    taxonomy_labels = load_labels_taxonomy(str(config.LABELS_CSV))
    dev = backbone_mod.device_from_arg(device)
    cache = embeddings_mod.get_or_build(
        backbone_dir, str(config.LABELS_CSV), taxonomy_labels, local_only=local_only, device=dev,
    )
    split = load_split(split_id)
    labels_train = labels_mod.select_train(labels, split_id)
    case_ids = sorted(labels_mod.training_case_ids(labels_train, cache.case_ids, split))
    return cache, labels_train, case_ids, dev


def run_case_presence_oof(
    labels: pd.DataFrame,
    split_id: str,
    seed: int,
    device: str,
    *,
    k: int = DEFAULT_K,
    backbone_dir: str | None = None,
    local_only: bool = True,
) -> CasePresenceOOF:
    """Gate OOF probabilities for every train case with an embedding."""
    cache, labels_train, case_ids, dev = _prepare(labels, split_id, backbone_dir, local_only, device)
    ids_arr = np.array(case_ids)
    targets = labels_mod.gate_targets(labels_train, case_ids)

    probs = np.zeros(len(case_ids), dtype=np.float32)
    cache_index = {cid: i for i, cid in enumerate(cache.case_ids)}
    with tempfile.TemporaryDirectory(prefix="oof_case_presence_") as tmp:
        tmp_path = Path(tmp)
        for fold, (train_pos, val_pos) in enumerate(_fold_indices(len(case_ids), k, seed)):
            checkpoint = tmp_path / f"fold_{fold}.pt"
            case_presence_mod.train_on_case_ids(
                labels_train, ids_arr[train_pos].tolist(), cache, seed, dev, checkpoint,
            )
            model = CasePresenceClassifier.load(checkpoint)
            val_ids = ids_arr[val_pos]
            idxs = [cache_index[cid] for cid in val_ids]
            embs = cache.col_embeddings[sections.CONCAT_3_KEY][idxs].astype(np.float32)
            probs[val_pos] = model.predict_proba(torch.from_numpy(embs)).numpy()

    return CasePresenceOOF(case_ids=case_ids, probs=probs, targets=targets)


def run_group_oof(
    labels: pd.DataFrame,
    split_id: str,
    seed: int,
    device: str,
    *,
    k: int = DEFAULT_K,
    backbone_dir: str | None = None,
    local_only: bool = True,
) -> GroupOOF:
    """Group OOF probabilities for every train case with an embedding.

    Group names / the Uncommon merge are fixed from the *whole* train
    partition (this function's own reference call to ``labels.group_targets``)
    so every fold's predictions land in the same columns; a fold's own
    ``group.train_on_case_ids`` call may merge a slightly different group set
    from its smaller case universe; predictions for a group missing from a
    fold's own head stay 0 for that fold's held-out cases.
    """
    cache, labels_train, case_ids, dev = _prepare(labels, split_id, backbone_dir, local_only, device)
    ids_arr = np.array(case_ids)

    reference = labels_mod.group_targets(
        labels_train, case_ids,
        uncommon_threshold=recipe.GROUP.uncommon_threshold, forced_uncommon=recipe.GROUP.excluded_groups,
    )
    group_names = reference.group_names
    group_index = {g: i for i, g in enumerate(group_names)}
    targets = reference.targets

    probs = np.zeros((len(case_ids), len(group_names)), dtype=np.float32)
    cache_index = {cid: i for i, cid in enumerate(cache.case_ids)}
    with tempfile.TemporaryDirectory(prefix="oof_group_") as tmp:
        tmp_path = Path(tmp)
        for fold, (train_pos, val_pos) in enumerate(_fold_indices(len(case_ids), k, seed)):
            checkpoint = tmp_path / f"fold_{fold}.pt"
            group_mod.train_on_case_ids(
                labels_train, ids_arr[train_pos].tolist(), cache, seed, dev, checkpoint, None,
            )
            model, fold_group_names = GroupClassifier.load(checkpoint)
            val_ids = ids_arr[val_pos]
            idxs = [cache_index[cid] for cid in val_ids]
            embs = cache.col_embeddings[sections.CONCAT_3_KEY][idxs].astype(np.float32)
            fold_probs = model.predict_proba(torch.from_numpy(embs)).numpy()
            for j, name in enumerate(fold_group_names):
                gi = group_index.get(name)
                if gi is not None:
                    probs[val_pos, gi] = fold_probs[:, j]

    return GroupOOF(case_ids=case_ids, group_names=group_names, probs=probs, targets=targets)
