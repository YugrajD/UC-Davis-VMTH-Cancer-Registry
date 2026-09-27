"""Stage 2: GroupClassifier trainer.

Faithful port of ``ml/training/group/{build_training_data,train}.py``: the
same stratified-by-first-positive-group train/val split, sparse-group
dropping (``min_group_cases``) before training, ``max_class_weight`` cap on
BCE ``pos_weight``, and "save whenever this run's val macro F1 improves"
(legacy also tracked a cross-run best via ``group_classifier_best.pt`` +
``.meta.json`` sitting beside many local experiment runs; a candidate
directory has no such history to compare against, so this trainer always
keeps the run's own best epoch). Legacy seeded only a local
``np.random.default_rng`` for the stratified split, not the global RNG state
or torch at all; ``recipe.seed_all`` seeds everything.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import config
from generations.splits import load_split
from report_mapping import sections
from report_mapping.inference.embedding_cache import EmbeddingCache
from report_mapping.model import backbone as backbone_mod
from report_mapping.model import generation as generation_mod
from report_mapping.model.heads import GroupClassifier
from report_mapping.training import embeddings as embeddings_mod
from report_mapping.training import labels as labels_mod
from report_mapping.training import recipe
from report_mapping.training.case_presence import default_backbone_dir
from taxonomy.taxonomy import load_labels_taxonomy


def _stratified_split(targets: np.ndarray, val_frac: float, seed: int) -> tuple[list[int], list[int]]:
    """Each case's stratum is its first positive group's column (-1 if none);
    ``val_frac`` of each stratum, shuffled, is held out."""
    rng = np.random.default_rng(seed)
    strata = np.array([int(np.where(row > 0)[0][0]) if row.sum() > 0 else -1 for row in targets])
    train_indices: list[int] = []
    val_indices: list[int] = []
    for stratum in np.unique(strata):
        idx = np.where(strata == stratum)[0]
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_frac))
        val_indices.extend(idx[:n_val].tolist())
        train_indices.extend(idx[n_val:].tolist())
    return train_indices, val_indices


def _macro_f1(probs: torch.Tensor, targets: torch.Tensor, threshold: float) -> float:
    preds = (probs >= threshold).float()
    f1s = []
    for g in range(probs.shape[1]):
        if targets[:, g].sum().item() == 0:
            continue
        tp = (preds[:, g] * targets[:, g]).sum().item()
        fp = (preds[:, g] * (1 - targets[:, g])).sum().item()
        fn = ((1 - preds[:, g]) * targets[:, g]).sum().item()
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2 * p * rc / (p + rc) if (p + rc) > 0 else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def train_on_case_ids(
    labels_train: pd.DataFrame,
    case_ids: list[str],
    cache: EmbeddingCache,
    seed: int,
    device: torch.device,
    checkpoint_path: str | Path,
    uncommon_groups_path: str | Path | None,
) -> dict:
    """Train the GroupClassifier against an explicit case-id universe and an
    already-built cache — the reusable core behind ``train()`` and ``oof.py``.
    Writes ``uncommon_groups.txt`` when ``uncommon_groups_path`` is given and
    any group merged (mirrors legacy: no file when there's nothing to merge)."""
    r = recipe.GROUP
    recipe.seed_all(seed)

    gt = labels_mod.group_targets(
        labels_train, case_ids, uncommon_threshold=r.uncommon_threshold, forced_uncommon=r.excluded_groups,
    )
    cache_index = {cid: i for i, cid in enumerate(cache.case_ids)}
    idxs = [cache_index[cid] for cid in case_ids]
    embeddings = cache.col_embeddings[sections.CONCAT_3_KEY][idxs].astype(np.float32)

    targets_t = torch.from_numpy(gt.targets)
    group_names = list(gt.group_names)
    class_weights = torch.from_numpy(gt.class_weights)

    if r.min_group_cases > 0:
        positive_counts = targets_t.sum(dim=0)
        keep_mask = positive_counts >= r.min_group_cases
        targets_t = targets_t[:, keep_mask]
        class_weights = class_weights[keep_mask]
        group_names = [g for g, k in zip(group_names, keep_mask.tolist()) if k]
    if r.max_class_weight > 0:
        class_weights = class_weights.clamp(max=r.max_class_weight)

    embeddings_t = torch.from_numpy(embeddings)
    G = len(group_names)

    train_idx, val_idx = _stratified_split(targets_t.numpy(), r.val_frac, seed)
    train_ds = TensorDataset(embeddings_t[train_idx], targets_t[train_idx])
    val_ds = TensorDataset(embeddings_t[val_idx], targets_t[val_idx])
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    model = GroupClassifier(num_groups=G, emb_dim=embeddings_t.shape[1], hidden_dim=r.hidden_dim,
                             dropout=r.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=r.lr, weight_decay=r.weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights.to(device))
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=1)
        if r.lr_schedule == "cosine" else None
    )

    checkpoint_path = Path(checkpoint_path)
    model.eval()
    with torch.no_grad():
        train_emb, train_tgt = train_ds.tensors
        initial_loss = criterion(model.net(train_emb.to(device)), train_tgt.to(device)).item()

    best_f1 = -1.0
    best_epoch = 0
    last_loss = float("nan")
    for epoch in range(1, r.epochs + 1):
        model.train()
        total_loss = 0.0
        for emb_batch, tgt_batch in train_loader:
            emb_batch, tgt_batch = emb_batch.to(device), tgt_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model.net(emb_batch), tgt_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(emb_batch)
        last_loss = total_loss / len(train_ds)

        model.eval()
        all_probs, all_targets = [], []
        with torch.no_grad():
            for emb_batch, tgt_batch in val_loader:
                logits = model.net(emb_batch.to(device))
                all_probs.append(torch.sigmoid(logits).cpu())
                all_targets.append(tgt_batch)
        macro_f1 = _macro_f1(torch.cat(all_probs), torch.cat(all_targets), r.threshold)
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_epoch = epoch
            model.save(checkpoint_path, group_names)
        if scheduler is not None:
            scheduler.step()

    if uncommon_groups_path is not None and gt.uncommon_groups:
        uncommon_groups_path = Path(uncommon_groups_path)
        uncommon_groups_path.parent.mkdir(parents=True, exist_ok=True)
        uncommon_groups_path.write_text("\n".join(gt.uncommon_groups) + "\n", encoding="utf-8", newline="\n")

    model.eval()
    with torch.no_grad():
        final_loss = criterion(model.net(train_emb.to(device)), train_tgt.to(device)).item()

    print(f"group: best macro F1 {best_f1:.4f} (epoch {best_epoch}/{r.epochs}), "
          f"{G} groups, train loss {initial_loss:.4f} -> {final_loss:.4f} (last epoch avg {last_loss:.4f})")
    return {
        "best_f1": best_f1, "best_epoch": best_epoch,
        "initial_loss": initial_loss, "final_loss": final_loss,
        "group_names": group_names, "uncommon_groups": gt.uncommon_groups, "n_cases": len(case_ids),
    }


def train(
    labels: pd.DataFrame,
    split_id: str,
    seed: int,
    device: str,
    *,
    backbone_dir: str | None = None,
    local_only: bool = True,
    out_dir: str | Path | None = None,
) -> dict:
    """Train the GroupClassifier on ``labels`` filtered to ``split_id``'s train
    partition. Writes ``checkpoints/group_classifier_best.pt`` and (if any
    group merges) ``checkpoints/uncommon_groups.txt`` under ``out_dir``
    (default: ``config.REPORT_MAPPING_CANDIDATE_DIR``)."""
    out_dir = Path(out_dir) if out_dir is not None else config.REPORT_MAPPING_CANDIDATE_DIR
    backbone_dir = backbone_dir if backbone_dir is not None else default_backbone_dir()

    taxonomy_labels = load_labels_taxonomy(str(config.LABELS_CSV))
    dev = backbone_mod.device_from_arg(device)
    cache = embeddings_mod.get_or_build(
        backbone_dir, str(config.LABELS_CSV), taxonomy_labels, local_only=local_only, device=dev,
    )

    split = load_split(split_id)
    labels_train = labels_mod.select_train(labels, split_id)
    case_ids = labels_mod.training_case_ids(labels_train, cache.case_ids, split)

    paths = generation_mod.generation_paths(out_dir)
    return train_on_case_ids(labels_train, case_ids, cache, seed, dev, paths.group_pt, paths.uncommon_groups_txt)
