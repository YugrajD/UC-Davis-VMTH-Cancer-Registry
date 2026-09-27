"""Stage 3a: per-group LabelPresenceClassifier trainers.

Faithful port of ``ml/training/label_presence/{build_training_pairs,
train}.py``: within-group (case, label) pairs (``labels.label_presence_pairs``),
case-disjoint ``GroupShuffleSplit``, ``WeightedRandomSampler`` +
``BCEWithLogitsLoss``, recall-weighted checkpoint selection. One head per
common group plus the shared "Uncommon" head (union of every merged group's
labels). Legacy already seeded torch + numpy for this trainer;
``recipe.seed_all`` adds python's ``random`` and CUDA.

Negative sampling (``labels.label_presence_pairs``) always uses
``recipe.LP_PAIR_SAMPLING_SEED`` (42), never the run's ``seed`` — see that
constant's docstring: legacy's pair builder took its own hardcoded seed,
independent of the model-training seed, so the pairs are identical across the
L3 3-seed retrain.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import config
from report_mapping import sections
from report_mapping.inference.embedding_cache import EmbeddingCache
from report_mapping.model import backbone as backbone_mod
from report_mapping.model import generation as generation_mod
from report_mapping.model.generation import safe_filename
from report_mapping.model.heads import GroupClassifier, LabelPresenceClassifier
from report_mapping.training import embeddings as embeddings_mod
from report_mapping.training import labels as labels_mod
from report_mapping.training import recipe
from report_mapping.training.case_presence import default_backbone_dir
from taxonomy.taxonomy import TaxonomyLabel, load_labels_taxonomy


class _PairDataset(Dataset):
    def __init__(self, report_embs: np.ndarray, label_embs: np.ndarray, targets: np.ndarray):
        self.r = torch.from_numpy(report_embs)
        self.l = torch.from_numpy(label_embs)
        self.t = torch.from_numpy(targets)

    def __len__(self) -> int:
        return len(self.t)

    def __getitem__(self, idx: int):
        return self.r[idx], self.l[idx], self.t[idx]


def _evaluate(model: LabelPresenceClassifier, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for r, l, t in loader:
            probs = torch.sigmoid(model(r.to(device), l.to(device))).cpu().numpy()
            preds.extend((probs >= 0.5).astype(int).tolist())
            trues.extend(t.numpy().astype(int).tolist())
    p, r_, f1, _ = precision_recall_fscore_support(trues, preds, average="binary", zero_division=0)
    return {"precision": float(p), "recall": float(r_), "f1": float(f1)}


def _read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def train_one_group(
    labels_train: pd.DataFrame,
    group_name: str,
    taxonomy_labels: list[TaxonomyLabel],
    cache: EmbeddingCache,
    seed: int,
    device: torch.device,
    out_path: str | Path,
    *,
    uncommon_group_names: list[str] | None = None,
) -> dict:
    """Train one group's LabelPresenceClassifier. Returns
    ``{"best_score": ..., "initial_loss": ..., "final_loss": ...}``.
    ``best_score`` is 0.0 (and no checkpoint written, no loss keys) when
    skipped: fewer than 2 taxonomy labels in the group, no train-partition
    annotations, or fewer than 10 pairs found in the cache — mirrors legacy's
    skip conditions."""
    r = recipe.LABEL_PRESENCE
    recipe.seed_all(seed)

    pairs = labels_mod.label_presence_pairs(
        labels_train, taxonomy_labels, group_name, uncommon_group_names=uncommon_group_names,
        negs_per_pos=r.negs_per_pos, seed=recipe.LP_PAIR_SAMPLING_SEED,
    )
    if pairs.empty:
        return {"best_score": 0.0, "initial_loss": None, "final_loss": None}

    case_id_to_idx = {cid: i for i, cid in enumerate(cache.case_ids)}
    label_text_to_idx = {t: i for i, t in enumerate(cache.label_texts)}
    concat = cache.col_embeddings[sections.CONCAT_3_KEY]
    label_embs_all = cache.label_embeddings

    report_list, label_list, target_list, case_id_list = [], [], [], []
    for row in pairs.itertuples(index=False):
        cidx = case_id_to_idx.get(row.case_id)
        lidx = label_text_to_idx.get(f"{row.label_term} {row.label_group}")
        if cidx is None or lidx is None:
            continue
        report_list.append(concat[cidx])
        label_list.append(label_embs_all[lidx])
        target_list.append(float(row.target))
        case_id_list.append(row.case_id)

    if len(target_list) < 10:
        return {"best_score": 0.0, "initial_loss": None, "final_loss": None}

    report_embs = np.array(report_list, dtype=np.float32)
    label_embs = np.array(label_list, dtype=np.float32)
    targets = np.array(target_list, dtype=np.float32)
    case_id_arr = np.array(case_id_list)

    splitter = GroupShuffleSplit(n_splits=1, test_size=r.val_split, random_state=seed)
    train_idx, val_idx = next(splitter.split(report_embs, targets, groups=case_id_arr))

    train_ds = _PairDataset(report_embs[train_idx], label_embs[train_idx], targets[train_idx])
    val_ds = _PairDataset(report_embs[val_idx], label_embs[val_idx], targets[val_idx])

    train_targets = targets[train_idx]
    n_pos = train_targets.sum()
    n_neg = len(train_targets) - n_pos
    sample_weights = np.where(train_targets == 1, n_neg / max(n_pos, 1), 1.0)
    sampler = WeightedRandomSampler(sample_weights.tolist(), len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=r.batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=r.batch_size, shuffle=False)

    model = LabelPresenceClassifier(
        emb_dim=label_embs_all.shape[1], hidden_dim=r.hidden_dim, dropout=r.dropout,
        n_cols=r.n_cols, col_pair_mode=r.col_pair_mode, col_combine=r.col_combine,
    ).to(device)
    pw = torch.tensor([r.pos_weight], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = torch.optim.AdamW(model.parameters(), lr=r.lr, weight_decay=r.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=r.epochs)

    out_path = Path(out_path)
    model.eval()
    with torch.no_grad():
        initial_loss = criterion(model(train_ds.r.to(device), train_ds.l.to(device)), train_ds.t.to(device)).item()

    best_score = -1.0
    last_loss = float("nan")
    for _epoch in range(1, r.epochs + 1):
        model.train()
        total_loss = 0.0
        for r_emb, l_emb, t in train_loader:
            r_emb, l_emb, t = r_emb.to(device), l_emb.to(device), t.to(device)
            optimizer.zero_grad()
            loss = criterion(model(r_emb, l_emb), t)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(t)
        scheduler.step()
        last_loss = total_loss / len(train_ds)
        m = _evaluate(model, val_loader, device)
        score = (1 - r.recall_weight) * m["precision"] + r.recall_weight * m["recall"]
        if score > best_score:
            best_score = score
            model.save(out_path)

    model.eval()
    with torch.no_grad():
        final_loss = criterion(model(train_ds.r.to(device), train_ds.l.to(device)), train_ds.t.to(device)).item()

    print(f"label_presence[{group_name}]: best score {best_score:.3f}, "
          f"train loss {initial_loss:.4f} -> {final_loss:.4f} (last epoch avg {last_loss:.4f}) -> {out_path.name}")
    return {"best_score": best_score, "initial_loss": initial_loss, "final_loss": final_loss}


def train(
    labels: pd.DataFrame,
    split_id: str,
    seed: int,
    device: str,
    *,
    backbone_dir: str | None = None,
    local_only: bool = True,
    out_dir: str | Path | None = None,
    group_names: list[str] | None = None,
    uncommon_groups: list[str] | None = None,
) -> dict:
    """Train every group's LabelPresenceClassifier (+ the merged Uncommon head,
    if any) on ``labels`` filtered to ``split_id``'s train partition.

    ``group_names`` / ``uncommon_groups`` default to this candidate's own
    ``checkpoints/group_classifier_best.pt`` + ``uncommon_groups.txt`` — the
    ``--stage heads`` path, where group training already ran this cycle.
    """
    out_dir = Path(out_dir) if out_dir is not None else config.REPORT_MAPPING_CANDIDATE_DIR
    backbone_dir = backbone_dir if backbone_dir is not None else default_backbone_dir()
    paths = generation_mod.generation_paths(out_dir)

    if group_names is None:
        _, group_names = GroupClassifier.load(paths.group_pt)
    if uncommon_groups is None:
        uncommon_groups = _read_lines(paths.uncommon_groups_txt)

    taxonomy_labels = load_labels_taxonomy(str(config.LABELS_CSV))
    dev = backbone_mod.device_from_arg(device)
    cache = embeddings_mod.get_or_build(
        backbone_dir, str(config.LABELS_CSV), taxonomy_labels, local_only=local_only, device=dev,
    )
    labels_train = labels_mod.select_train(labels, split_id)

    paths.label_presence_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    for group_name in group_names:
        if group_name == "Uncommon":
            continue
        out_pt = paths.label_presence_dir / f"{safe_filename(group_name)}.pt"
        results[group_name] = train_one_group(labels_train, group_name, taxonomy_labels, cache, seed, dev, out_pt)

    if "Uncommon" in group_names and uncommon_groups:
        out_pt = paths.label_presence_dir / f"{safe_filename('Uncommon')}.pt"
        results["Uncommon"] = train_one_group(
            labels_train, "Uncommon", taxonomy_labels, cache, seed, dev, out_pt,
            uncommon_group_names=uncommon_groups,
        )

    scores = {name: result["best_score"] for name, result in results.items()}
    trained = sum(1 for s in scores.values() if s > 0)
    return {"trained": trained, "skipped": len(scores) - trained, "scores": scores, "results": results}
