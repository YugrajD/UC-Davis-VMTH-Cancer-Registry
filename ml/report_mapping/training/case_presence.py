"""Stage 1 gate: CasePresenceClassifier trainer.

Faithful port of ``ml/training/binary/{build_case_presence_dataset,
train_case_presence}.py``: any case with >= 1 confirmed cancer annotation is a
positive, every other case with an embedding is a non-cancer negative;
stratified train/val split, ``WeightedRandomSampler`` + ``BCEWithLogitsLoss``,
recall-weighted checkpoint selection (``recipe.GATE.recall_weight``). Legacy
already seeded torch + numpy here; ``recipe.seed_all`` adds python's
``random`` and CUDA for full determinism (``ml-rewrite-plan.md``, Findings).

``train_on_case_ids`` is the reusable core (labels already train-filtered, an
explicit case-id universe, and a pre-built embedding cache) that both
``train`` (the split-driven, CLI-facing entry point) and ``oof.py``
(k-fold, reusing the same cache across folds) call.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import config
from generations.splits import load_split
from report_mapping import sections
from report_mapping.inference.embedding_cache import EmbeddingCache
from report_mapping.model import backbone as backbone_mod
from report_mapping.model import generation as generation_mod
from report_mapping.model.heads import CasePresenceClassifier
from report_mapping.training import embeddings as embeddings_mod
from report_mapping.training import labels as labels_mod
from report_mapping.training import recipe
from taxonomy.taxonomy import load_labels_taxonomy


class _CaseDataset(Dataset):
    def __init__(self, embeddings: np.ndarray, targets: np.ndarray):
        self.embeddings = torch.from_numpy(embeddings)
        self.targets = torch.from_numpy(targets)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        return self.embeddings[idx], self.targets[idx]


def _evaluate(model: CasePresenceClassifier, loader: DataLoader, device: torch.device,
              threshold: float = 0.5) -> dict[str, float]:
    model.eval()
    preds: list[int] = []
    truths: list[int] = []
    with torch.no_grad():
        for emb, target in loader:
            probs = torch.sigmoid(model(emb.to(device))).cpu().numpy()
            preds.extend((probs >= threshold).astype(int).tolist())
            truths.extend(target.numpy().astype(int).tolist())
    p, r, f1, _ = precision_recall_fscore_support(truths, preds, average="binary", zero_division=0)
    return {"precision": float(p), "recall": float(r), "f1": float(f1)}


def default_backbone_dir() -> str:
    """The current generation's ``petbert/`` — the default backbone for
    heads-only training (ml-rewrite-plan.md WP5: "by default the current
    generation's, for the L3 'frozen old backbone' case")."""
    return str(generation_mod.generation_paths(config.REPORT_MAPPING_CURRENT_DIR).petbert_dir)


def train_on_case_ids(
    labels_train: pd.DataFrame,
    case_ids: list[str],
    cache: EmbeddingCache,
    seed: int,
    device: torch.device,
    checkpoint_path: str | Path,
) -> dict:
    """Train the gate on an explicit case-id universe against an already-built
    cache. The reusable core: ``train()`` resolves ``case_ids`` from a split;
    ``oof.py`` resolves them per fold, reusing the same ``cache``."""
    r = recipe.GATE
    recipe.seed_all(seed)

    cache_index = {cid: i for i, cid in enumerate(cache.case_ids)}
    idxs = [cache_index[cid] for cid in case_ids]
    embeddings = cache.col_embeddings[sections.CONCAT_3_KEY][idxs].astype(np.float32)
    targets = labels_mod.gate_targets(labels_train, case_ids)

    indices = np.arange(len(targets))
    train_idx, val_idx = train_test_split(
        indices, test_size=r.val_split, random_state=seed, stratify=targets.astype(int)
    )
    train_ds = _CaseDataset(embeddings[train_idx], targets[train_idx])
    val_ds = _CaseDataset(embeddings[val_idx], targets[val_idx])

    train_targets = targets[train_idx]
    n_train_pos = train_targets.sum()
    n_train_neg = len(train_targets) - n_train_pos
    sample_weights = np.where(train_targets == 1, n_train_neg / max(n_train_pos, 1), 1.0)
    sampler = WeightedRandomSampler(weights=sample_weights.tolist(), num_samples=len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=r.batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=r.batch_size, shuffle=False)

    model = CasePresenceClassifier(emb_dim=embeddings.shape[1], hidden_dim=r.hidden_dim, dropout=r.dropout).to(device)
    pw = torch.tensor([r.pos_weight], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = torch.optim.AdamW(model.parameters(), lr=r.lr, weight_decay=r.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=r.epochs)

    checkpoint_path = Path(checkpoint_path)
    model.eval()
    with torch.no_grad():
        initial_loss = criterion(model(train_ds.embeddings.to(device)), train_ds.targets.to(device)).item()

    best_score = -1.0
    best_epoch = 0
    last_loss = float("nan")
    for epoch in range(1, r.epochs + 1):
        model.train()
        total_loss = 0.0
        for emb, target in train_loader:
            emb, target = emb.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(emb), target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(target)
        scheduler.step()
        last_loss = total_loss / len(train_ds)
        m = _evaluate(model, val_loader, device)
        score = (1 - r.recall_weight) * m["precision"] + r.recall_weight * m["recall"]
        if score > best_score:
            best_score = score
            best_epoch = epoch
            model.save(checkpoint_path)

    model.eval()
    with torch.no_grad():
        final_loss = criterion(model(train_ds.embeddings.to(device)), train_ds.targets.to(device)).item()

    print(f"gate: best score {best_score:.3f} (epoch {best_epoch}/{r.epochs}), "
          f"train loss {initial_loss:.4f} -> {final_loss:.4f} (last epoch avg {last_loss:.4f})")
    return {
        "best_score": best_score, "best_epoch": best_epoch,
        "initial_loss": initial_loss, "final_loss": final_loss,
        "n_cases": len(case_ids), "n_cancer": int(targets.sum()),
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
    """Train the gate on ``labels`` filtered to ``split_id``'s train partition.

    Writes ``checkpoints/case_presence_classifier.pt`` under ``out_dir``
    (default: ``config.REPORT_MAPPING_CANDIDATE_DIR``).
    """
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

    checkpoint_path = generation_mod.generation_paths(out_dir).case_presence_pt
    return train_on_case_ids(labels_train, case_ids, cache, seed, dev, checkpoint_path)
