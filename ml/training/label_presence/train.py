"""Train a LabelPresenceClassifier for a single ICD group.

Reads the cache entry under "tfidf_selected" (alias). Under the canonical
concat-3 pipeline this is a 2304-dim per-row concat (3 sections × 768) and the
LP head is built with n_cols=3, col_pair_mode=True, col_combine="learned" so
each section scores the label independently through a shared MLP, then a
learned 3→1 weighted sum combines per-section logits. Under legacy single-col
mode (n_cols=1) the report embedding is 768-dim and col_pair_mode=False
collapses to the original [report | label] → MLP concat path.

Designed to be called once per ICD group by run_training.py --mode train-label-presence.
"""

import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from model.constants import DEFAULT_HIDDEN_DIM, PETBERT_EMB_DIM
from model.label_presence_classifier import LabelPresenceClassifier
from production.petbert_pipeline.embedding_cache import load_cache


class _PairDataset(Dataset):
    def __init__(self, report_embs: np.ndarray, label_embs: np.ndarray, targets: np.ndarray):
        self.r = torch.from_numpy(report_embs)
        self.l = torch.from_numpy(label_embs)
        self.t = torch.from_numpy(targets)

    def __len__(self) -> int:
        return len(self.t)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.r[idx], self.l[idx], self.t[idx]


def _evaluate(model: LabelPresenceClassifier, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for r, l, t in loader:
            logits = model(r.to(device), l.to(device))
            probs = torch.sigmoid(logits).cpu().numpy()
            preds.extend((probs >= 0.5).astype(int).tolist())
            trues.extend(t.numpy().astype(int).tolist())
    p, r, f1, _ = precision_recall_fscore_support(trues, preds, average="binary", zero_division=0)
    return {"precision": float(p), "recall": float(r), "f1": float(f1)}


def train_label_presence(
    *,
    pairs_csv: str,
    embedding_cache: str,
    out_path: str,
    epochs: int = 25,
    batch_size: int = 256,
    lr: float = 1e-3,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    dropout: float = 0.3,
    val_split: float = 0.15,
    pos_weight: float = 1.0,
    recall_weight: float = 0.5,
    weight_decay: float = 1e-4,
    patience: int = 0,
    device: str = "auto",
    seed: int = 42,
    model_name: str = "SAVSNET/PetBERT",
    labels_csv: str = "ml/ICD_labels/labels.csv",
    report_csv: str = "ml/data/report.csv",
    n_cols: int = 3,
    col_pair_mode: bool = True,
    col_combine: str = "learned",
) -> float:
    """Train LabelPresenceClassifier for one group. Returns best validation score."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    # --- Load pairs CSV -------------------------------------------------------
    with open(pairs_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"  Error: {pairs_csv} is empty")
        return 0.0

    targets = np.array([float(r["target"]) for r in rows], dtype=np.float32)
    label_strings = [f"{r['label_term']} {r['label_group']}" for r in rows]
    case_ids = [r["case_id"] for r in rows]

    # --- Load embeddings from cache -------------------------------------------
    cache = load_cache(
        embedding_cache,
        model_name=model_name,
        report_csv_path=report_csv,
        labels_csv_path=labels_csv,
    )
    if cache is None:
        print(f"  Error: embedding cache not found or stale: {embedding_cache}")
        return 0.0

    case_id_to_idx = {cid: i for i, cid in enumerate(cache["case_ids"])}
    label_text_to_idx = {t: i for i, t in enumerate(cache["label_texts"])}

    # Use TF-IDF selected embedding (768-dim, single column) as report embedding
    tfidf_col = "tfidf_selected"
    if tfidf_col not in cache["col_embeddings"]:
        print(f"  Error: 'tfidf_selected' column not found in cache. Run TF-IDF pipeline first.")
        return 0.0
    tfidf_embs = cache["col_embeddings"][tfidf_col]   # (N_cases, 768)
    label_embs_all = cache["label_embeddings"]          # (M_labels, 768)

    report_list, label_list, target_list, kept_case_ids = [], [], [], []
    skipped = 0
    for row, lstr, cid in zip(rows, label_strings, case_ids):
        cidx = case_id_to_idx.get(cid)
        lidx = label_text_to_idx.get(lstr)
        if cidx is None or lidx is None:
            skipped += 1
            continue
        report_list.append(tfidf_embs[cidx])
        label_list.append(label_embs_all[lidx])
        target_list.append(float(row["target"]))
        kept_case_ids.append(cid)

    if skipped:
        print(f"  Warning: {skipped}/{len(rows)} pairs skipped (case or label not in cache)")
    if len(target_list) < 10:
        print(f"  Error: too few pairs after cache lookup ({len(target_list)}), skipping")
        return 0.0

    report_embs = np.array(report_list, dtype=np.float32)
    label_embs  = np.array(label_list,  dtype=np.float32)
    targets     = np.array(target_list,  dtype=np.float32)
    case_id_arr = np.array(kept_case_ids)

    # --- Case-disjoint train/val split (QW2) ---------------------------------
    # GroupShuffleSplit ensures the same case never appears in both train and val,
    # preventing target-stratified leakage that previously inflated val scores.
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
    train_idx, val_idx = next(splitter.split(report_embs, targets, groups=case_id_arr))

    train_ds = _PairDataset(report_embs[train_idx], label_embs[train_idx], targets[train_idx])
    val_ds   = _PairDataset(report_embs[val_idx],   label_embs[val_idx],   targets[val_idx])

    train_targets = targets[train_idx]
    n_pos = train_targets.sum()
    n_neg = len(train_targets) - n_pos
    sample_weights = np.where(train_targets == 1, n_neg / max(n_pos, 1), 1.0)
    sampler = WeightedRandomSampler(sample_weights.tolist(), len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    print(f"  Split: {len(train_ds)} train / {len(val_ds)} val  (pos={int(n_pos)}, neg={int(n_neg)})")

    # --- Model ----------------------------------------------------------------
    # Default: n_cols=3, col_pair_mode=True, col_combine="learned"
    # — each section's 768-dim embedding scores the label independently via a
    # shared 1536→hidden→1 MLP; per-section logits are combined by a learned 3→1
    # head. Report embeddings expected shape: (B, n_cols * 768) = (B, 2304).
    model = LabelPresenceClassifier(
        emb_dim=PETBERT_EMB_DIM,
        hidden_dim=hidden_dim,
        dropout=dropout,
        n_cols=n_cols,
        col_pair_mode=col_pair_mode,
        col_combine=col_combine,
    ).to(dev)

    pw = torch.tensor([pos_weight], dtype=torch.float32, device=dev)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_score = -1.0
    epochs_since_best = 0
    best_epoch = 0
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"  {'Epoch':>5}  {'Loss':>8}  {'F1':>6}  {'P':>6}  {'R':>6}  {'Score':>7}")
    print("  " + "-" * 48)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for r, l, t in train_loader:
            r, l, t = r.to(dev), l.to(dev), t.to(dev)
            optimizer.zero_grad()
            loss = criterion(model(r, l), t)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(t)
        scheduler.step()

        avg_loss = total_loss / len(train_ds)
        m = _evaluate(model, val_loader, dev)
        score = (1 - recall_weight) * m["precision"] + recall_weight * m["recall"]
        improved = score > best_score
        marker = " *" if improved else ""
        print(
            f"  {epoch:>5}  {avg_loss:>8.4f}  {m['f1']:>6.3f}  "
            f"{m['precision']:>6.3f}  {m['recall']:>6.3f}  {score:>7.3f}{marker}"
        )
        if improved:
            best_score = score
            best_epoch = epoch
            epochs_since_best = 0
            model.save(out_path)
        else:
            epochs_since_best += 1
            if patience > 0 and epochs_since_best >= patience:
                print(f"  Early stop at epoch {epoch} (patience={patience}; best={best_epoch})")
                break

    print(f"  Best score: {best_score:.3f} (epoch {best_epoch}) -> {out_path}")
    return best_score
