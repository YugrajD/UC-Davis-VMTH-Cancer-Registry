"""Train CasePresenceClassifier on the case-level cancer/no-cancer dataset.

Steps:
  1. Load the NPZ dataset built by build_case_presence_dataset.py.
  2. Train/val split (stratified).
  3. WeightedRandomSampler + BCEWithLogitsLoss for class imbalance.
  4. Checkpoint by recall-weighted score — recall is prioritised because
     a false negative (missed cancer case) is worse than a false positive
     (a non-cancer case that reaches the GroupClassifier).
  5. Save best checkpoint to --out-dir/case_presence_classifier.pt.

Usage:
  python ml/training/binary/train_case_presence.py --device xpu
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import config
from model.case_presence_classifier import CasePresenceClassifier
from production.petbert_pipeline import device_from_arg


class _CaseDataset(Dataset):
    def __init__(self, embeddings: np.ndarray, targets: np.ndarray):
        self.embeddings = torch.from_numpy(embeddings)
        self.targets = torch.from_numpy(targets)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[idx], self.targets[idx]


def _evaluate(
    model: CasePresenceClassifier,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, float]:
    model.eval()
    preds: list[int] = []
    truths: list[int] = []
    with torch.no_grad():
        for emb, target in loader:
            probs = torch.sigmoid(model(emb.to(device))).cpu().numpy()
            preds.extend((probs >= threshold).astype(int).tolist())
            truths.extend(target.numpy().astype(int).tolist())
    p, r, f1, _ = precision_recall_fscore_support(
        truths, preds, average="binary", zero_division=0
    )
    return {"precision": float(p), "recall": float(r), "f1": float(f1)}


def train(
    *,
    dataset_npz: str = config.CASE_PRESENCE_DATASET_NPZ,
    out_dir: str = config.CHECKPOINT_CONTRASTIVE_DIR,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    hidden_dim: int = 512,
    dropout: float = 0.3,
    val_split: float = 0.15,
    device: str = "auto",
    seed: int = 42,
    pos_weight: float = 1.0,
    recall_weight: float = 0.7,
) -> int:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = device_from_arg(device)

    dataset_path = Path(dataset_npz)
    if not dataset_path.exists():
        print(
            f"Error: dataset not found at {dataset_npz}\n"
            "Run build_case_presence_dataset.py first."
        )
        return 1

    data = np.load(dataset_path, allow_pickle=True)
    embeddings = data["embeddings"].astype(np.float32)
    targets = data["targets"].astype(np.float32)

    n_pos = int(targets.sum())
    n_neg = len(targets) - n_pos
    print(f"Dataset: {len(targets)} cases  ({n_pos} cancer, {n_neg} non-cancer)")

    indices = np.arange(len(targets))
    train_idx, val_idx = train_test_split(
        indices, test_size=val_split, random_state=seed, stratify=targets.astype(int)
    )

    train_ds = _CaseDataset(embeddings[train_idx], targets[train_idx])
    val_ds = _CaseDataset(embeddings[val_idx], targets[val_idx])

    train_targets = targets[train_idx]
    n_train_pos = train_targets.sum()
    n_train_neg = len(train_targets) - n_train_pos
    sample_weights = np.where(
        train_targets == 1, n_train_neg / max(n_train_pos, 1), 1.0
    )
    sampler = WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=len(train_ds),
        replacement=True,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Split: {len(train_ds)} train / {len(val_ds)} val")

    model = CasePresenceClassifier(
        emb_dim=embeddings.shape[1],
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(dev)
    pw = torch.tensor([pos_weight], dtype=torch.float32, device=dev)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    checkpoint = out_path / "case_presence_classifier.pt"

    best_score = -1.0
    rw = recall_weight
    print(
        f"\n{'Epoch':>5}  {'Loss':>8}  {'F1':>6}  {'P':>6}  {'R':>6}  {'Score':>7}"
    )
    print("-" * 48)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for emb, target in train_loader:
            emb, target = emb.to(dev), target.to(dev)
            optimizer.zero_grad()
            loss = criterion(model(emb), target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(target)

        scheduler.step()
        avg_loss = total_loss / len(train_ds)
        m = _evaluate(model, val_loader, dev)
        score = (1 - rw) * m["precision"] + rw * m["recall"]
        marker = " *" if score > best_score else ""
        print(
            f"{epoch:>5}  {avg_loss:>8.4f}  {m['f1']:>6.3f}  "
            f"{m['precision']:>6.3f}  {m['recall']:>6.3f}  {score:>7.3f}{marker}"
        )
        if score > best_score:
            best_score = score
            model.save(checkpoint)

    print(f"\nBest checkpoint (recall_weight={rw}): {best_score:.3f}")
    print(f"Saved: {checkpoint}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train CasePresenceClassifier on case-level cancer/no-cancer data."
    )
    parser.add_argument("--dataset-npz", default=config.CASE_PRESENCE_DATASET_NPZ)
    parser.add_argument("--out-dir", default=config.CHECKPOINT_CONTRASTIVE_DIR)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps", "xpu"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pos-weight",
        type=float,
        default=1.0,
        help="BCEWithLogitsLoss pos_weight (>1 penalises false negatives more).",
    )
    parser.add_argument(
        "--recall-weight",
        type=float,
        default=0.7,
        help="Checkpoint selection weight for recall vs precision (default: 0.7). "
             "Higher = prefer fewer missed cancer cases over fewer false positives.",
    )
    args = parser.parse_args()
    return train(
        dataset_npz=args.dataset_npz,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        val_split=args.val_split,
        device=args.device,
        seed=args.seed,
        pos_weight=args.pos_weight,
        recall_weight=args.recall_weight,
    )


if __name__ == "__main__":
    raise SystemExit(main())
