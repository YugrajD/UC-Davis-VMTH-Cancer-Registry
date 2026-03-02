"""Train the binary presence classifier on (case, label) pairs.

Steps:
  1. Load training_pairs.csv.
  2. Embed all unique report texts with PetBERT (frozen — no gradients through it).
  3. Embed all unique taxonomy label strings with PetBERT (frozen).
  4. Build a PyTorch Dataset from the cached embedding pairs.
  5. Train PresenceClassifier with BCEWithLogitsLoss, using a weighted sampler
     and pos_weight to handle class imbalance.
  6. Evaluate precision / recall / F1 on a held-out validation split after each epoch.
  7. Save the best checkpoint (by validation F1) to ml/model/checkpoints/.

Usage:
  python ml/model/train_classifier.py
  python ml/model/train_classifier.py --epochs 40 --device cuda
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.presence_classifier import PresenceClassifier
from petbert_scan.embedding import embed_texts, load_tokenizer_and_model
from petbert_scan.utils import device_from_arg


class PairDataset(Dataset):
    def __init__(
        self,
        report_embs: np.ndarray,  # (N, 768)
        label_embs: np.ndarray,   # (N, 768)
        targets: np.ndarray,      # (N,) float32
    ):
        self.report_embs = torch.from_numpy(report_embs)
        self.label_embs = torch.from_numpy(label_embs)
        self.targets = torch.from_numpy(targets)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.report_embs[idx], self.label_embs[idx], self.targets[idx]


def evaluate(
    model: PresenceClassifier,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, float]:
    model.eval()
    all_preds: list[int] = []
    all_targets: list[int] = []
    with torch.no_grad():
        for report_emb, label_emb, target in loader:
            logits = model(report_emb.to(device), label_emb.to(device))
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend((probs >= threshold).astype(int).tolist())
            all_targets.extend(target.numpy().astype(int).tolist())
    p, r, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="binary", zero_division=0
    )
    acc = sum(int(p == t) for p, t in zip(all_preds, all_targets)) / max(len(all_targets), 1)
    return {"precision": float(p), "recall": float(r), "f1": float(f1), "accuracy": float(acc)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the binary presence classifier.")
    parser.add_argument("--pairs-csv", default="ml/data/training_pairs.csv",
                        help="Output of build_training_pairs.py")
    parser.add_argument("--model", default="SAVSNET/PetBERT",
                        help="HuggingFace model name or local path for embedding")
    parser.add_argument("--local-only", action="store_true",
                        help="Use only locally cached model files")
    parser.add_argument("--out-dir", default="ml/model/checkpoints",
                        help="Directory to save the best checkpoint")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for classifier training")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--val-split", type=float, default=0.15,
                        help="Fraction of data to hold out for validation")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda", "mps", "xpu"])
    parser.add_argument("--max-length", type=int, default=512,
                        help="Tokenizer max_length for PetBERT embedding")
    parser.add_argument("--embed-batch-size", type=int, default=16,
                        help="Batch size for PetBERT embedding pass")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = device_from_arg(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load training pairs ---------------------------------------------
    print("Loading training pairs...")
    with open(args.pairs_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print(f"Error: {args.pairs_csv} is empty. Run build_training_pairs.py first.")
        return 1

    report_texts = [r["merged_text"] for r in rows]
    label_terms  = [r["label_term"]  for r in rows]
    label_groups = [r["label_group"] for r in rows]
    targets = np.array([float(r["target"]) for r in rows], dtype=np.float32)

    n_pos = int(targets.sum())
    n_neg = len(targets) - n_pos
    print(f"  Total pairs: {len(rows)}  (positives={n_pos}, negatives={n_neg})")

    # --- Embed with PetBERT (frozen) ------------------------------------
    print(f"\nLoading PetBERT ({args.model})...")
    tokenizer, petbert = load_tokenizer_and_model(args.model, local_only=args.local_only)

    print("Embedding unique report texts...")
    unique_texts = list(dict.fromkeys(report_texts))
    text_to_idx = {t: i for i, t in enumerate(unique_texts)}
    unique_report_embs, _ = embed_texts(
        tokenizer, petbert, unique_texts,
        device=device, batch_size=args.embed_batch_size, max_length=args.max_length,
        desc="Reports",
    )
    report_embs = unique_report_embs[[text_to_idx[t] for t in report_texts]]

    print("Embedding taxonomy label strings...")
    label_strings = [f"{term} {group}" for term, group in zip(label_terms, label_groups)]
    unique_label_strings = list(dict.fromkeys(label_strings))
    label_str_to_idx = {s: i for i, s in enumerate(unique_label_strings)}
    unique_label_embs, _ = embed_texts(
        tokenizer, petbert, unique_label_strings,
        device=device, batch_size=args.embed_batch_size, max_length=args.max_length,
        desc="Labels",
    )
    label_embs = unique_label_embs[[label_str_to_idx[s] for s in label_strings]]

    # Free PetBERT memory — embeddings are now in numpy arrays on CPU
    petbert.cpu()
    del petbert, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Train / val split -----------------------------------------------
    indices = np.arange(len(rows))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=args.val_split,
        random_state=args.seed,
        stratify=targets.astype(int),
    )

    train_ds = PairDataset(report_embs[train_idx], label_embs[train_idx], targets[train_idx])
    val_ds   = PairDataset(report_embs[val_idx],   label_embs[val_idx],   targets[val_idx])

    # Weighted sampler so each epoch sees a balanced mix of pos/neg examples
    train_targets = targets[train_idx]
    n_train_pos = train_targets.sum()
    n_train_neg = len(train_targets) - n_train_pos
    sample_weights = np.where(train_targets == 1, n_train_neg / max(n_train_pos, 1), 1.0)
    sampler = WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=len(train_ds),
        replacement=True,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    print(f"\nSplit: {len(train_ds)} train / {len(val_ds)} val")

    # --- Model, loss, optimiser ------------------------------------------
    classifier = PresenceClassifier(
        emb_dim=768, hidden_dim=args.hidden_dim, dropout=args.dropout,
    ).to(device)

    pos_weight = torch.tensor([n_train_neg / max(n_train_pos, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --- Training loop ---------------------------------------------------
    best_f1 = -1.0
    best_checkpoint = out_dir / "presence_classifier_best.pt"

    print(f"\n{'Epoch':>5}  {'Loss':>8}  {'F1':>6}  {'P':>6}  {'R':>6}  {'Acc':>6}")
    print("-" * 45)

    for epoch in range(1, args.epochs + 1):
        classifier.train()
        total_loss = 0.0
        for report_emb, label_emb, target in train_loader:
            report_emb = report_emb.to(device)
            label_emb  = label_emb.to(device)
            target     = target.to(device)

            optimizer.zero_grad()
            logits = classifier(report_emb, label_emb)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(target)

        scheduler.step()
        avg_loss = total_loss / len(train_ds)
        m = evaluate(classifier, val_loader, device)

        marker = " *" if m["f1"] > best_f1 else ""
        print(
            f"{epoch:>5}  {avg_loss:>8.4f}  {m['f1']:>6.3f}  "
            f"{m['precision']:>6.3f}  {m['recall']:>6.3f}  {m['accuracy']:>6.3f}{marker}"
        )

        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            classifier.save(best_checkpoint)

    print(f"\nBest validation F1: {best_f1:.3f}")
    print(f"Checkpoint saved to: {best_checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
