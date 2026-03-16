"""Train the binary presence classifier on (case, label) pairs.

Steps:
  1. Load training_pairs.csv.
  2. Look up per-column mean embeddings from the embedding cache (built by petbert_scan).
     Using cached embeddings ensures train/inference consistency: both use the same
     mean-of-columns representation, not a merged-text approximation.
  3. Build a PyTorch Dataset from the cached embedding pairs.
  4. Train PresenceClassifier with BCEWithLogitsLoss, using a weighted sampler
     and pos_weight to handle class imbalance.
  5. Evaluate precision / recall / F1 on a held-out validation split after each epoch.
  6. Save the best checkpoint (by validation F1) to ml/model/checkpoints/.

Requires an embedding cache built by petbert_scan --embedding-cache <path>.
Use run_training_cycle.py to orchestrate the full cycle; it builds the cache automatically.

Usage:
  python ml/scripts/utils/train_classifier.py --embedding-cache ml/data/embedding_cache.npz
  python ml/scripts/utils/train_classifier.py --embedding-cache ml/data/embedding_cache.npz --epochs 40
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

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from model.presence_classifier import PresenceClassifier
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
    parser.add_argument("--pos-weight", type=float, default=1.0,
                        help="BCEWithLogitsLoss pos_weight: >1 penalises false negatives more "
                             "(e.g. 2.0 = FN twice as costly as FP). Default 1.0 (balanced).")
    parser.add_argument("--recall-weight", type=float, default=0.5,
                        help="Checkpoint selection weight given to recall vs precision. "
                             "0.5 = standard F1, 1.0 = pure recall. Default 0.5.")
    parser.add_argument("--embedding-cache", default=None,
                        help="Path to embedding cache npz (from petbert_scan --embedding-cache). "
                             "When provided, PetBERT is not loaded — embeddings are read from "
                             "the cache using case_id, fixing the train/inference mismatch. "
                             "Falls back to embedding merged_text if the cache is missing.")
    parser.add_argument("--report-csv", default="ml/data/report.csv",
                        help="Path to report CSV (used only for cache validation).")
    parser.add_argument("--labels-csv", default="ml/labels/labels.csv",
                        help="Path to labels CSV (used only for cache validation).")
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

    label_terms  = [r["label_term"]  for r in rows]
    label_groups = [r["label_group"] for r in rows]
    label_strings = [f"{term} {group}" for term, group in zip(label_terms, label_groups)]
    targets = np.array([float(r["target"]) for r in rows], dtype=np.float32)

    n_pos = int(targets.sum())
    n_neg = len(targets) - n_pos
    print(f"  Total pairs: {len(rows)}  (positives={n_pos}, negatives={n_neg})")

    # --- Load or compute embeddings --------------------------------------
    cache = None
    if args.embedding_cache:
        from petbert_scan.embedding_cache import load_cache
        cache = load_cache(
            args.embedding_cache,
            model_name=args.model,
            report_csv_path=args.report_csv,
            labels_csv_path=args.labels_csv,
        )

    if cache is not None:
        # Use per-column mean embeddings from cache — matches petbert_scan inference exactly.
        print(f"Using cached embeddings from {args.embedding_cache}")
        case_id_to_idx   = {cid: i for i, cid in enumerate(cache["case_ids"])}
        label_to_idx     = {t: i   for i, t   in enumerate(cache["label_texts"])}

        report_embs_list: list[np.ndarray] = []
        label_embs_list:  list[np.ndarray] = []
        targets_list:     list[float]      = []
        skipped = 0
        for row, lstr in zip(rows, label_strings):
            ridx = case_id_to_idx.get(row["case_id"])
            lidx = label_to_idx.get(lstr)
            if ridx is None or lidx is None:
                skipped += 1
                continue
            report_embs_list.append(cache["mean_embeddings"][ridx])
            label_embs_list.append(cache["label_embeddings"][lidx])
            targets_list.append(float(row["target"]))
        if skipped:
            print(f"  Warning: {skipped} rows skipped (case or label not in cache)")
        report_embs = np.array(report_embs_list, dtype=np.float32)
        label_embs  = np.array(label_embs_list,  dtype=np.float32)
        targets     = np.array(targets_list,      dtype=np.float32)
        print(f"  Using {len(targets)} pairs after cache lookup")
    else:
        # No cache available — abort with a clear message rather than silently using
        # merged-text embeddings (which would reintroduce the train/inference mismatch).
        cache_path = args.embedding_cache or "ml/data/embedding_cache.npz"
        print(
            f"\nError: embedding cache not found or stale.\n"
            f"Build it first by running petbert_scan once:\n\n"
            f"  python -m petbert_scan --embedding-cache {cache_path}\n\n"
            f"Or use run_training_cycle.py, which builds the cache automatically on first run."
        )
        return 1

    # --- Train / val split -----------------------------------------------
    indices = np.arange(len(targets))
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

    # The WeightedRandomSampler already produces balanced 50/50 batches.
    # pos_weight > 1.0 additionally penalises false negatives (missed cancers)
    # more than false positives during training.
    pos_weight = torch.tensor([args.pos_weight], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --- Training loop ---------------------------------------------------
    best_score = -1.0
    best_checkpoint = out_dir / "presence_classifier_best.pt"
    rw = args.recall_weight  # shorthand

    print(f"\n{'Epoch':>5}  {'Loss':>8}  {'F1':>6}  {'P':>6}  {'R':>6}  {'Acc':>6}  {'Score':>7}")
    print("-" * 54)

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

        # Weighted score: rw=0.5 → standard F1, rw=1.0 → pure recall
        score = (1 - rw) * m["precision"] + rw * m["recall"]
        marker = " *" if score > best_score else ""
        print(
            f"{epoch:>5}  {avg_loss:>8.4f}  {m['f1']:>6.3f}  "
            f"{m['precision']:>6.3f}  {m['recall']:>6.3f}  {m['accuracy']:>6.3f}  {score:>7.3f}{marker}"
        )

        if score > best_score:
            best_score = score
            classifier.save(best_checkpoint)

    print(f"\nBest checkpoint score (recall_weight={rw}): {best_score:.3f}")
    print(f"Checkpoint saved to: {best_checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
