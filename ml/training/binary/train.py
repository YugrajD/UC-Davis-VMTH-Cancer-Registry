"""Train the binary presence classifier on (case, label) pairs.

Steps:
  0. (auto) Build embedding cache via petbert_pipeline if missing.
  1. (auto) Build training_pairs.csv via build_training_pairs if missing.
  2. Load training_pairs.csv.
  3. Look up per-column mean embeddings from the embedding cache.
  4. Build a PyTorch Dataset from the cached embedding pairs.
  5. Train PresenceClassifier with BCEWithLogitsLoss, using a weighted sampler
     and pos_weight to handle class imbalance.
  6. Evaluate precision / recall / F1 on a held-out validation split after each epoch.
  7. Save the best checkpoint (by validation F1) to ml/output/checkpoints/.

Usage:
  python ml/training/binary/train.py --embedding-cache ml/output/training/embedding_cache.npz
  python ml/training/binary/train.py --embedding-cache ml/output/training/embedding_cache.npz --epochs 40
  python ml/training/binary/train.py --skip-prerequisites --embedding-cache ml/output/training/embedding_cache.npz
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import config
from model.constants import DEFAULT_HIDDEN_DIM, DEFAULT_TEXT_COLS, PETBERT_EMB_DIM
from model.presence_classifier import PresenceClassifier
from production.petbert_pipeline import run_scan, ScanConfig, device_from_arg
from training.binary.build_training_pairs import build_pairs as build_training_pairs


def _print_banner(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def _ensure_embedding_cache(
    cache_path: str,
    *,
    local_only: bool = False,
    device: str = "auto",
    embedding_min_sim: float = 0.5,
) -> None:
    """Build the embedding cache if it doesn't exist yet (i.e., run_cycle Step 0)."""
    if Path(cache_path).exists():
        return
    _print_banner("Prerequisite — Build embedding cache (first run only)")
    run_scan(ScanConfig(
        csv_path=config.REPORTS_CSV,
        id_col="case_id",
        text_cols=DEFAULT_TEXT_COLS,
        col_weights={},
        model_name="SAVSNET/PetBERT",
        local_only=local_only,
        out_dir=f"{config.OUTPUT_PRODUCTION_DIR}/binary",
        max_rows=None,
        batch_size=16,
        max_length=512,
        neighbors_k=3,
        task="categorize",
        embedding_min_sim=embedding_min_sim,
        device=device,
        labels_csv_path=config.LABELS_CSV,
        presence_classifier_path=None,
        embedding_cache_path=cache_path,
    ))


def _ensure_training_pairs(
    pairs_csv: str,
    *,
    co_neg_per_case: int = 3,
    fp_neg_per_case: int = 10,
    max_pos_per_group: int = 0,
) -> None:
    """Build training_pairs.csv if it doesn't exist yet (run_cycle Step 1)."""
    if Path(pairs_csv).exists():
        return
    _print_banner("Prerequisite — Build training pairs")
    build_training_pairs(
        co_neg_per_case=co_neg_per_case,
        fp_neg_per_case=fp_neg_per_case,
        max_pos_per_group=max_pos_per_group,
    )


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


def train(
    *,
    pairs_csv: str = config.TRAINING_PAIRS_CSV,
    embedding_cache: str | None = None,
    report_csv: str = config.REPORTS_CSV,
    labels_csv: str = config.LABELS_CSV,
    out_dir: str = config.CHECKPOINT_BINARY_DIR,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    dropout: float = 0.3,
    val_split: float = 0.15,
    device: str = "auto",
    seed: int = 42,
    pos_weight: float = 1.0,
    recall_weight: float = 0.5,
    skip_prerequisites: bool = False,
    local_only: bool = False,
    model_name: str = "SAVSNET/PetBERT",
) -> int:
    # Auto-build prerequisites if missing (i.e., run_cycle Steps 0–1)
    if not skip_prerequisites:
        cache_path = embedding_cache or config.EMBEDDING_CACHE_NPZ
        _ensure_embedding_cache(
            cache_path,
            local_only=local_only,
            device=device,
        )
        _ensure_training_pairs(pairs_csv)

    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = device_from_arg(device)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # --- Load training pairs ---------------------------------------------
    print("Loading training pairs...")
    with open(pairs_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print(f"Error: {pairs_csv} is empty. Run build_training_pairs.py first.")
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
    if embedding_cache:
        from production.petbert_pipeline import load_cache
        cache = load_cache(
            embedding_cache,
            model_name=model_name,
            report_csv_path=report_csv,
            labels_csv_path=labels_csv,
        )

    if cache is not None:
        # Use per-column embeddings from cache — each column is fed to the classifier
        # independently so it can learn column-specific signal instead of a diluted mean.
        print(f"Using cached embeddings from {embedding_cache}")
        case_id_to_idx   = {cid: i for i, cid in enumerate(cache["case_ids"])}
        label_to_idx     = {t: i   for i, t   in enumerate(cache["label_texts"])}

        # Use enriched label embeddings when available so the classifier trains on
        # the same representations used during inference.
        cached_label_embs = (
            cache["enriched_label_embeddings"]
            if cache.get("enriched_label_embeddings") is not None
            else cache["label_embeddings"]
        )
        if cache.get("enriched_label_embeddings") is not None:
            print("  Using enriched label embeddings from cache.")

        # Precompute (N_cases, n_cols * PETBERT_EMB_DIM) with empty columns zeroed.
        col_names = cache["col_names"]
        n_cols = len(col_names)
        n_cases = len(cache["case_ids"])
        col_emb_matrix = np.zeros((n_cases, n_cols * PETBERT_EMB_DIM), dtype=np.float32)
        for ci, col in enumerate(col_names):
            has  = cache["col_has_content"][col]   # (N,) bool
            embs = cache["col_embeddings"][col]    # (N, 768)
            col_emb_matrix[:, ci * PETBERT_EMB_DIM:(ci + 1) * PETBERT_EMB_DIM] = (
                np.where(has[:, None], embs, 0.0)
            )
        print(f"  Using {n_cols} report columns: {col_names}")

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
            report_embs_list.append(col_emb_matrix[ridx])
            label_embs_list.append(cached_label_embs[lidx])
            targets_list.append(float(row["target"]))
        if skipped:
            print(f"  Warning: {skipped} rows skipped (case or label not in cache)")
        report_embs = np.array(report_embs_list, dtype=np.float32)
        label_embs  = np.array(label_embs_list,  dtype=np.float32)
        targets     = np.array(targets_list,      dtype=np.float32)
        print(f"  Using {len(targets)} pairs after cache lookup")
    else:
        cache_path = embedding_cache or config.EMBEDDING_CACHE_NPZ
        print(
            f"\nError: embedding cache not found or stale.\n"
            f"Build it first by running petbert_pipeline once:\n\n"
            f"  python -m petbert_pipeline --embedding-cache {cache_path}\n\n"
            f"Or use training/binary/run_cycle.py, which builds the cache automatically on first run."
        )
        return 1

    # --- Train / val split -----------------------------------------------
    indices = np.arange(len(targets))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_split,
        random_state=seed,
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

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    print(f"\nSplit: {len(train_ds)} train / {len(val_ds)} val")

    # --- Model, loss, optimiser ------------------------------------------
    classifier = PresenceClassifier(
        emb_dim=PETBERT_EMB_DIM, hidden_dim=hidden_dim, dropout=dropout,
        n_cols=n_cols, col_pair_mode=False,
    ).to(dev)

    pw = torch.tensor([pos_weight], dtype=torch.float32, device=dev)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # --- Training loop ---------------------------------------------------
    best_score = -1.0
    best_checkpoint = out_path / "presence_classifier_current.pt"
    rw = recall_weight

    print(f"\n{'Epoch':>5}  {'Loss':>8}  {'F1':>6}  {'P':>6}  {'R':>6}  {'Acc':>6}  {'Score':>7}")
    print("-" * 54)

    for epoch in range(1, epochs + 1):
        classifier.train()
        total_loss = 0.0
        for report_emb, label_emb, target in train_loader:
            report_emb = report_emb.to(dev)
            label_emb  = label_emb.to(dev)
            target     = target.to(dev)

            optimizer.zero_grad()
            logits = classifier(report_emb, label_emb)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(target)

        scheduler.step()
        avg_loss = total_loss / len(train_ds)
        m = evaluate(classifier, val_loader, dev)

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


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the binary presence classifier.")
    parser.add_argument("--pairs-csv", default=config.TRAINING_PAIRS_CSV,
                        help="Output of build_training_pairs.py")
    parser.add_argument("--out-dir", default=config.CHECKPOINT_BINARY_DIR,
                        help="Directory to save the best checkpoint")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for classifier training")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--val-split", type=float, default=0.15,
                        help="Fraction of data to hold out for validation")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda", "mps", "xpu"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pos-weight", type=float, default=1.0,
                        help="BCEWithLogitsLoss pos_weight: >1 penalises false negatives more "
                             "(e.g. 2.0 = FN twice as costly as FP). Default 1.0 (balanced).")
    parser.add_argument("--recall-weight", type=float, default=0.5,
                        help="Checkpoint selection weight given to recall vs precision. "
                             "0.5 = standard F1, 1.0 = pure recall. Default 0.5.")
    parser.add_argument("--embedding-cache", default=None,
                        help="Path to embedding cache npz (from petbert_pipeline --embedding-cache). "
                             "When provided, PetBERT is not loaded — embeddings are read from "
                             "the cache using case_id, fixing the train/inference mismatch.")
    parser.add_argument("--report-csv", default=config.REPORTS_CSV,
                        help="Path to report CSV (used only for cache validation).")
    parser.add_argument("--labels-csv", default=config.LABELS_CSV,
                        help="Path to labels CSV (used only for cache validation).")
    parser.add_argument("--skip-prerequisites", action="store_true",
                        help="Skip auto-building embedding cache and training pairs. "
                             "Use when these files already exist and you want to skip the checks.")
    parser.add_argument("--local-only", action="store_true",
                        help="Use only locally cached PetBERT model files (no network calls). "
                             "Only relevant when the embedding cache needs to be built.")
    args = parser.parse_args()
    return train(
        pairs_csv=args.pairs_csv,
        embedding_cache=args.embedding_cache,
        report_csv=args.report_csv,
        labels_csv=args.labels_csv,
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
        skip_prerequisites=args.skip_prerequisites,
        local_only=args.local_only,
    )


if __name__ == "__main__":
    raise SystemExit(main())
