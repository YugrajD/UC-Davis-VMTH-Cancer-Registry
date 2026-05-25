"""Train the GroupClassifier on cached report embeddings.

Reads pre-built training data from build_group_training_data.py and trains
a multi-label MLP that maps report embeddings → cancer group probabilities.

Unlike the binary PresenceClassifier training cycle, this script does not need
to iterate — it trains once on the cached embeddings and keyword ground truth.
Re-train whenever the keyword pipeline improves (more coverage = better labels).

Usage:
  python ml/training/group/train.py
  python ml/training/group/train.py --epochs 100 --device mps

After training, run the PetBERT pipeline with:
  ml/.venv/bin/python3 -m petbert_pipeline \\
      --group-classifier ml/output/checkpoints/group/group_classifier_best.pt \\
      --embedding-cache ml/output/training/embedding_cache.npz \\
      --local-only

Note: group_classifier_current.pt is overwritten each run (best epoch within the run).
      group_classifier_best.pt is only updated when val macro F1 beats all prior runs.

Improvement flags (added Phase 27):
  --dropout           Try 0.05 or 0.1 -- Phase 26 train/val gap is tight (0.247/0.258),
                      suggesting 0.3 over-regularises. Experiment 3 found 0.3→0.05
                      gave +0.18 macro F1 at lower data volumes.
  --lr-schedule cosine  CosineAnnealingWarmRestarts(T_0=100) -- addresses the fixed-LR
                      plateau seen at epoch 219 in Phase 26.
  --focal-loss        Replace BCEWithLogitsLoss with Focal Loss (gamma=2 default).
                      Down-weights easy non-cancer examples; focuses gradient on hard
                      group assignments. Better for imbalanced multi-label.
  --focal-gamma       Focal loss focusing parameter (default: 2.0). Higher = more focus
                      on hard examples. Typical range: 1.0 -- 5.0.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from model.constants import DEFAULT_HIDDEN_DIM
from model.group_classifier import GroupClassifier


def _resolve_device(arg: str) -> torch.device:
    if arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        try:
            import intel_extension_for_pytorch as ipex  # noqa: F401
            if ipex.xpu.is_available():
                return torch.device("xpu")
        except (ImportError, AttributeError):
            pass
        return torch.device("cpu")
    return torch.device(arg)


def _macro_f1(probs: torch.Tensor, targets: torch.Tensor, threshold: float) -> float:
    """Macro-averaged F1 across all groups (groups with no positives in val are excluded)."""
    preds = (probs >= threshold).float()
    f1s = []
    for g in range(probs.shape[1]):
        has_positives = targets[:, g].sum().item() > 0
        if not has_positives:
            continue  # Skip groups absent from validation set
        tp = (preds[:, g] * targets[:, g]).sum().item()
        fp = (preds[:, g] * (1 - targets[:, g])).sum().item()
        fn = ((1 - preds[:, g]) * targets[:, g]).sum().item()
        denom_p = tp + fp
        denom_r = tp + fn
        p = tp / denom_p if denom_p > 0 else 0.0
        r = tp / denom_r if denom_r > 0 else 0.0
        f1s.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def _per_group_stats(
    probs: torch.Tensor,
    targets: torch.Tensor,
    group_names: list[str],
    threshold: float,
) -> None:
    """Print per-group precision, recall, F1 on validation set."""
    preds = (probs >= threshold).float()
    print(f"\n{'Group':<50} {'P':>6} {'R':>6} {'F1':>6} {'Support':>8}")
    print("-" * 78)
    for g, name in enumerate(group_names):
        tp = (preds[:, g] * targets[:, g]).sum().item()
        fp = (preds[:, g] * (1 - targets[:, g])).sum().item()
        fn = ((1 - preds[:, g]) * targets[:, g]).sum().item()
        support = int(targets[:, g].sum().item())
        denom_p = tp + fp
        denom_r = tp + fn
        p = tp / denom_p if denom_p > 0 else 0.0
        r = tp / denom_r if denom_r > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        print(f"{name:<50} {p:>6.3f} {r:>6.3f} {f1:>6.3f} {support:>8}")


def _stratified_split(
    targets: np.ndarray,
    val_frac: float,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Train/val split that preserves per-group case distribution.

    Each case is assigned to the stratum of its first positive group
    (or -1 for non-cancer). Indices within each stratum are shuffled,
    then the first val_frac are held out for validation.

    Returns (train_indices, val_indices).
    """
    rng = np.random.default_rng(seed)
    strata = np.array([
        int(np.where(row > 0)[0][0]) if row.sum() > 0 else -1
        for row in targets
    ])
    train_indices: list[int] = []
    val_indices: list[int] = []
    for stratum in np.unique(strata):
        idx = np.where(strata == stratum)[0]
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_frac))
        val_indices.extend(idx[:n_val].tolist())
        train_indices.extend(idx[n_val:].tolist())
    return train_indices, val_indices


class _FocalBCEWithLogitsLoss(nn.Module):
    """Focal loss for multi-label binary classification.

    FL(p_t) = -(1 - p_t)^gamma * log(p_t)

    Down-weights easy examples (high p_t) so training focuses on the
    hard group assignments that the model currently gets wrong.
    pos_weight mirrors BCEWithLogitsLoss semantics for class imbalance.
    """

    def __init__(self, pos_weight: torch.Tensor, gamma: float = 2.0) -> None:
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        p_t = torch.exp(-bce)
        return ((1 - p_t) ** self.gamma * bce).mean()


class _ASLLoss(nn.Module):
    """Asymmetric Loss for multi-label classification (Ridnik et al., 2021).

    Applies separate focusing for positives (gamma_pos) and negatives (gamma_neg),
    with a probability floor `margin` that shifts easy negatives down before the log
    so they contribute near-zero gradient. This avoids the collapse seen with focal
    loss (gamma=2) while still suppressing precision-killing easy-negative gradients.

    Recommended starting point: gamma_pos=1.0, gamma_neg=4.0, margin=0.05.
    pos_weight mirrors BCEWithLogitsLoss semantics for class imbalance.
    """

    def __init__(
        self,
        pos_weight: torch.Tensor,
        gamma_pos: float = 1.0,
        gamma_neg: float = 4.0,
        margin: float = 0.05,
    ) -> None:
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.margin = margin

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)

        # Shift negatives down by margin and clamp to [0, 1] before log
        p_neg = (p - self.margin).clamp(min=0.0)

        # Per-element log-probabilities for positives and shifted negatives
        xs_pos = torch.log(p.clamp(min=1e-8))
        xs_neg = torch.log((1.0 - p_neg).clamp(min=1e-8))

        # Weighted BCE using pos_weight for the positive term
        pw = self.pos_weight.unsqueeze(0)  # (1, G) broadcast over batch
        loss = -(pw * targets * xs_pos + (1.0 - targets) * xs_neg)

        # Asymmetric focusing
        p_t = targets * p + (1.0 - targets) * (1.0 - p_neg)
        gamma = targets * self.gamma_pos + (1.0 - targets) * self.gamma_neg
        loss = loss * (1.0 - p_t) ** gamma

        return loss.mean()


def train(
    *,
    training_data_path: str,
    out_path: str,
    epochs: int,
    lr: float,
    hidden_dim: int,
    val_frac: float,
    threshold: float,
    device_arg: str,
    weight_decay: float,
    max_class_weight: float,
    min_group_cases: int,
    max_group_cases: int,
    dropout: float,
    lr_schedule: str,
    focal_loss: bool,
    focal_gamma: float,
    asl: bool = False,
    asl_gamma_pos: float = 1.0,
    asl_gamma_neg: float = 4.0,
    asl_margin: float = 0.05,
) -> None:
    # --- Load training data --------------------------------------------------
    print(f"Loading training data: {training_data_path}")
    if not Path(training_data_path).exists():
        print(f"ERROR: training data not found at {training_data_path}")
        print("Run build_training_data.py first:")
        print("  ml/.venv/Scripts/python.exe ml/training/group/build_training_data.py")
        sys.exit(1)

    data = np.load(training_data_path, allow_pickle=True)
    embeddings = torch.from_numpy(data["embeddings"].astype(np.float32))  # (N, D)
    targets = torch.from_numpy(data["targets"].astype(np.float32))        # (N, G)
    group_names: list[str] = list(data["group_names"])
    class_weights = torch.from_numpy(data["class_weights"].astype(np.float32))  # (G,)

    # --- Drop sparse groups --------------------------------------------------
    if min_group_cases > 0:
        positive_counts = targets.sum(dim=0)  # (G,)
        keep_mask = positive_counts >= min_group_cases
        n_dropped = int((~keep_mask).sum().item())
        if n_dropped > 0:
            dropped = [group_names[i] for i in range(len(group_names)) if not keep_mask[i]]
            print(f"Dropping {n_dropped} group(s) with < {min_group_cases} cases: {dropped}")
            targets = targets[:, keep_mask]
            class_weights = class_weights[keep_mask]
            group_names = [g for g, k in zip(group_names, keep_mask.tolist()) if k]

    if max_class_weight > 0:
        class_weights = class_weights.clamp(max=max_class_weight)

    # --- Cap positive samples per group --------------------------------------
    if max_group_cases > 0:
        rng = np.random.default_rng(42)
        selected: set[int] = set()
        for g in range(len(group_names)):
            pos_idx = torch.where(targets[:, g] > 0)[0].numpy()
            if len(pos_idx) > max_group_cases:
                pos_idx = rng.choice(pos_idx, max_group_cases, replace=False)
            selected.update(pos_idx.tolist())
        non_cancer_idx = torch.where(targets.sum(dim=1) == 0)[0].numpy().tolist()
        keep = torch.tensor(sorted(selected | set(non_cancer_idx)))
        old_N = embeddings.shape[0]
        embeddings = embeddings[keep]
        targets = targets[keep]
        n_cancer = int((targets.sum(dim=1) > 0).sum())
        print(
            f"Per-group cap {max_group_cases}: {embeddings.shape[0]}/{old_N} cases kept "
            f"({n_cancer} cancer, {embeddings.shape[0] - n_cancer} non-cancer)"
        )
        # Recalculate class weights from the capped dataset — stored weights are
        # computed from the original full dataset and are stale after row removal.
        pos_counts = targets.sum(dim=0).clamp(min=1)
        neg_counts = targets.shape[0] - pos_counts
        class_weights = (neg_counts / pos_counts).float()
        if max_class_weight > 0:
            class_weights = class_weights.clamp(max=max_class_weight)

    G = len(group_names)
    N = embeddings.shape[0]
    emb_dim = embeddings.shape[1]
    device = _resolve_device(device_arg)

    # sizes resolved after stratified split below
    print(f"Device: {device} | Cases: {N} | Groups: {G} | Emb dim: {emb_dim}")
    print(f"Val fraction: {val_frac} | Threshold: {threshold}")
    print(f"Cancer cases: {int((targets.sum(dim=1) > 0).sum())}")
    print()

    # --- Data loaders --------------------------------------------------------
    train_idx, val_idx = _stratified_split(targets.numpy(), val_frac)
    train_size = len(train_idx)
    val_size   = len(val_idx)
    train_ds = torch.utils.data.Subset(TensorDataset(embeddings, targets), train_idx)
    val_ds   = torch.utils.data.Subset(TensorDataset(embeddings, targets), val_idx)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False)
    print(f"Train: {train_size} | Val: {val_size} (stratified)")

    # --- Model, optimizer, loss ----------------------------------------------
    model = GroupClassifier(num_groups=G, emb_dim=emb_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Criterion: focal loss focuses gradient on hard examples; BCE is the standard baseline.
    # Both use raw logits — sigmoid is applied internally.
    pw = class_weights.to(device)
    if asl:
        # ASL handles class imbalance via gamma_neg — do not also apply pos_weight or the
        # two corrections compound (pos_weight up to 50x * easy-negative suppression →
        # recall=1 / precision≈0 collapse). Pass ones so ASL's focusing is the sole lever.
        criterion: nn.Module = _ASLLoss(
            pos_weight=torch.ones_like(pw),
            gamma_pos=asl_gamma_pos,
            gamma_neg=asl_gamma_neg,
            margin=asl_margin,
        )
        print(f"Loss: ASL (gamma_pos={asl_gamma_pos}, gamma_neg={asl_gamma_neg}, margin={asl_margin}, pos_weight=1)")
    elif focal_loss:
        criterion = _FocalBCEWithLogitsLoss(pos_weight=pw, gamma=focal_gamma)
        print(f"Loss: Focal BCE (gamma={focal_gamma})")
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
        print("Loss: BCE with pos_weight")

    # LR scheduler: cosine warm restarts help escape fixed-LR plateaus.
    if lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=100, T_mult=1
        )
        print("LR schedule: CosineAnnealingWarmRestarts(T_0=100)")
    else:
        scheduler = None
        print("LR schedule: none (fixed)")

    best_f1 = -1.0
    best_epoch = 0

    print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>9}  {'Macro F1':>9}  {'LR':>9}  {'Best':>5}")
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        # --- Train -----------------------------------------------------------
        model.train()
        train_loss = 0.0
        for emb_batch, tgt_batch in train_loader:
            emb_batch = emb_batch.to(device)
            tgt_batch = tgt_batch.to(device)
            optimizer.zero_grad()
            logits = model.net(emb_batch)  # raw logits for BCEWithLogitsLoss
            loss = criterion(logits, tgt_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(emb_batch)
        train_loss /= train_size

        # --- Validate --------------------------------------------------------
        model.eval()
        val_loss = 0.0
        all_probs, all_targets = [], []
        with torch.no_grad():
            for emb_batch, tgt_batch in val_loader:
                emb_batch = emb_batch.to(device)
                tgt_batch = tgt_batch.to(device)
                logits = model.net(emb_batch)
                val_loss += criterion(logits, tgt_batch).item() * len(emb_batch)
                all_probs.append(torch.sigmoid(logits).cpu())
                all_targets.append(tgt_batch.cpu())
        val_loss /= val_size

        probs = torch.cat(all_probs, dim=0)   # (val_size, G)
        tgts = torch.cat(all_targets, dim=0)  # (val_size, G)
        macro_f1 = _macro_f1(probs, tgts, threshold)

        is_best = macro_f1 > best_f1
        if is_best:
            best_f1 = macro_f1
            best_epoch = epoch
            model.save(out_path, group_names)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"{epoch:>5}  {train_loss:>10.4f}  {val_loss:>9.4f}  {macro_f1:>9.4f}  {current_lr:>9.2e}  {'*' if is_best else ''}")

        if scheduler is not None:
            scheduler.step()

    print(f"\nBest: epoch {best_epoch}, macro F1 = {best_f1:.4f}")
    print(f"Saved: {out_path}")

    # --- Per-group breakdown on validation set (using best checkpoint) -------
    best_model, _ = GroupClassifier.load(out_path)
    best_model.to(device)
    best_model.eval()
    all_probs, all_targets = [], []
    with torch.no_grad():
        for emb_batch, tgt_batch in val_loader:
            all_probs.append(best_model.predict_proba(emb_batch.to(device)).cpu())
            all_targets.append(tgt_batch)
    probs = torch.cat(all_probs, dim=0)
    tgts = torch.cat(all_targets, dim=0)
    _per_group_stats(probs, tgts, group_names, threshold)

    # --- Save production checkpoint if this run beats the previous best -------
    production_path = Path(out_path).parent / "group_classifier_best.pt"
    meta_path = production_path.with_suffix(".meta.json")
    prev_best_f1 = 0.0
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            prev_best_f1 = json.load(f).get("best_f1", 0.0)
    if best_f1 > prev_best_f1:
        shutil.copy2(out_path, production_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"best_f1": best_f1, "best_epoch": best_epoch}, f)
        print(f"\n* New best macro F1: {best_f1:.4f} -- checkpoint saved to {production_path}")
    else:
        print(f"\n  Production checkpoint unchanged (best F1 {prev_best_f1:.4f} > this run {best_f1:.4f})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the GroupClassifier.")
    parser.add_argument(
        "--training-data",
        default=config.GROUP_TRAINING_DATA_NPZ,
        help="Path to training data npz from build_group_training_data.py",
    )
    parser.add_argument(
        "--out",
        default=f"{config.CHECKPOINT_GROUP_DIR}/group_classifier_current.pt",
        help="Output checkpoint path",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs (default: 50)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM,
                        help=f"Hidden layer size (default: {DEFAULT_HIDDEN_DIM})")
    parser.add_argument("--val-frac", type=float, default=0.2, help="Validation fraction (default: 0.2)")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Group probability threshold for F1 evaluation and inference (default: 0.3)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps", "xpu"],
        help="Compute device (default: auto)",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-3, help="Adam weight decay (default: 1e-3)")
    parser.add_argument("--dropout", type=float, default=0.3, help="MLP dropout probability (default: 0.3)")
    parser.add_argument(
        "--max-class-weight",
        type=float,
        default=500.0,
        help="Cap per-class BCE pos_weight at this value (default: 500). 0 = no cap.",
    )
    parser.add_argument(
        "--min-group-cases",
        type=int,
        default=10,
        help="Drop groups with fewer than this many positive cases (default: 10).",
    )
    parser.add_argument(
        "--max-group-cases",
        type=int,
        default=0,
        help="Cap positive training samples per group at this value (default: 0 = no cap). "
             "Rows are removed entirely (not label-zeroed) to avoid false negatives.",
    )
    parser.add_argument(
        "--lr-schedule",
        default="none",
        choices=["none", "cosine"],
        help="LR schedule (default: none). 'cosine' uses CosineAnnealingWarmRestarts(T_0=100) "
             "to escape fixed-LR plateaus. Recommended: try with --epochs 600.",
    )
    parser.add_argument(
        "--focal-loss",
        action="store_true",
        default=False,
        help="Use Focal Loss instead of BCEWithLogitsLoss. Down-weights easy non-cancer "
             "examples and focuses gradient on hard group assignments. See --focal-gamma.",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Focal loss focusing parameter (default: 2.0). Higher = more focus on hard "
             "examples. Typical range: 1.0-5.0. Only used when --focal-loss is set.",
    )
    parser.add_argument(
        "--asl",
        action="store_true",
        default=False,
        help="Use Asymmetric Loss (ASL) instead of BCEWithLogitsLoss. Applies separate "
             "focusing for positives (--asl-gamma-pos) and negatives (--asl-gamma-neg), "
             "with a probability floor (--asl-margin) to suppress easy-negative gradients.",
    )
    parser.add_argument(
        "--asl-gamma-pos",
        type=float,
        default=1.0,
        help="ASL focusing exponent for positives (default: 1.0).",
    )
    parser.add_argument(
        "--asl-gamma-neg",
        type=float,
        default=4.0,
        help="ASL focusing exponent for negatives (default: 4.0). Higher = stronger "
             "suppression of easy negatives.",
    )
    parser.add_argument(
        "--asl-margin",
        type=float,
        default=0.05,
        help="ASL probability floor: negative probabilities are shifted down by this "
             "amount before the log, zeroing gradient for very confident negatives (default: 0.05).",
    )
    args = parser.parse_args()
    train(
        training_data_path=args.training_data,
        out_path=args.out,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        val_frac=args.val_frac,
        threshold=args.threshold,
        device_arg=args.device,
        weight_decay=args.weight_decay,
        max_class_weight=args.max_class_weight,
        min_group_cases=args.min_group_cases,
        max_group_cases=args.max_group_cases,
        dropout=args.dropout,
        lr_schedule=args.lr_schedule,
        focal_loss=args.focal_loss,
        focal_gamma=args.focal_gamma,
        asl=args.asl,
        asl_gamma_pos=args.asl_gamma_pos,
        asl_gamma_neg=args.asl_gamma_neg,
        asl_margin=args.asl_margin,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
