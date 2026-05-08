"""Contrastive fine-tuning of PetBERT using InfoNCE (in-batch negatives).

For each batch of N (report_text, label_text) pairs:
  1. Embed report texts and label texts through PetBERT's base transformer,
     mean-pooling over non-padding tokens → 768-dim vectors.
  2. L2-normalise both sets of embeddings.
  3. Compute an (N, N) cosine similarity matrix scaled by temperature.
  4. Symmetric cross-entropy loss: rows (report → label) and columns
     (label → report) both target the diagonal.
  5. Backpropagate through PetBERT's base transformer weights only
     (the MLM head receives no gradients since it is not called).

After training, the full AutoModelForMaskedLM checkpoint is saved to
--out-dir. The production pipeline can then use it by passing:
    --model <out-dir> --local-only

A cold start is required after fine-tuning: delete the embedding cache,
CO bank, and current classifier checkpoint before retraining the
LabelPresenceClassifier from scratch with the new backbone.
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import config
from production.petbert_pipeline import device_from_arg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool over non-padding tokens. Matches embedding.py exactly."""
    mask = attention_mask.unsqueeze(-1).float()          # (B, T, 1)
    summed = (last_hidden_state * mask).sum(dim=1)       # (B, 768)
    counts = mask.sum(dim=1).clamp(min=1e-9)             # (B, 1)
    return summed / counts                                # (B, 768)


def _hard_neg_loss(
    report_emb: torch.Tensor,
    correct_emb: torch.Tensor,
    wrong_emb: torch.Tensor,
    temperature: float,
    margin: float,
) -> torch.Tensor:
    """Margin loss that pushes each report away from its known wrong label.

    For each triplet (report, correct_label, wrong_label):
        loss = relu(sim(report, wrong) - sim(report, correct) + margin)

    All inputs are assumed to be L2-normalised. The margin (default 0.3) sets
    the minimum required gap between positive and negative cosine similarities
    scaled by temperature. Loss is zero when correct is already > wrong + margin.
    """
    pos_sim = (report_emb * correct_emb).sum(dim=-1) / temperature   # (M,)
    neg_sim = (report_emb * wrong_emb).sum(dim=-1) / temperature     # (M,)
    return F.relu(neg_sim - pos_sim + margin).mean()


def _infonce_loss(
    report_emb: torch.Tensor,
    label_emb: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Symmetric InfoNCE loss over a batch of L2-normalised embeddings.

    Both inputs are assumed to already be L2-normalised (unit vectors).
    The diagonal of the (N, N) similarity matrix represents positive pairs;
    all off-diagonal entries are treated as negatives.

    Note on false negatives: when the same label appears twice in a batch
    the off-diagonal entry for that pair is a false negative. At batch_size=32
    and 857 unique labels this collision rate is low (~4%) and acceptable.
    """
    sim = report_emb @ label_emb.T / temperature   # (N, N)
    n = sim.shape[0]
    targets = torch.arange(n, device=sim.device)
    loss_r2l = F.cross_entropy(sim, targets)        # each report → its label
    loss_l2r = F.cross_entropy(sim.T, targets)      # each label → its report
    return (loss_r2l + loss_l2r) / 2.0


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _load_csv_rows(path: str, fields: list[str]) -> list[tuple[str, ...]]:
    """Read rows from a CSV, keeping only rows where every field is non-empty."""
    rows: list[tuple[str, ...]] = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vals = tuple(row.get(fld, "").strip() for fld in fields)
            if all(vals):
                rows.append(vals)
    return rows


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ContrastivePairDataset(Dataset):
    """Loads (report_text, label_text) pairs from the CSV built by
    build_contrastive_dataset.py. Tokenization is deferred to the collator
    so that each batch can be padded to its own maximum length."""

    def __init__(self, pairs_csv: str) -> None:
        self.pairs = _load_csv_rows(pairs_csv, ["report_text", "label_text"])
        print(f"  Loaded {len(self.pairs)} pairs from {pairs_csv}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        return self.pairs[idx]  # type: ignore[return-value]


class HardNegPairDataset(Dataset):
    """Loads (report_text, correct_label_text, wrong_label_text) triplets from
    the CSV built by build_hard_neg_pairs() in build_contrastive_dataset.py.

    Each triplet represents a case where the model was previously observed to
    predict the wrong label for a report that has a known correct label."""

    def __init__(self, hard_neg_csv: str) -> None:
        self.triplets = _load_csv_rows(
            hard_neg_csv, ["report_text", "correct_label_text", "wrong_label_text"]
        )
        print(f"  Loaded {len(self.triplets)} hard-negative triplets from {hard_neg_csv}")

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> tuple[str, str, str]:
        return self.triplets[idx]  # type: ignore[return-value]


def _make_collator(tokenizer: AutoTokenizer, max_length: int):
    """Return a collate_fn that tokenises a batch of (report, label) pairs."""
    def collate(batch: list[tuple[str, str]]) -> dict[str, torch.Tensor]:
        report_texts = [item[0] for item in batch]
        label_texts  = [item[1] for item in batch]

        report_enc = tokenizer(
            report_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        label_enc = tokenizer(
            label_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {
            "report_input_ids":      report_enc["input_ids"],
            "report_attention_mask": report_enc["attention_mask"],
            "label_input_ids":       label_enc["input_ids"],
            "label_attention_mask":  label_enc["attention_mask"],
        }
    return collate


def _make_hard_neg_collator(tokenizer: AutoTokenizer, max_length: int):
    """Return a collate_fn for (report, correct_label, wrong_label) triplets."""
    def collate(batch: list[tuple[str, str, str]]) -> dict[str, torch.Tensor]:
        report_texts  = [item[0] for item in batch]
        correct_texts = [item[1] for item in batch]
        wrong_texts   = [item[2] for item in batch]

        def enc(texts: list[str]) -> dict[str, torch.Tensor]:
            return tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

        r = enc(report_texts)
        c = enc(correct_texts)
        w = enc(wrong_texts)
        return {
            "report_input_ids":        r["input_ids"],
            "report_attention_mask":   r["attention_mask"],
            "correct_input_ids":       c["input_ids"],
            "correct_attention_mask":  c["attention_mask"],
            "wrong_input_ids":         w["input_ids"],
            "wrong_attention_mask":    w["attention_mask"],
        }
    return collate


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    *,
    pairs_csv: str = config.CONTRASTIVE_PAIRS_CSV,
    out_dir: str = config.CHECKPOINT_CONTRASTIVE_DIR,
    model_name: str = "SAVSNET/PetBERT",
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 2e-5,
    temperature: float = 0.07,
    max_length: int = 256,
    device_arg: str = "auto",
    local_only: bool = False,
    hard_neg_csv: str | None = None,
    hard_neg_weight: float = 0.5,
    hard_neg_margin: float = 0.3,
) -> None:

    device = device_from_arg(device_arg)
    print(f"Device: {device}")

    # --- Load tokenizer and model -------------------------------------------
    print(f"Loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_only)
    model = AutoModelForMaskedLM.from_pretrained(model_name, local_files_only=local_only)
    model.to(device)
    model.train()

    # --- Dataset and loader -------------------------------------------------
    print(f"Loading pairs from {pairs_csv}...")
    dataset = ContrastivePairDataset(pairs_csv)
    if len(dataset) == 0:
        raise ValueError(f"No pairs found in {pairs_csv}. Run build_contrastive_dataset.py first.")

    # drop_last=True: InfoNCE targets are torch.arange(batch_size); an incomplete
    # final batch would produce wrong-sized targets.
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_make_collator(tokenizer, max_length),
        drop_last=True,
    )

    # Optional hard-negative loader (interleaved one batch per positive batch).
    # Stored in a list so the inner loop can reassign [0] when the iterator
    # is exhausted and cycles back to the start.
    hard_neg_loader = None
    hn_iter_box: list = []  # hn_iter_box[0] = current iterator, or empty if disabled
    if hard_neg_csv is not None:
        print(f"Loading hard-negative triplets from {hard_neg_csv}...")
        hn_dataset = HardNegPairDataset(hard_neg_csv)
        if len(hn_dataset) == 0:
            print("  Warning: hard-neg CSV is empty — skipping hard-neg loss.")
        else:
            hard_neg_loader = DataLoader(
                hn_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=_make_hard_neg_collator(tokenizer, max_length),
                drop_last=True,
            )
            hn_iter_box.append(iter(hard_neg_loader))
            print(f"  Hard-neg weight={hard_neg_weight}, margin={hard_neg_margin}")

    # --- Optimiser and scheduler --------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    total_steps  = len(loader) * epochs
    warmup_steps = max(1, int(total_steps * 0.06))

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 1.0 - progress)   # linear decay to 0 after warmup

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    # --- Training loop ------------------------------------------------------
    hn_label = f", hard-neg weight={hard_neg_weight}" if hard_neg_loader else ""
    print(
        f"\nContrastive fine-tuning: epochs={epochs}, batch={batch_size}, "
        f"lr={lr}, temperature={temperature}, "
        f"steps={total_steps} ({warmup_steps} warmup){hn_label}\n"
    )

    def _embed(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = model.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state
        return F.normalize(_mean_pool(hidden, attention_mask), dim=-1)

    global_step = 0
    epoch_pbar = tqdm(range(1, epochs + 1), desc="Contrastive training", unit="epoch")
    for epoch in epoch_pbar:
        epoch_loss = 0.0
        epoch_hn_loss = 0.0
        n_batches = 0

        pbar = tqdm(loader, desc=f"  Epoch {epoch}/{epochs}", unit="batch", leave=False)
        for batch in pbar:
            report_input_ids      = batch["report_input_ids"].to(device)
            report_attention_mask = batch["report_attention_mask"].to(device)
            label_input_ids       = batch["label_input_ids"].to(device)
            label_attention_mask  = batch["label_attention_mask"].to(device)

            # Forward through PetBERT base transformer only — MLM head is never
            # called so its parameters receive no gradients and are not updated.
            report_emb = _embed(report_input_ids, report_attention_mask)
            label_emb  = _embed(label_input_ids,  label_attention_mask)

            loss = _infonce_loss(report_emb, label_emb, temperature)

            # Hard-negative margin loss (optional)
            if hn_iter_box:
                try:
                    hn_batch = next(hn_iter_box[0])
                except StopIteration:
                    # Hard-neg dataset is shorter than positive dataset — cycle back
                    hn_iter_box[0] = iter(hard_neg_loader)
                    hn_batch = next(hn_iter_box[0])

                hn_r  = _embed(hn_batch["report_input_ids"].to(device),
                               hn_batch["report_attention_mask"].to(device))
                hn_c  = _embed(hn_batch["correct_input_ids"].to(device),
                               hn_batch["correct_attention_mask"].to(device))
                hn_w  = _embed(hn_batch["wrong_input_ids"].to(device),
                               hn_batch["wrong_attention_mask"].to(device))

                hn_loss = _hard_neg_loss(hn_r, hn_c, hn_w, temperature, hard_neg_margin)
                loss = loss + hard_neg_weight * hn_loss
                epoch_hn_loss += hn_loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1
            epoch_loss += loss.item()
            n_batches  += 1

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        avg = epoch_loss / max(1, n_batches)
        postfix: dict = {"avg_loss": f"{avg:.4f}"}
        hn_suffix = ""
        if hn_iter_box:
            avg_hn = epoch_hn_loss / max(1, n_batches)
            hn_suffix = f", avg hard-neg loss: {avg_hn:.4f}"
            postfix["avg_hn_loss"] = f"{avg_hn:.4f}"
        epoch_pbar.set_postfix(postfix)
        print(f"Epoch {epoch}/{epochs} complete — avg loss: {avg:.4f}{hn_suffix}")

    # --- Save checkpoint ----------------------------------------------------
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving fine-tuned checkpoint to {out_path} ...")
    model.save_pretrained(str(out_path))
    tokenizer.save_pretrained(str(out_path))

    print("\nDone.")
    print("Next steps (cold start required — embedding space has changed):")
    print("  1. rm -f ml/output/training/embedding_cache.npz")
    print("  2. Retrain in order: --mode train-case-presence, --mode train-groups,")
    print(f"     --mode train-label-presence (each with --model {out_dir} --local-only).")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Contrastive InfoNCE fine-tuning of PetBERT on (report, label) pairs."
    )
    parser.add_argument("--pairs-csv", default=config.CONTRASTIVE_PAIRS_CSV,
                        help="Path to CSV from build_contrastive_dataset.py")
    parser.add_argument("--out-dir", default=config.CHECKPOINT_CONTRASTIVE_DIR,
                        help="Directory to save the fine-tuned HuggingFace checkpoint")
    parser.add_argument("--model", default="SAVSNET/PetBERT",
                        help="Base model name or local path (default: SAVSNET/PetBERT)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Training epochs (default: 3; keep low to avoid overfitting)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (default: 32; larger = more in-batch negatives)")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Peak learning rate (default: 2e-5)")
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="InfoNCE temperature (default: 0.07; lower = harder negatives)")
    parser.add_argument("--max-length", type=int, default=256,
                        help="Max token length for report and label texts (default: 256)")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda", "mps", "xpu"],
                        help="Compute device (default: auto)")
    parser.add_argument("--local-only", action="store_true",
                        help="Use only locally cached model files (no HuggingFace download)")
    parser.add_argument("--hard-neg-csv", default=None,
                        help="Path to hard-negative triplets CSV from build_hard_neg_pairs(). "
                             "If provided, adds a margin loss that pushes reports away from "
                             "their known wrong-group labels alongside InfoNCE. (default: disabled)")
    parser.add_argument("--hard-neg-weight", type=float, default=0.5,
                        help="Weight applied to the hard-negative margin loss (default: 0.5)")
    parser.add_argument("--hard-neg-margin", type=float, default=0.3,
                        help="Margin for the hard-negative loss: loss fires when "
                             "sim(report, wrong) > sim(report, correct) - margin (default: 0.3)")
    args = parser.parse_args(argv)

    train(
        pairs_csv=args.pairs_csv,
        out_dir=args.out_dir,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        temperature=args.temperature,
        max_length=args.max_length,
        device_arg=args.device,
        local_only=args.local_only,
        hard_neg_csv=args.hard_neg_csv,
        hard_neg_weight=args.hard_neg_weight,
        hard_neg_margin=args.hard_neg_margin,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
