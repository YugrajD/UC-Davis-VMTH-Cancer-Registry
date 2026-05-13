"""SimCSE: unsupervised contrastive fine-tuning of PetBERT.

Each input text is passed through the encoder TWICE with different dropout
masks. The two resulting embeddings form a positive pair. All other texts
in the batch are in-batch negatives. The InfoNCE loss pushes the two views
of the same text together while spreading apart different texts.

This requires NO labels — only raw report text. The result is an encoder
whose embedding space clusters semantically similar reports, enabling
kNN retrieval against label definitions.

Reference: Gao et al. "SimCSE: Simple Contrastive Learning of Sentence
Embeddings" (EMNLP 2021).

Usage:
  python ml/training/simcse/train_simcse.py --epochs 3 --device cpu
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
from model.constants import DEFAULT_TEXT_COLS
from production.petbert_pipeline import device_from_arg


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class ReportTextDataset(Dataset):
    """Loads report texts from the report CSV, concatenating text columns."""

    def __init__(self, csv_path: str, text_cols: tuple[str, ...], cases_txt: str = ""):
        filter_ids = None
        if cases_txt and Path(cases_txt).exists():
            with open(cases_txt) as f:
                filter_ids = {line.strip() for line in f if line.strip()}

        self.texts: list[str] = []
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if filter_ids and row["case_id"] not in filter_ids:
                    continue
                parts = [row.get(col, "").strip() for col in text_cols]
                combined = " ".join(p for p in parts if p)
                if combined:
                    self.texts.append(combined)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]


def train_simcse(
    model_name: str,
    out_dir: str,
    *,
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 3e-5,
    temperature: float = 0.05,
    max_length: int = 512,
    device_arg: str = "cpu",
    local_only: bool = True,
    cases_txt: str = "",
) -> None:
    device = device_from_arg(device_arg)
    print(f"SimCSE training | device={device} | epochs={epochs} | batch_size={batch_size}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_only)
    model = AutoModelForMaskedLM.from_pretrained(model_name, local_files_only=local_only)
    model.to(device)
    model.train()

    dataset = ReportTextDataset(config.REPORTS_CSV, DEFAULT_TEXT_COLS, cases_txt=cases_txt)
    print(f"Training texts: {len(dataset)}")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)

    optimizer = torch.optim.AdamW(model.base_model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(loader))

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", unit="batch")
        for batch_texts in pbar:
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            # Pass 1: get embeddings with one dropout mask
            out1 = model.base_model(input_ids=input_ids, attention_mask=attention_mask)
            emb1 = _mean_pool(out1.last_hidden_state, attention_mask)

            # Pass 2: same input, different dropout mask
            out2 = model.base_model(input_ids=input_ids, attention_mask=attention_mask)
            emb2 = _mean_pool(out2.last_hidden_state, attention_mask)

            # L2 normalize
            emb1 = F.normalize(emb1, dim=-1)
            emb2 = F.normalize(emb2, dim=-1)

            # Cosine similarity matrix scaled by temperature
            sim = emb1 @ emb2.T / temperature  # (B, B)
            labels = torch.arange(sim.size(0), device=device)
            loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        print(f"  Epoch {epoch} avg loss: {avg_loss:.4f}")

    # Save the full model checkpoint
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_path))
    tokenizer.save_pretrained(str(out_path))
    print(f"SimCSE model saved to {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="SimCSE unsupervised contrastive training.")
    parser.add_argument("--model", default="ml/output/checkpoints/contrastive",
                        help="Base model to fine-tune (default: adapted backbone)")
    parser.add_argument("--out-dir", default="ml/output/checkpoints/simcse",
                        help="Output directory for the SimCSE checkpoint")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda", "mps", "xpu"])
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--train-cases", default="",
                        help="Only use training split texts (recommended)")
    args = parser.parse_args()

    train_simcse(
        model_name=args.model,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        temperature=args.temperature,
        max_length=args.max_length,
        device_arg=args.device,
        local_only=args.local_only,
        cases_txt=args.train_cases,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
