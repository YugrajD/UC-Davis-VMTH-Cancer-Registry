"""Contrastive PetBERT backbone adaptation.

Faithful port of ``ml/training/contrastive/train_contrastive.py``: InfoNCE
with in-batch negatives over per-section (report_text, label_text) pairs
(``labels.contrastive_pairs``), mean-pool + L2-normalise, symmetric
cross-entropy, linear warmup then linear decay, gradient clipping. Legacy
seeded nothing at all here (ml-rewrite-plan.md, Findings — "contrastive
trainer unseeded"); ``recipe.seed_all`` seeds every RNG before this trainer
shuffles its pair loader.

Saves the full HuggingFace checkpoint (model + tokenizer) to
``<out_dir>/petbert/`` (default: the candidate's own bundled backbone), which
the head trainers then read via ``report_mapping.training.embeddings.
get_or_build``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer

import config
import io_utils
from report_mapping.model import backbone as backbone_mod
from report_mapping.model.generation import generation_paths
from report_mapping.training import labels as labels_mod
from report_mapping.training import recipe

DEFAULT_MODEL_NAME = "SAVSNET/PetBERT"


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _infonce_loss(report_emb: torch.Tensor, label_emb: torch.Tensor, temperature: float) -> torch.Tensor:
    sim = report_emb @ label_emb.T / temperature
    n = sim.shape[0]
    targets = torch.arange(n, device=sim.device)
    return (F.cross_entropy(sim, targets) + F.cross_entropy(sim.T, targets)) / 2.0


def _embed_pair(model: AutoModelForMaskedLM, enc: dict, device: torch.device) -> torch.Tensor:
    hidden = model.base_model(
        input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device),
    ).last_hidden_state
    return F.normalize(_mean_pool(hidden, enc["attention_mask"].to(device)), dim=-1)


def _dataset_loss(model, tokenizer, dataset: "_PairDataset", batch_size: int, max_length: int,
                   temperature: float, device: torch.device) -> float:
    """Average InfoNCE loss over the whole dataset, batched but unshuffled —
    an eval-mode, order-independent-ish measurement used only to compare
    "before training" against "after training" (the real training loop uses
    its own shuffled, dropout-affected loader)."""
    was_training = model.training
    model.eval()
    eval_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=_make_collator(tokenizer, max_length),
    )
    total, n = 0.0, 0
    with torch.no_grad():
        for report_enc, label_enc in eval_loader:
            if report_enc["input_ids"].shape[0] < 2:
                continue  # InfoNCE needs >= 2 examples to have any negatives
            r_emb = _embed_pair(model, report_enc, device)
            l_emb = _embed_pair(model, label_enc, device)
            total += _infonce_loss(r_emb, l_emb, temperature).item()
            n += 1
    if was_training:
        model.train()
    return total / max(1, n)


class _PairDataset(Dataset):
    def __init__(self, pairs: pd.DataFrame):
        self.pairs = list(zip(pairs["report_text"].tolist(), pairs["label_text"].tolist()))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        return self.pairs[idx]


def _make_collator(tokenizer: AutoTokenizer, max_length: int):
    def collate(batch: list[tuple[str, str]]):
        report_texts = [item[0] for item in batch]
        label_texts = [item[1] for item in batch]
        report_enc = tokenizer(report_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        label_enc = tokenizer(label_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        return report_enc, label_enc
    return collate


def train(
    labels: pd.DataFrame,
    split_id: str,
    seed: int,
    device: str,
    *,
    model_name: str | None = None,
    local_only: bool = True,
    out_dir: str | Path | None = None,
    report_frame: pd.DataFrame | None = None,
) -> dict:
    """Fine-tune PetBERT on per-section (report, label) pairs built from
    ``labels`` filtered to ``split_id``'s train partition.

    ``report_frame`` defaults to reading ``config.REPORT_CSV``; tests pass a
    synthetic frame directly. Saves the full HF checkpoint to
    ``<out_dir>/petbert/`` (default: ``config.REPORT_MAPPING_CANDIDATE_DIR``).
    """
    r = recipe.BACKBONE
    recipe.seed_all(seed)
    out_dir = Path(out_dir) if out_dir is not None else config.REPORT_MAPPING_CANDIDATE_DIR
    model_name = model_name if model_name is not None else DEFAULT_MODEL_NAME

    if report_frame is None:
        report_frame = io_utils.read_csv(config.REPORT_CSV, encoding="latin-1")

    labels_train = labels_mod.select_train(labels, split_id)
    pairs = labels_mod.contrastive_pairs(labels_train, report_frame)
    if pairs.empty:
        raise ValueError("No contrastive pairs built from the train partition; nothing to train on")

    dev = backbone_mod.device_from_arg(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_only)
    model = AutoModelForMaskedLM.from_pretrained(model_name, local_files_only=local_only)
    model.to(dev)
    model.train()

    dataset = _PairDataset(pairs)
    # drop_last=True: InfoNCE targets are torch.arange(batch_size); a short final batch would be wrong-sized.
    loader = DataLoader(
        dataset, batch_size=r.batch_size, shuffle=True,
        collate_fn=_make_collator(tokenizer, r.max_length), drop_last=True,
    )
    if len(loader) == 0:
        raise ValueError(f"Fewer than batch_size={r.batch_size} pairs ({len(dataset)} total); nothing to train on")

    optimizer = torch.optim.AdamW(model.parameters(), lr=r.lr, weight_decay=r.weight_decay)
    total_steps = len(loader) * r.epochs
    warmup_steps = max(1, int(total_steps * r.warmup_frac))

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 1.0 - progress)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    initial_loss = _dataset_loss(model, tokenizer, dataset, r.batch_size, r.max_length, r.temperature, dev)

    last_loss = float("nan")
    for epoch in range(1, r.epochs + 1):
        epoch_loss, n_batches = 0.0, 0
        for report_enc, label_enc in loader:
            report_emb = _embed_pair(model, report_enc, dev)
            label_emb = _embed_pair(model, label_enc, dev)
            loss = _infonce_loss(report_emb, label_emb, r.temperature)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            n_batches += 1
        last_loss = epoch_loss / max(1, n_batches)
        print(f"backbone: epoch {epoch}/{r.epochs} avg loss {last_loss:.4f}")

    final_loss = _dataset_loss(model, tokenizer, dataset, r.batch_size, r.max_length, r.temperature, dev)
    print(f"backbone: dataset loss {initial_loss:.4f} -> {final_loss:.4f} (last epoch avg {last_loss:.4f})")

    petbert_dir = generation_paths(out_dir).petbert_dir
    petbert_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(petbert_dir))
    tokenizer.save_pretrained(str(petbert_dir))

    return {
        "n_pairs": len(pairs), "initial_loss": initial_loss, "final_loss": final_loss,
        "petbert_dir": str(petbert_dir),
    }
