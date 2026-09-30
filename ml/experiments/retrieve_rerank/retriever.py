"""ModernBERT encoder -> mean pool -> linear head with one logit per diagnosis term, plus the two losses.

Mean pooling rather than CLS: BioClinical ModernBERT is MLM-pretrained (no CLS objective), and the
diagnosis is stated anywhere in a multi-thousand-token report, so averaging all token states is the
more robust read-out (and matches how the repo already pools PetBERT).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

ENCODER_ID = "thomas-sounack/BioClinical-ModernBERT-base"
HEAD_FILE, TERMS_FILE = "head.pt", "terms.json"


class TermRetriever(nn.Module):
    def __init__(self, encoder, terms: list[str]):
        super().__init__()
        self.encoder = encoder
        self.terms = terms
        self.head = nn.Linear(encoder.config.hidden_size, len(terms))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.head(pooled.float())

    def init_bias_from_frequency(self, freq: torch.Tensor) -> None:
        """Start every term at its train prior so early steps aren't spent learning the base rate."""
        p = freq.clamp(1e-6, 1 - 1e-6)
        with torch.no_grad():
            self.head.bias.copy_(torch.log(p / (1 - p)))

    def save(self, directory: Path, tokenizer) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(directory / "encoder")
        tokenizer.save_pretrained(directory / "encoder")
        torch.save(self.head.state_dict(), directory / HEAD_FILE)
        (directory / TERMS_FILE).write_text(json.dumps(self.terms), encoding="utf-8")


def new_encoder(model_id: str = ENCODER_ID):
    # sdpa: flash-attn is not available on Windows. reference_compile off: it needs triton.
    config = AutoConfig.from_pretrained(model_id)
    config.reference_compile = False
    return AutoModel.from_pretrained(model_id, config=config, attn_implementation="sdpa")


def load_tokenizer(model_id: str = ENCODER_ID):
    return AutoTokenizer.from_pretrained(model_id)


def load_retriever(directory: Path) -> tuple[TermRetriever, object]:
    terms = json.loads((directory / TERMS_FILE).read_text(encoding="utf-8"))
    model = TermRetriever(new_encoder(str(directory / "encoder")), terms)
    model.head.load_state_dict(torch.load(directory / HEAD_FILE, map_location="cpu"))
    return model, load_tokenizer(str(directory / "encoder"))


def bce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """BCEWithLogits over the full multi-hot vector: summed over terms, averaged over cases."""
    return F.binary_cross_entropy_with_logits(logits, targets, reduction="none").sum(1).mean()


def softmax_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Per-positive softmax CE. For each gold term g of a case: -log softmax over {g} + the case's
    NON-gold terms (the other gold terms are masked out, so they are never pushed down as negatives).
    Averaged over a case's gold terms, then over cases. Every case needs >= 1 gold term."""
    gold = targets > 0
    neg_lse = torch.logsumexp(logits.masked_fill(gold, float("-inf")), dim=1, keepdim=True)
    per_pair = torch.logaddexp(neg_lse, logits) - logits  # -log( e^z_g / (e^z_g + sum_neg e^z) )
    per_case = (per_pair * gold).sum(1) / gold.sum(1)
    return per_case.mean()


LOSSES = {"bce": bce_loss, "softmax": softmax_loss}
