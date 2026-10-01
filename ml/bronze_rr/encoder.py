"""ModernBERT encoder -> mean pool -> fp32 linear head. The one network shape behind the retriever
(one logit per diagnosis term) and the gate (one logit: is there a cancer diagnosis).

Mean pooling rather than CLS: BioClinical ModernBERT is MLM-pretrained (no CLS objective), and the
diagnosis is stated anywhere in a multi-thousand-token report, so averaging all token states is the
more robust read-out.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

ENCODER_ID = "thomas-sounack/BioClinical-ModernBERT-base"
ENCODER_DIR, HEAD_FILE = "encoder", "head.pt"


class PooledClassifier(nn.Module):
    def __init__(self, encoder, n_out: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.config.hidden_size, n_out)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        with torch.autocast(device_type=pooled.device.type, enabled=False):  # fp32 head: bf16 logits tie ranks
            return self.head(pooled.float())

    def init_bias_from_frequency(self, freq: torch.Tensor) -> None:
        """Start every output at its train prior so early steps aren't spent learning the base rate."""
        p = freq.clamp(1e-6, 1 - 1e-6)
        with torch.no_grad():
            self.head.bias.copy_(torch.log(p / (1 - p)))

    def save(self, directory: Path, tokenizer) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(directory / ENCODER_DIR)
        tokenizer.save_pretrained(directory / ENCODER_DIR)
        torch.save(self.head.state_dict(), directory / HEAD_FILE)


def new_encoder(model_id: str = ENCODER_ID):
    # sdpa: flash-attn is not available on Windows. reference_compile off: it needs triton.
    config = AutoConfig.from_pretrained(model_id)
    config.reference_compile = False
    return AutoModel.from_pretrained(model_id, config=config, attn_implementation="sdpa")


def load_tokenizer(model_id: str = ENCODER_ID):
    return AutoTokenizer.from_pretrained(model_id)


def load_saved(directory: Path, n_out: int) -> tuple[PooledClassifier, object]:
    """(model, tokenizer) saved by ``PooledClassifier.save`` into ``directory``."""
    model = PooledClassifier(new_encoder(str(directory / ENCODER_DIR)), n_out)
    model.head.load_state_dict(torch.load(directory / HEAD_FILE, map_location="cpu"))
    return model, load_tokenizer(str(directory / ENCODER_DIR))


def bce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """BCEWithLogits over the full multi-hot vector: summed over outputs, averaged over cases."""
    return F.binary_cross_entropy_with_logits(logits, targets, reduction="none").sum(1).mean()
