"""One ModernBERT trained jointly for the term head and the case-presence gate.

Encoder -> mean pool -> two fp32 linear heads: a term head (one logit per diagnosis term) and a gate head (one
logit: is there a cancer diagnosis). One encoder pass per report gives both outputs.

Mean pooling rather than CLS: BioClinical ModernBERT is MLM-pretrained (no CLS objective), and the
diagnosis is stated anywhere in a multi-thousand-token report, so averaging all token states is the
more robust read-out.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from report_mapping.generation import ENCODER_DIR, GATE_HEAD_FILE, HEAD_FILE, TERMS_FILE

ENCODER_ID = "thomas-sounack/BioClinical-ModernBERT-base"


def resolve_device(name: str) -> str:
    """"auto" -> cuda when available, else cpu; "cpu" and "cuda" as given."""
    return ("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else name


def new_encoder(model_id: str = ENCODER_ID):
    # sdpa: flash-attn is not available on Windows. reference_compile off: it needs triton.
    config = AutoConfig.from_pretrained(model_id)
    config.reference_compile = False
    return AutoModel.from_pretrained(model_id, config=config, attn_implementation="sdpa")


def load_tokenizer(model_id: str = ENCODER_ID):
    return AutoTokenizer.from_pretrained(model_id)


class JointClassifier(nn.Module):
    def __init__(self, encoder, terms: list[str]):
        super().__init__()
        self.encoder = encoder
        self.terms = terms
        hidden = encoder.config.hidden_size
        self.head = nn.Linear(hidden, len(terms))
        self.gate_head = nn.Linear(hidden, 1)

    def pooled(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)

    def heads_logits(self, pooled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(term logits (N, n_terms), gate logits (N,)), fp32: bf16 logits tie ranks."""
        with torch.autocast(device_type=pooled.device.type, enabled=False):
            pooled = pooled.float()
            return self.head(pooled), self.gate_head(pooled)[:, 0]

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.heads_logits(self.pooled(input_ids, attention_mask))

    def init_biases(self, term_freq: torch.Tensor, gate_rate: float) -> None:
        """Start every output at its train prior (term frequency among cancer cases; positive rate for the gate)."""
        for layer, p in ((self.head, term_freq), (self.gate_head, torch.tensor([gate_rate]))):
            p = p.to(layer.bias.device).clamp(1e-6, 1 - 1e-6)
            with torch.no_grad():
                layer.bias.copy_(torch.log(p / (1 - p)))

    def save(self, directory: Path, tokenizer) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(directory / ENCODER_DIR)
        tokenizer.save_pretrained(directory / ENCODER_DIR)
        torch.save(self.head.state_dict(), directory / HEAD_FILE)
        torch.save(self.gate_head.state_dict(), directory / GATE_HEAD_FILE)
        (directory / TERMS_FILE).write_text(json.dumps(self.terms), encoding="utf-8")


def load_joint(directory: Path) -> tuple[JointClassifier, object]:
    """(model, tokenizer) saved by ``JointClassifier.save``. Refuses a head that does not match terms.json."""
    terms = json.loads((directory / TERMS_FILE).read_text(encoding="utf-8"))
    head = torch.load(directory / HEAD_FILE, map_location="cpu", weights_only=True)
    if head["weight"].shape[0] != len(terms):
        raise ValueError(f"{directory}: terms.json has {len(terms)} terms but the term head has "
                         f"{head['weight'].shape[0]} outputs")
    model = JointClassifier(new_encoder(str(directory / ENCODER_DIR)), terms)
    model.head.load_state_dict(head)
    model.gate_head.load_state_dict(torch.load(directory / GATE_HEAD_FILE, map_location="cpu", weights_only=True))
    return model, load_tokenizer(str(directory / ENCODER_DIR))
