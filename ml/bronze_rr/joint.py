"""One ModernBERT trained jointly for the term retriever and the case-presence gate.

Encoder -> mean pool (the same read-out as ``encoder.PooledClassifier``) -> two fp32 linear heads: a term head
(one logit per diagnosis term) and a gate head (one logit: is there a cancer diagnosis). One encoder pass per
report gives both outputs.

Layout on disk: ``encoder/``, ``head.pt`` (the term head, so the directory also loads unchanged as a retriever
with ``retriever.load_retriever``), ``gate_head.pt``, ``terms.json``, ``train_log.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from bronze_rr import encoder as enc
from bronze_rr import retriever

DIR_NAME = "joint"   # <root>/joint: where train-joint writes and the bake-off looks
GATE_HEAD_FILE = "gate_head.pt"


class JointClassifier(nn.Module):
    def __init__(self, encoder, terms: list[str]):
        super().__init__()
        self.encoder = encoder
        self.terms = terms
        hidden = encoder.config.hidden_size
        self.head = nn.Linear(hidden, len(terms))   # named "head" so head.pt is a retriever's head.pt
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
        self.encoder.save_pretrained(directory / enc.ENCODER_DIR)
        tokenizer.save_pretrained(directory / enc.ENCODER_DIR)
        torch.save(self.head.state_dict(), directory / enc.HEAD_FILE)
        torch.save(self.gate_head.state_dict(), directory / GATE_HEAD_FILE)
        (directory / retriever.TERMS_FILE).write_text(json.dumps(self.terms), encoding="utf-8")


def new_joint(terms: list[str]) -> JointClassifier:
    return JointClassifier(enc.new_encoder(), terms)


def load_joint(directory: Path) -> tuple[JointClassifier, object]:
    model = JointClassifier(enc.new_encoder(str(directory / enc.ENCODER_DIR)), retriever.read_terms(directory))
    model.head.load_state_dict(torch.load(directory / enc.HEAD_FILE, map_location="cpu"))
    model.gate_head.load_state_dict(torch.load(directory / GATE_HEAD_FILE, map_location="cpu"))
    return model, enc.load_tokenizer(str(directory / enc.ENCODER_DIR))


def step_loss(term_logits: torch.Tensor, gate_logits: torch.Tensor, term_targets: torch.Tensor,
              gate_targets: torch.Tensor, is_cancer: torch.Tensor, n_cancer_step: int, n_step: int
              ) -> tuple[torch.Tensor, torch.Tensor]:
    """(term loss, gate loss) of one micro-batch, each already divided by its optimizer STEP's count, so summing the
    parts over a step's micro-batches equals the loss of the whole step as one batch.

    term: BCE summed over terms, summed over the micro-batch's CANCER cases (``is_cancer`` is 0/1), / cancer cases in
    the step; non-cancer cases contribute nothing (their all-zero term targets would otherwise teach "no term").
    gate: BCE over every case, / cases in the step.
    """
    per_case = F.binary_cross_entropy_with_logits(term_logits, term_targets, reduction="none").sum(1)
    term = (per_case * is_cancer).sum() / max(n_cancer_step, 1)
    gate = F.binary_cross_entropy_with_logits(gate_logits, gate_targets, reduction="sum") / n_step
    return term, gate
