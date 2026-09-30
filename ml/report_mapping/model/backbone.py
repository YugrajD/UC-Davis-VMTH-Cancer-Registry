"""Load an HF checkpoint dir/name and mean-pool text into PetBERT embeddings.

Carried from ``production/petbert_pipeline/embedding.py``: same loader
(``AutoModelForMaskedLM``, base transformer only — the MLM head is never
used), same mean-pool-over-attended-tokens algorithm, so embeddings are
bit-identical to production's for the same weights and text. Per CLAUDE.md,
``model_dir_or_name`` accepts any HuggingFace checkpoint dir or HF name;
production's default is the current generation's ``petbert/``.

``max_length`` is 512 at inference (this module's default) vs 256 at training
time (report_mapping/training/backbone.py, WP5) — both are load-bearing and
kept distinct; see ml-rewrite-plan.md Findings.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import transformers.utils.logging as hf_logging

from generations.manifest import sha256_file

INFERENCE_MAX_LENGTH = 512
DEFAULT_BATCH_SIZE = 16


@dataclass
class Backbone:
    tokenizer: AutoTokenizer
    model: AutoModelForMaskedLM
    source: str  # the dir/name passed to load_backbone, for logging only


def device_from_arg(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_backbone(model_dir_or_name: str, *, local_only: bool) -> Backbone:
    tokenizer = AutoTokenizer.from_pretrained(model_dir_or_name, local_files_only=local_only)
    hf_logging.set_verbosity_error()
    model = AutoModelForMaskedLM.from_pretrained(model_dir_or_name, local_files_only=local_only)
    hf_logging.set_verbosity_warning()
    return Backbone(tokenizer=tokenizer, model=model, source=str(model_dir_or_name))


@torch.inference_mode()
def embed_texts(
    backbone: Backbone,
    texts: list[str],
    *,
    device: torch.device,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_length: int = INFERENCE_MAX_LENGTH,
) -> np.ndarray:
    """Mean-pool over attended (non-padding) tokens. Returns (len(texts), 768) float32."""
    model, tokenizer = backbone.model, backbone.tokenizer
    model.eval()
    model.to(device)

    all_embeddings: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        outputs = model.base_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        mean_embedding = summed / counts
        all_embeddings.append(mean_embedding.detach().cpu().numpy().astype(np.float32, copy=False))
    if not all_embeddings:
        return np.zeros((0, model.config.hidden_size), dtype=np.float32)
    return np.vstack(all_embeddings)


def _dir_fingerprint(directory: Path) -> str:
    """A single sha256 over every file's (relative path, sha256), sorted — a
    stable digest of an entire directory's contents (e.g. an HF checkpoint dir)."""
    digest = hashlib.sha256()
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        digest.update(f"{path.relative_to(directory).as_posix()}:{sha256_file(path)}\n".encode("utf-8"))
    return digest.hexdigest()


def model_fingerprint(model_dir_or_name: str) -> str:
    """sha256 of an on-disk checkpoint dir's contents, or of the name string itself
    for a bare HuggingFace hub name (nothing local to hash)."""
    path = Path(model_dir_or_name)
    if path.is_dir():
        return _dir_fingerprint(path)
    return hashlib.sha256(str(model_dir_or_name).encode("utf-8")).hexdigest()
