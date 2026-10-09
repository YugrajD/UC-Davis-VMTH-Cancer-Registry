"""The training recipe and seeding.

The values are the ones the first generation (the frozen ``rr-20261008T210937Z`` weights) was trained with.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch

# The production seed; ``train.py --seed`` may override it per run.
PRODUCTION_SEED = 42


def seed_all(seed: int) -> None:
    """Seed python's ``random``, numpy, torch (CPU) and CUDA (if present)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class Recipe:
    epochs: int = 3
    max_length: int = 4096
    batch_size: int = 32          # cases per optimizer step
    token_budget: int = 16384     # padded tokens per micro-batch
    lr: float = 5e-5              # encoder
    head_lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_frac: float = 0.1
    grad_clip: float = 1.0
    gradient_checkpointing: bool = True
    gate_weight: float = 1.0      # total loss = term loss + gate_weight * gate loss
    holdout_fraction: float = 0.05   # stratified on the gate target; scored after every epoch, never trained on
    holdout_seed: int = 1234         # fixed: the held-out set must not move with the training seed
    seed: int = PRODUCTION_SEED
