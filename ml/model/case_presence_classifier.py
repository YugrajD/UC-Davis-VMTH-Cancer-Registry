"""Case-level binary cancer presence classifier.

Input:  case report mean embedding (768-dim from PetBERT / contrastive backbone)
Output: scalar sigmoid probability in [0, 1]

Used as the first gate in the sequential pipeline:
  CasePresenceClassifier → GroupClassifier → KW term selection

Unlike PresenceClassifier (which scores (case, label) pairs), this
classifier operates at the case level only — no label embedding required.
A case below the threshold is predicted Uncategorized without reaching
the GroupClassifier.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from model.constants import DEFAULT_DROPOUT, DEFAULT_HIDDEN_DIM, PETBERT_EMB_DIM


class CasePresenceClassifier(nn.Module):
    """Binary case-level cancer presence classifier."""

    def __init__(
        self,
        emb_dim: int = PETBERT_EMB_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        dropout: float = DEFAULT_DROPOUT,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits (B,) for BCEWithLogitsLoss."""
        return self.net(x).squeeze(-1)

    @torch.inference_mode()
    def predict_proba(
        self,
        embeddings: torch.Tensor,
        batch_size: int = 512,
    ) -> torch.Tensor:
        """Return (N,) cancer probabilities in [0, 1]."""
        self.eval()
        device = next(self.parameters()).device
        results = []
        for start in range(0, embeddings.shape[0], batch_size):
            batch = embeddings[start : start + batch_size].to(device)
            results.append(torch.sigmoid(self.forward(batch)).cpu())
        return torch.cat(results, dim=0)

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "state_dict": self.state_dict(),
                "emb_dim": torch.tensor(self.emb_dim),
                "hidden_dim": torch.tensor(self.net[0].out_features),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "CasePresenceClassifier":
        data = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(
            emb_dim=int(data["emb_dim"].item()),
            hidden_dim=int(data["hidden_dim"].item()),
        )
        model.load_state_dict(data["state_dict"])
        return model
