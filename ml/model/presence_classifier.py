"""Binary presence classifier head for the PetBERT training pipeline.

Takes the concatenated PetBERT embeddings of a case report and a taxonomy label
and predicts whether that label is a confirmed diagnosis for that case.

Input:  [report_emb (768) | label_emb (768)] → 1536-dim concatenation
Output: scalar logit (pass through sigmoid to get presence probability)
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class PresenceClassifier(nn.Module):
    """Lightweight binary classifier head on top of frozen PetBERT embeddings."""

    def __init__(self, emb_dim: int = 768, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, report_emb: torch.Tensor, label_emb: torch.Tensor) -> torch.Tensor:
        """Return raw logits (not probabilities) suitable for BCEWithLogitsLoss.

        Args:
            report_emb: (B, 768) report embeddings
            label_emb:  (B, 768) label embeddings
        Returns:
            (B,) raw logits
        """
        x = torch.cat([report_emb, label_emb], dim=-1)  # (B, 1536)
        return self.net(x).squeeze(-1)  # (B,)

    @torch.inference_mode()
    def score_matrix(
        self,
        report_embeddings: torch.Tensor,  # (N, 768)
        label_embeddings: torch.Tensor,   # (M, 768)
        batch_size: int = 512,
    ) -> torch.Tensor:
        """Compute an (N, M) presence probability matrix efficiently.

        For each of the N cases, scores all M labels by running the classifier
        in row-batches to avoid materialising the full N×M×1536 tensor at once.
        """
        self.eval()
        n = report_embeddings.shape[0]
        m = label_embeddings.shape[0]
        device = next(self.parameters()).device
        scores = torch.empty(n, m, dtype=torch.float32, device="cpu")

        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            # Expand: (B, 1, 768) tiled to (B, M, 768)
            r = report_embeddings[start:end].to(device).unsqueeze(1).expand(-1, m, -1)
            l = label_embeddings.to(device).unsqueeze(0).expand(end - start, -1, -1)
            # Flatten to (B*M, 768) for a single forward pass
            b = end - start
            logits = self.forward(r.reshape(b * m, -1), l.reshape(b * m, -1))
            scores[start:end] = torch.sigmoid(logits).reshape(b, m).cpu()

        return scores  # (N, M) float32

    def save(self, path: str | Path) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        emb_dim: int = 768,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ) -> "PresenceClassifier":
        model = cls(emb_dim=emb_dim, hidden_dim=hidden_dim, dropout=dropout)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        return model
