"""Binary presence classifier head for the PetBERT training pipeline.

Takes the concatenated PetBERT embeddings of a case report and a taxonomy label
and predicts whether that label is a confirmed diagnosis for that case.

Input:  [col1_emb (768) | ... | colN_emb (768) | label_emb (768)] → (n_cols+1)*768-dim
Output: scalar logit (pass through sigmoid to get presence probability)

n_cols=1 (default) reproduces the original mean-embedding behaviour.
n_cols=3 feeds each report column independently so the classifier can learn
column-specific diagnostic signal instead of receiving a diluted average.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from model.constants import DEFAULT_DROPOUT, DEFAULT_HIDDEN_DIM, PETBERT_EMB_DIM


class PresenceClassifier(nn.Module):
    """Lightweight binary classifier head on top of frozen PetBERT embeddings."""

    def __init__(
        self,
        emb_dim: int = PETBERT_EMB_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        dropout: float = DEFAULT_DROPOUT,
        n_cols: int = 1,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_cols = n_cols
        self.net = nn.Sequential(
            nn.Linear(n_cols * emb_dim + emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, report_emb: torch.Tensor, label_emb: torch.Tensor) -> torch.Tensor:
        """Return raw logits (not probabilities) suitable for BCEWithLogitsLoss.

        Args:
            report_emb: (B, n_cols * emb_dim) concatenated per-column embeddings
            label_emb:  (B, emb_dim) label embeddings
        Returns:
            (B,) raw logits
        """
        x = torch.cat([report_emb, label_emb], dim=-1)
        return self.net(x).squeeze(-1)

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
        torch.save({
            "state_dict": self.state_dict(),
            "n_cols": torch.tensor(self.n_cols),
            "emb_dim": torch.tensor(self.emb_dim),
        }, path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        dropout: float = DEFAULT_DROPOUT,
    ) -> "PresenceClassifier":
        data = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(data, dict) and "state_dict" in data:
            n_cols = int(data.get("n_cols", torch.tensor(1)).item())
            emb_dim = int(data.get("emb_dim", torch.tensor(PETBERT_EMB_DIM)).item())
            model = cls(emb_dim=emb_dim, hidden_dim=hidden_dim, dropout=dropout, n_cols=n_cols)
            model.load_state_dict(data["state_dict"])
        else:
            # Legacy format — plain state dict saved before n_cols was introduced (n_cols=1)
            model = cls(emb_dim=PETBERT_EMB_DIM, hidden_dim=hidden_dim, dropout=dropout, n_cols=1)
            model.load_state_dict(data)
        return model
