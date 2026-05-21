"""Multi-class group classifier for the PetBERT pipeline.

Replaces the LabelPresenceClassifier pair-wise approach with a single global
group decision per report. Instead of scoring each (report, label) pair
independently, this classifier takes a report mean embedding and outputs
independent sigmoid probabilities for each of the cancer groups.

In inference, predicted groups (above a threshold) trigger a term-selection step:
cosine similarity against only terms within the predicted group. This two-stage
approach eliminates the ~42% completely-off floor caused by implicit group
competition in the binary approach.

See ml/documentation/multiclass-classifier-plan.md for the full design.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from model.constants import DEFAULT_DROPOUT, DEFAULT_HIDDEN_DIM, PETBERT_EMB_DIM


class GroupClassifier(nn.Module):
    """Multi-label group classifier on top of frozen PetBERT embeddings.

    Input:  mean report embedding (emb_dim = 768)
    Output: per-group sigmoid probabilities (num_groups,)

    Each output is independent (sigmoid, not softmax) because a report can
    belong to multiple groups simultaneously (e.g. mast cell tumor + SCC).
    """

    def __init__(
        self,
        num_groups: int,
        emb_dim: int = PETBERT_EMB_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        dropout: float = DEFAULT_DROPOUT,
    ):
        super().__init__()
        self.num_groups = num_groups
        self.emb_dim = emb_dim
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_groups),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-group sigmoid probabilities.

        Args:
            x: (B, emb_dim) report mean embeddings
        Returns:
            (B, num_groups) probabilities in [0, 1]
        """
        return torch.sigmoid(self.net(x))

    @torch.inference_mode()
    def predict_proba(
        self,
        embeddings: torch.Tensor,
        batch_size: int = 512,
    ) -> torch.Tensor:
        """Compute (N, num_groups) probability matrix for N cases."""
        self.eval()
        device = next(self.parameters()).device
        results = []
        for start in range(0, embeddings.shape[0], batch_size):
            batch = embeddings[start : start + batch_size].to(device)
            results.append(self.forward(batch).cpu())
        return torch.cat(results, dim=0)

    def save(self, path: str | Path, group_names: list[str]) -> None:
        """Save model weights and group name mapping to a checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "group_names": group_names,
                "num_groups": self.num_groups,
                "emb_dim": self.emb_dim,
                "hidden_dim": self.net[0].out_features,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> tuple["GroupClassifier", list[str]]:
        """Load model and return (model, group_names)."""
        data = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(
            num_groups=data["num_groups"],
            emb_dim=data["emb_dim"],
            hidden_dim=data.get("hidden_dim", DEFAULT_HIDDEN_DIM),
        )
        model.load_state_dict(data["state_dict"])
        return model, data["group_names"]
