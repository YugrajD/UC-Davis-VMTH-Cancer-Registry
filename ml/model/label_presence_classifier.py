"""Label presence classifier head for the PetBERT pipeline.

Takes the concatenated PetBERT embeddings of a case report and a taxonomy label
and predicts whether that label is a confirmed diagnosis for that case.

Two modes (controlled by col_pair_mode):

  col_pair_mode=True  (per-pair architecture):
    For each column, form the pair [colN_emb (768) | label_emb (768)] → 1536-dim.
    The shared MLP scores each (column, label) pair independently, producing n_cols
    scalar logits. These are then combined via col_combine:

      col_combine="max"      — max-pool: most informative column wins (Phase 14)
      col_combine="mean"     — average across columns
      col_combine="learned"  — Linear(n_cols → 1) learns per-column weights globally
                               (e.g., "FINAL COMMENT matters 2× more than ANCILLARY TESTS")

    Input to shared MLP: 2 * emb_dim = 1536

  col_pair_mode=False (legacy — concat architecture):
    Input: [col1_emb (768) | ... | colN_emb (768) | label_emb (768)] → (n_cols+1)*768-dim
    The MLP sees all columns and the label simultaneously.
    Input to MLP: (n_cols + 1) * emb_dim = 3072 for n_cols=3  (Phase 13 best)

n_cols=1 reproduces the original single-column behaviour in either mode.
n_cols, col_pair_mode, and col_combine are saved into the checkpoint.
Legacy checkpoints (no col_pair_mode key) load as col_pair_mode=False.
Phase 14 checkpoints (col_pair_mode=True, no col_combine key) load as col_combine="max".
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from model.constants import DEFAULT_DROPOUT, DEFAULT_HIDDEN_DIM, PETBERT_EMB_DIM


class LabelPresenceClassifier(nn.Module):
    """Lightweight binary classifier head on top of frozen PetBERT embeddings."""

    def __init__(
        self,
        emb_dim: int = PETBERT_EMB_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        dropout: float = DEFAULT_DROPOUT,
        n_cols: int = 1,
        col_pair_mode: bool = True,
        col_combine: str = "learned",  # "max" | "mean" | "learned"
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_cols = n_cols
        self.col_pair_mode = col_pair_mode
        self.col_combine = col_combine
        input_dim = 2 * emb_dim if col_pair_mode else (n_cols * emb_dim + emb_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        # Learned column combiner: 3 scalar logits → 1 weighted sum
        # Only used when col_pair_mode=True and col_combine="learned"
        self.col_combiner = (
            nn.Linear(n_cols, 1, bias=True)
            if (col_pair_mode and col_combine == "learned")
            else None
        )

    def forward(self, report_emb: torch.Tensor, label_emb: torch.Tensor) -> torch.Tensor:
        """Return raw logits (not probabilities) suitable for BCEWithLogitsLoss.

        Args:
            report_emb: (B, n_cols * emb_dim) concatenated per-column embeddings
            label_emb:  (B, emb_dim) label embeddings
        Returns:
            (B,) raw logits
        """
        if self.col_pair_mode:
            B = report_emb.shape[0]
            # (B, n_cols * emb_dim) → (B, n_cols, emb_dim)
            col_embs = report_emb.view(B, self.n_cols, self.emb_dim)
            # (B, emb_dim) → (B, n_cols, emb_dim)
            lbl = label_emb.unsqueeze(1).expand(-1, self.n_cols, -1)
            # (B, n_cols, 2 * emb_dim)
            pairs = torch.cat([col_embs, lbl], dim=-1)
            # (B * n_cols, 2 * emb_dim) → net → (B * n_cols, 1) → (B, n_cols)
            per_col_logits = self.net(pairs.view(B * self.n_cols, 2 * self.emb_dim))
            per_col_logits = per_col_logits.view(B, self.n_cols)
            if self.col_combine == "max":
                return per_col_logits.max(dim=1).values
            elif self.col_combine == "mean":
                return per_col_logits.mean(dim=1)
            else:  # "learned"
                return self.col_combiner(per_col_logits).squeeze(-1)
        else:
            x = torch.cat([report_emb, label_emb], dim=-1)
            return self.net(x).squeeze(-1)

    @torch.inference_mode()
    def score_matrix(
        self,
        report_embeddings: torch.Tensor,  # (N, n_cols * 768)
        label_embeddings: torch.Tensor,   # (M, 768)
        batch_size: int = 512,
    ) -> torch.Tensor:
        """Compute an (N, M) presence probability matrix efficiently."""
        self.eval()
        n = report_embeddings.shape[0]
        m = label_embeddings.shape[0]
        device = next(self.parameters()).device
        scores = torch.empty(n, m, dtype=torch.float32, device="cpu")

        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            b = end - start
            r = report_embeddings[start:end].to(device).unsqueeze(1).expand(-1, m, -1)
            l = label_embeddings.to(device).unsqueeze(0).expand(b, -1, -1)
            logits = self.forward(r.reshape(b * m, -1), l.reshape(b * m, -1))
            scores[start:end] = torch.sigmoid(logits).reshape(b, m).cpu()

        return scores  # (N, M) float32

    def save(self, path: str | Path) -> None:
        torch.save({
            "state_dict": self.state_dict(),
            "n_cols": torch.tensor(self.n_cols),
            "emb_dim": torch.tensor(self.emb_dim),
            "hidden_dim": torch.tensor(self.net[0].out_features),
            "col_pair_mode": torch.tensor(self.col_pair_mode),
            "col_combine": self.col_combine,
        }, path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        dropout: float = DEFAULT_DROPOUT,
    ) -> "LabelPresenceClassifier":
        data = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(data, dict) and "state_dict" in data:
            n_cols = int(data.get("n_cols", torch.tensor(1)).item())
            emb_dim = int(data.get("emb_dim", torch.tensor(PETBERT_EMB_DIM)).item())
            # Checkpoints trained with a non-default hidden_dim store the value directly.
            # Fall back to the caller-supplied hidden_dim for legacy checkpoints.
            hidden_dim = int(data.get("hidden_dim", torch.tensor(hidden_dim)).item())
            # Phase 1–13: no col_pair_mode key → False (concat)
            col_pair_mode = bool(data.get("col_pair_mode", torch.tensor(False)).item())
            # Phase 14: col_pair_mode=True but no col_combine key → "max"
            col_combine = data.get("col_combine", "max") if col_pair_mode else "max"
            model = cls(
                emb_dim=emb_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                n_cols=n_cols,
                col_pair_mode=col_pair_mode,
                col_combine=col_combine,
            )
            model.load_state_dict(data["state_dict"])
        else:
            # Legacy format — plain state dict (n_cols=1, concat)
            model = cls(
                emb_dim=PETBERT_EMB_DIM,
                hidden_dim=hidden_dim,
                dropout=dropout,
                n_cols=1,
                col_pair_mode=False,
                col_combine="max",
            )
            model.load_state_dict(data)
        return model
