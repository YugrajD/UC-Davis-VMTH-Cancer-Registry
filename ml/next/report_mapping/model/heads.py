"""The three trainable heads. State_dict-compatible with the legacy ``.pt`` files.

Ported from ``ml/model/{case_presence_classifier,group_classifier,
label_presence_classifier}.py``: same ``nn.Sequential`` module names
(``net.0``, ``net.3``, ``col_combiner``) and the same forward math, so
``load_state_dict`` on a legacy checkpoint's ``state_dict`` works unchanged.
The demographics hooks in the legacy stage callers are dropped (Decisions:
``features/`` is dropped) — these classes never had a demographics parameter
themselves, so nothing here changes for that.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

PETBERT_EMB_DIM = 768
DEFAULT_HIDDEN_DIM = 512
DEFAULT_DROPOUT = 0.3


class CasePresenceClassifier(nn.Module):
    """Stage 1 gate: case-level binary cancer-presence classifier."""

    def __init__(self, emb_dim: int = PETBERT_EMB_DIM, hidden_dim: int = DEFAULT_HIDDEN_DIM,
                 dropout: float = DEFAULT_DROPOUT):
        super().__init__()
        self.emb_dim = emb_dim
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Raw logits (B,), for BCEWithLogitsLoss."""
        return self.net(x).squeeze(-1)

    @torch.inference_mode()
    def predict_proba(self, embeddings: torch.Tensor, batch_size: int = 512) -> torch.Tensor:
        self.eval()
        device = next(self.parameters()).device
        results = []
        for start in range(0, embeddings.shape[0], batch_size):
            batch = embeddings[start : start + batch_size].to(device)
            results.append(torch.sigmoid(self.forward(batch)).cpu())
        return torch.cat(results, dim=0)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "emb_dim": torch.tensor(self.emb_dim),
            "hidden_dim": torch.tensor(self.net[0].out_features),
        }, path)

    @classmethod
    def load(cls, path: str | Path) -> "CasePresenceClassifier":
        data = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(emb_dim=int(data["emb_dim"].item()), hidden_dim=int(data["hidden_dim"].item()))
        model.load_state_dict(data["state_dict"])
        return model


class GroupClassifier(nn.Module):
    """Stage 2: multi-label group classifier (independent sigmoid per group)."""

    def __init__(self, num_groups: int, emb_dim: int = PETBERT_EMB_DIM,
                 hidden_dim: int = DEFAULT_HIDDEN_DIM, dropout: float = DEFAULT_DROPOUT):
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
        """Per-group sigmoid probabilities (B, num_groups)."""
        return torch.sigmoid(self.net(x))

    @torch.inference_mode()
    def predict_proba(self, embeddings: torch.Tensor, batch_size: int = 512) -> torch.Tensor:
        self.eval()
        device = next(self.parameters()).device
        results = []
        for start in range(0, embeddings.shape[0], batch_size):
            batch = embeddings[start : start + batch_size].to(device)
            results.append(self.forward(batch).cpu())
        return torch.cat(results, dim=0)

    def save(self, path: str | Path, group_names: list[str]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "group_names": group_names,
            "num_groups": self.num_groups,
            "emb_dim": self.emb_dim,
            "hidden_dim": self.net[0].out_features,
        }, path)

    @classmethod
    def load(cls, path: str | Path) -> tuple["GroupClassifier", list[str]]:
        data = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(num_groups=data["num_groups"], emb_dim=data["emb_dim"],
                    hidden_dim=data.get("hidden_dim", DEFAULT_HIDDEN_DIM))
        model.load_state_dict(data["state_dict"])
        return model, list(data["group_names"])


class LabelPresenceClassifier(nn.Module):
    """Stage 3a: per-group (case, label) presence classifier.

    ``n_cols=3, col_pair_mode=True, col_combine="learned"`` in production: each
    of the 3 section views pairs with the label embedding, the shared MLP
    scores each pair, and a learned Linear(3 -> 1) combines the per-section
    logits. See load() for the legacy-checkpoint compatibility fallbacks.
    """

    def __init__(self, emb_dim: int = PETBERT_EMB_DIM, hidden_dim: int = DEFAULT_HIDDEN_DIM,
                 dropout: float = DEFAULT_DROPOUT, n_cols: int = 1,
                 col_pair_mode: bool = True, col_combine: str = "learned"):
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
        self.col_combiner = (
            nn.Linear(n_cols, 1, bias=True) if (col_pair_mode and col_combine == "learned") else None
        )

    def forward(self, report_emb: torch.Tensor, label_emb: torch.Tensor) -> torch.Tensor:
        """Raw logits (B,). ``report_emb`` is (B, n_cols * emb_dim), ``label_emb`` (B, emb_dim)."""
        if self.col_pair_mode:
            B = report_emb.shape[0]
            col_embs = report_emb.view(B, self.n_cols, self.emb_dim)
            lbl = label_emb.unsqueeze(1).expand(-1, self.n_cols, -1)
            pairs = torch.cat([col_embs, lbl], dim=-1)
            per_col_logits = self.net(pairs.view(B * self.n_cols, 2 * self.emb_dim)).view(B, self.n_cols)
            if self.col_combine == "max":
                return per_col_logits.max(dim=1).values
            elif self.col_combine == "mean":
                return per_col_logits.mean(dim=1)
            else:  # "learned"
                return self.col_combiner(per_col_logits).squeeze(-1)
        x = torch.cat([report_emb, label_emb], dim=-1)
        return self.net(x).squeeze(-1)

    @torch.inference_mode()
    def score_matrix(self, report_embeddings: torch.Tensor, label_embeddings: torch.Tensor,
                      batch_size: int = 512) -> torch.Tensor:
        """(N, M) presence-probability matrix."""
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
        return scores

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "n_cols": torch.tensor(self.n_cols),
            "emb_dim": torch.tensor(self.emb_dim),
            "hidden_dim": torch.tensor(self.net[0].out_features),
            "col_pair_mode": torch.tensor(self.col_pair_mode),
            "col_combine": self.col_combine,
        }, path)

    @classmethod
    def load(cls, path: str | Path, *, hidden_dim: int = DEFAULT_HIDDEN_DIM,
              dropout: float = DEFAULT_DROPOUT) -> "LabelPresenceClassifier":
        data = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(data, dict) and "state_dict" in data:
            n_cols = int(data.get("n_cols", torch.tensor(1)).item())
            emb_dim = int(data.get("emb_dim", torch.tensor(PETBERT_EMB_DIM)).item())
            hidden_dim = int(data.get("hidden_dim", torch.tensor(hidden_dim)).item())
            col_pair_mode = bool(data.get("col_pair_mode", torch.tensor(False)).item())
            col_combine = data.get("col_combine", "max") if col_pair_mode else "max"
            model = cls(emb_dim=emb_dim, hidden_dim=hidden_dim, dropout=dropout, n_cols=n_cols,
                        col_pair_mode=col_pair_mode, col_combine=col_combine)
            model.load_state_dict(data["state_dict"])
        else:
            # Legacy plain state dict (pre-Phase-13): n_cols=1, concat mode.
            model = cls(emb_dim=PETBERT_EMB_DIM, hidden_dim=hidden_dim, dropout=dropout,
                        n_cols=1, col_pair_mode=False, col_combine="max")
            model.load_state_dict(data)
        return model
