"""A case-presence gate that rides on the retriever's encoder: a small head on the frozen pooled report vector.

The retriever already mean-pools the encoder output before its term head. One encoder pass therefore gives both
the term logits and the pooled vector, and this module's head turns the pooled vector into the gate probability,
so a deployed system runs one network instead of two.

Cache files (numbers and case ids only, never text; derived from patient text, so they live under the gitignored
ml/output), per (retriever, partition) under ``config.RR_PHASE1_DIR/scores/`` (or ``out_dir``):
  pooled_<retriever tag>_<partition>.npy       (N, hidden) float32 pooled vectors
  pooled_<retriever tag>_<partition>_ids.csv   case_id, in the npy's row order
A cache miss runs the retriever over the partition ONCE; the same pass also fills the retriever score cache
(``scoring.cache_path("retriever", ...)``) when that is absent.

A trained head lives in ``<root>/gate_shared_<retriever tag>/``: head.pt, train_log.json and source.json (the
retriever it was trained on, with sha256 of the retriever's head and encoder weights). A head is refused with any
other retriever, because its input space is that retriever's pooled vectors.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import config
import io_utils
from bronze_rr import encoder as enc, gate as gate_mod, report_text, retriever, scoring
from bronze_rr.data import BronzeData
from generations.manifest import sha256_file
from report_mapping.training.recipe import PRODUCTION_SEED, seed_all

GATE_PREFIX = "gate_shared_"
HEAD_FILE, SOURCE_FILE, LOG_FILE = enc.HEAD_FILE, "source.json", "train_log.json"
ENCODER_WEIGHTS = Path(enc.ENCODER_DIR) / "model.safetensors"
KINDS = ("linear", "mlp")
MLP_WIDTH, MLP_DROPOUT = 256, 0.3


def gate_dir_for(root: Path, retriever_dir: Path) -> Path:
    return root / f"{GATE_PREFIX}{scoring.model_tag(retriever_dir)}"


def available(root: Path) -> list[Path]:
    """Every trained shared-gate directory under ``root``."""
    return [d for d in sorted(root.glob(f"{GATE_PREFIX}*")) if all((d / f).is_file() for f in (HEAD_FILE, SOURCE_FILE, LOG_FILE))]


# --- the retriever a head belongs to -------------------------------------------------------------------------------

def _retriever_fingerprint(retriever_dir: Path) -> dict:
    return {"retriever_tag": scoring.model_tag(retriever_dir),
            "retriever_head_sha256": sha256_file(retriever_dir / enc.HEAD_FILE),
            "retriever_encoder_sha256": sha256_file(retriever_dir / ENCODER_WEIGHTS)}


def write_source(gate_dir: Path, retriever_dir: Path) -> None:
    source = {"retriever_dir": str(retriever_dir), **_retriever_fingerprint(retriever_dir)}
    (gate_dir / SOURCE_FILE).write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")


def read_source(gate_dir: Path) -> dict:
    return json.loads((gate_dir / SOURCE_FILE).read_text(encoding="utf-8"))


def verify_source(gate_dir: Path, retriever_dir: Path) -> None:
    """Refuse ``retriever_dir`` unless its head and encoder weights are the ones the head in ``gate_dir`` was trained on."""
    recorded, actual = read_source(gate_dir), _retriever_fingerprint(retriever_dir)
    wrong = [k for k, v in actual.items() if recorded.get(k) != v]
    if wrong:
        raise ValueError(f"shared gate {gate_dir.name} was not trained on retriever {retriever_dir}: {', '.join(wrong)} differ")


# --- pooled vectors -------------------------------------------------------------------------------------------------

def _pooled_paths(retriever_dir: Path, partition: str, out_dir: Path) -> tuple[Path, Path]:
    stem = f"pooled_{scoring.model_tag(retriever_dir)}_{partition}"
    return out_dir / f"{stem}.npy", out_dir / f"{stem}_ids.csv"


def _embed_partition(retriever_dir: Path, data: BronzeData, partition: str, *, out_dir: Path, limit: int,
                     token_budget: int, device) -> tuple[list[str], np.ndarray]:
    """Cache hit -> the cached (case_ids, pooled); miss -> one retriever pass over the partition, which writes the
    pooled cache and, if absent, the retriever score cache too."""
    npy, ids_csv = _pooled_paths(retriever_dir, partition, out_dir)
    if npy.is_file() and ids_csv.is_file():
        case_ids = io_utils.read_csv(ids_csv, encoding="utf-8", dtype={"case_id": str})["case_id"].tolist()
        pooled = np.load(npy)
        if len(case_ids) != len(pooled):
            raise ValueError(f"{npy} and {ids_csv} disagree on the number of cases")
        return case_ids, pooled
    case_ids = data.ids_for(partition)[:limit] if limit else data.ids_for(partition)
    model, tokenizer = retriever.load_retriever(retriever_dir)
    model.to(device)
    by_case = report_text.build_report_texts(data.reports)
    start = time.time()
    pooled, logits, over = scoring.pooled_and_logits_for(
        model, tokenizer, [by_case[c] for c in case_ids], scoring.trained_max_length(retriever_dir), token_budget, device)
    print(f"pooled {scoring.model_tag(retriever_dir)} {partition}: {len(case_ids)} cases, {over} over max length, "
          f"{time.time() - start:.0f}s", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_csv = scoring.cache_path("retriever", retriever_dir, partition, out_dir)
    if not scores_csv.is_file():
        io_utils.write_csv(scoring.retriever_frame(case_ids, logits), scores_csv)
    np.save(npy, pooled)
    io_utils.write_csv(pd.DataFrame({"case_id": case_ids}), ids_csv)
    del model
    torch.cuda.empty_cache()
    return case_ids, pooled


def pooled_vectors(retriever_dir: Path, data: BronzeData, partition: str, *, out_dir: Path = scoring.SCORES_DIR,
                   limit: int = 0, token_budget: int = 32768, device="cuda") -> tuple[list[str], np.ndarray]:
    """(case_ids, pooled) for a split partition or "all", in the order ``scoring.retriever_scores`` uses."""
    parts = [_embed_partition(retriever_dir, data, p, out_dir=out_dir, limit=limit, token_budget=token_budget,
                              device=device) for p in scoring.partitions(partition)]
    return [c for ids, _ in parts for c in ids], np.concatenate([x for _, x in parts])


# --- the head -------------------------------------------------------------------------------------------------------

class SharedHead(nn.Module):
    """Standardise the pooled vector (fit-set mean/std, not trained), then a linear or one-hidden-layer head."""

    def __init__(self, kind: str, hidden: int, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.kind, self.hidden = kind, hidden
        self.register_buffer("mean", mean.clone())
        self.register_buffer("std", std.clone())
        self.net = nn.Linear(hidden, 1) if kind == "linear" else nn.Sequential(
            nn.Linear(hidden, MLP_WIDTH), nn.ReLU(), nn.Dropout(MLP_DROPOUT), nn.Linear(MLP_WIDTH, 1))

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.net((pooled - self.mean) / self.std)


def _new_head(kind: str, x_fit: torch.Tensor) -> SharedHead:
    return SharedHead(kind, x_fit.shape[1], x_fit.mean(0), x_fit.std(0).clamp(min=1e-6))


def save_head(head: SharedHead, path: Path) -> None:
    torch.save({"kind": head.kind, "hidden": head.hidden, "state_dict": head.state_dict()}, path)


def load_head(gate_dir: Path) -> SharedHead:
    blob = torch.load(gate_dir / HEAD_FILE, map_location="cpu", weights_only=True)
    zeros = torch.zeros(blob["hidden"])
    head = SharedHead(blob["kind"], blob["hidden"], zeros, zeros + 1)
    head.load_state_dict(blob["state_dict"])
    return head.eval()


def head_probs(head: SharedHead, pooled: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return torch.sigmoid(head.eval()(torch.from_numpy(pooled))[:, 0]).double().numpy()


def _fit_variant(kind: str, x_fit, y_fit, x_held, y_held, *, epochs: int, batch_size: int, lr: float,
                 weight_decay: float, seed: int) -> tuple[SharedHead, dict]:
    """Train one head variant; return the epoch with the best held-out AUC (first on ties) and the per-epoch log."""
    torch.manual_seed(seed)
    head = _new_head(kind, x_fit)
    head.to(x_fit.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    order = torch.Generator().manual_seed(seed)
    best_state, best, log = None, {"epoch": None, "auc": float("-inf"), "loss": None}, []
    for epoch in range(epochs):
        head.train()
        train_losses = []
        for batch in torch.randperm(len(x_fit), generator=order).split(batch_size):
            batch = batch.to(x_fit.device)
            loss = F.binary_cross_entropy_with_logits(head(x_fit[batch]), y_fit[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        head.eval()
        with torch.no_grad():
            held_logits = head(x_held)
            held_loss = F.binary_cross_entropy_with_logits(held_logits, y_held).item()
        auc = gate_mod.roc_auc(held_logits[:, 0].cpu().numpy(), y_held[:, 0].cpu().numpy())
        log.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "heldout_loss": held_loss, "heldout_auc": auc})
        if best["epoch"] is None or auc > best["auc"]:
            best, best_state = {"epoch": epoch, "auc": auc, "loss": held_loss}, copy.deepcopy(head.state_dict())
    head.load_state_dict(best_state)
    return head.eval(), {"best_epoch": best["epoch"], "best_heldout_auc": best["auc"], "heldout_loss_at_best": best["loss"],
                         "epochs": log}


def train_shared_gate(data: BronzeData, retriever_dir: Path, out: Path, *, scores_dir: Path = scoring.SCORES_DIR,
                      limit: int = 0, seed: int = PRODUCTION_SEED, epochs: int = 50, batch_size: int = 512,
                      lr: float = 1e-3, weight_decay: float = 0.01, token_budget: int = 32768, device="cuda") -> dict:
    """Train both head variants on the cached pooled TRAIN vectors of the gate's training universe (same cases, same
    targets and same stratified held-out 5% as the ModernBERT gate), keep the variant whose best epoch has the
    highest held-out AUC, and write head.pt, source.json and train_log.json into ``out``."""
    if (out / HEAD_FILE).exists():
        raise SystemExit(f"{out} already holds a head; choose another output directory or delete it")
    seed_all(seed)
    case_ids, pooled = pooled_vectors(retriever_dir, data, "train", out_dir=scores_dir, limit=limit,
                                      token_budget=token_budget, device=device)
    row = {c: i for i, c in enumerate(case_ids)}
    keep = [i for i, c in enumerate(data.gate_ids) if c in row]   # all of gate_ids, except in a --limit smoke
    targets = data.gate_targets[keep].astype(np.float32).reshape(-1, 1)
    features = pooled[[row[data.gate_ids[i]] for i in keep]]
    fit_idx, held_idx = gate_mod.split_holdout(targets[:, 0])

    place = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, y = torch.from_numpy(features).to(place), torch.from_numpy(targets).to(place)
    x_fit, y_fit, x_held, y_held = x[fit_idx], y[fit_idx], x[held_idx], y[held_idx]
    print(f"shared gate: train {len(fit_idx)} (positive {targets[fit_idx].mean():.3f}), held-out {len(held_idx)}, "
          f"hidden {x.shape[1]}, head device {place.type}", flush=True)

    variants = {}
    for kind in KINDS:
        head, info = _fit_variant(kind, x_fit, y_fit, x_held, y_held, epochs=epochs, batch_size=batch_size, lr=lr,
                                  weight_decay=weight_decay, seed=seed)
        variants[kind] = (head, info)
        print(f"  {kind}: best epoch {info['best_epoch']}, held-out AUC {info['best_heldout_auc']:.4f}, "
              f"loss {info['heldout_loss_at_best']:.4f}", flush=True)
    chosen = max(KINDS, key=lambda k: variants[k][1]["best_heldout_auc"])   # first in KINDS on ties

    out.mkdir(parents=True, exist_ok=True)
    save_head(variants[chosen][0].cpu(), out / HEAD_FILE)
    write_source(out, retriever_dir)
    summary = {"chosen": chosen, "train_cases": len(fit_idx), "heldout_cases": len(held_idx), "hidden": int(x.shape[1]),
               **{f"{k}_best_heldout_auc": variants[k][1]["best_heldout_auc"] for k in KINDS},
               **{f"{k}_heldout_loss_at_best": variants[k][1]["heldout_loss_at_best"] for k in KINDS},
               "args": {"seed": seed, "epochs": epochs, "batch_size": batch_size, "lr": lr, "weight_decay": weight_decay,
                        "limit": limit, "retriever_dir": str(retriever_dir)}}
    (out / LOG_FILE).write_text(json.dumps({"summary": summary, "variants": {k: v[1] for k, v in variants.items()}}),
                                encoding="utf-8")
    return summary


def gate_scores(gate_dir: Path, retriever_dir: Path, data: BronzeData, partition: str, *,
                out_dir: Path = scoring.SCORES_DIR, limit: int = 0, token_budget: int = 32768,
                device="cuda") -> dict[str, float]:
    """{case_id: gate probability} from the shared head in ``gate_dir`` on ``retriever_dir``'s pooled vectors."""
    verify_source(gate_dir, retriever_dir)
    case_ids, pooled = pooled_vectors(retriever_dir, data, partition, out_dir=out_dir, limit=limit,
                                      token_budget=token_budget, device=device)
    return dict(zip(case_ids, head_probs(load_head(gate_dir), pooled).tolist()))
