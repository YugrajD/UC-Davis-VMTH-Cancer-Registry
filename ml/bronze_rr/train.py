"""Train the term retriever or the case-presence gate end to end (BCE; the recipe is the Phase 0 retriever's).

Both write ``train_log.json`` ({"summary": {..., "args": {...}}, "steps": [...], "epochs": [...]}) so
``scoring.trained_max_length`` can read the max length a model was trained with.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from bronze_rr import batching, encoder as enc, encoder_train, gate as gate_mod, report_text, retriever, scoring
from bronze_rr.data import BronzeData
from report_mapping.training.recipe import PRODUCTION_SEED, seed_all


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
    seed: int = PRODUCTION_SEED
    limit_train: int = 0          # smoke: only the first N training cases
    max_steps: int = 0            # smoke: stop after N optimizer steps


def _refuse_existing(out: Path) -> None:
    if (out / "train_log.json").exists():
        raise SystemExit(f"{out} already holds a trained model; choose another --out or delete it")


def _fit_kwargs(r: Recipe) -> dict:
    return dict(epochs=r.epochs, lr=r.lr, head_lr=r.head_lr, weight_decay=r.weight_decay, warmup_frac=r.warmup_frac,
                batch_size=r.batch_size, token_budget=r.token_budget, seed=r.seed, max_steps=r.max_steps)


def _write_log(out: Path, recipe: Recipe, summary: dict, steps: list, epochs: list) -> None:
    summary = {**summary, "args": {k: str(v) for k, v in asdict(recipe).items()}}
    (out / "train_log.json").write_text(json.dumps({"summary": summary, "steps": steps, "epochs": epochs}),
                                        encoding="utf-8")


def train_retriever(data: BronzeData, out: Path, recipe: Recipe, device="cuda") -> dict:
    _refuse_existing(out)
    seed_all(recipe.seed)
    case_ids = data.retriever_ids[:recipe.limit_train] if recipe.limit_train else data.retriever_ids
    term_index = {t: i for i, t in enumerate(data.vocab)}
    targets = np.zeros((len(case_ids), len(data.vocab)), dtype=np.float32)
    for row, case_id in enumerate(case_ids):
        for term in data.gold[case_id]:
            targets[row, term_index[term]] = 1.0

    tokenizer = enc.load_tokenizer()
    by_case = report_text.build_report_texts(data.reports)
    token_ids, over = batching.encode_reports(tokenizer, [by_case[c] for c in case_ids], recipe.max_length)
    print(f"train cases {len(case_ids)}, terms {len(data.vocab)}, reports over {recipe.max_length} tokens: {over}", flush=True)

    model = retriever.new_retriever(data.vocab).to(device)
    model.init_bias_from_frequency(torch.tensor(targets.mean(0), device=device))
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    steps, epochs = encoder_train.fit(model, tokenizer, token_ids, targets, enc.bce_loss, device=device,
                                      **_fit_kwargs(recipe))
    model.save(out, tokenizer)
    summary = {"train_cases": len(case_ids), "terms": len(data.vocab), "reports_over_max_length": over,
               "steps": len(steps), "train_seconds": time.time() - start,
               "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30}
    _write_log(out, recipe, summary, steps, epochs)
    return summary


def train_gate(data: BronzeData, out: Path, recipe: Recipe, device="cuda") -> dict:
    """BCE on every train case that has a silver label row; 5% stratified held out to pick the best epoch."""
    _refuse_existing(out)
    seed_all(recipe.seed)
    n = recipe.limit_train or len(data.gate_ids)
    case_ids, targets = data.gate_ids[:n], data.gate_targets[:n].reshape(-1, 1)
    fit_idx, held_idx = gate_mod.split_holdout(targets[:, 0])

    tokenizer = enc.load_tokenizer()
    by_case = report_text.build_report_texts(data.reports)
    token_ids, over = batching.encode_reports(tokenizer, [by_case[case_ids[i]] for i in fit_idx], recipe.max_length)
    held_texts = [by_case[case_ids[i]] for i in held_idx]
    held_targets = targets[held_idx]
    print(f"gate: train {len(fit_idx)} (positive {targets[fit_idx].mean():.3f}), held-out {len(held_idx)}, "
          f"reports over {recipe.max_length} tokens: {over}", flush=True)

    model = enc.PooledClassifier(enc.new_encoder(), 1).to(device)
    model.init_bias_from_frequency(torch.tensor(targets[fit_idx].mean(0), device=device))
    best = {"auc": -1.0, "epoch": None}

    def after_epoch(epoch: int) -> dict:
        logits, _ = scoring.logits_for(model, tokenizer, held_texts, recipe.max_length, 2 * recipe.token_budget, device)
        loss = float(enc.bce_loss(torch.tensor(logits), torch.tensor(held_targets)))
        auc = gate_mod.roc_auc(logits[:, 0], held_targets[:, 0])
        if best["epoch"] is None or auc > best["auc"]:
            best.update(auc=auc, epoch=epoch)
            model.save(out, tokenizer)
        return {"heldout_loss": loss, "heldout_auc": auc, "saved": best["epoch"] == epoch}

    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    steps, epochs = encoder_train.fit(model, tokenizer, token_ids, targets[fit_idx], enc.bce_loss, device=device,
                                      on_epoch_end=after_epoch, **_fit_kwargs(recipe))
    summary = {"train_cases": len(fit_idx), "heldout_cases": len(held_idx), "reports_over_max_length": over,
               "best_epoch": best["epoch"], "best_heldout_auc": best["auc"], "steps": len(steps),
               "train_seconds": time.time() - start, "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30}
    _write_log(out, recipe, summary, steps, epochs)
    return summary
