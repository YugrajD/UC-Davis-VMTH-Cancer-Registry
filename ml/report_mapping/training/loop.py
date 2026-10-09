"""The joint training loop: AdamW, linear warmup/decay, gradient clipping, bf16 autocast on CUDA, gradient
checkpointing, length-bucketed steps cut into token-budgeted micro-batches (gradient accumulation)."""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from report_mapping import batching
from report_mapping.model import JointClassifier
from report_mapping.training.recipe import Recipe


def step_loss(term_logits: torch.Tensor, gate_logits: torch.Tensor, term_targets: torch.Tensor,
              gate_targets: torch.Tensor, is_cancer: torch.Tensor, n_cancer_step: int, n_step: int
              ) -> tuple[torch.Tensor, torch.Tensor]:
    """(term loss, gate loss) of one micro-batch, each already divided by its optimizer STEP's count, so summing the
    parts over a step's micro-batches equals the loss of the whole step as one batch.

    term: BCE summed over terms, summed over the micro-batch's CANCER cases (``is_cancer`` is 0/1), / cancer cases in
    the step; non-cancer cases contribute nothing (their all-zero term targets would otherwise teach "no term").
    gate: BCE over every case, / cases in the step.
    """
    per_case = F.binary_cross_entropy_with_logits(term_logits, term_targets, reduction="none").sum(1)
    term = (per_case * is_cancer).sum() / max(n_cancer_step, 1)
    gate = F.binary_cross_entropy_with_logits(gate_logits, gate_targets, reduction="sum") / n_step
    return term, gate


def fit(model: JointClassifier, tokenizer, token_ids: list[list[int]], term_targets: np.ndarray,
        gate_targets: np.ndarray, recipe: Recipe, device, max_steps: int = 0,
        on_epoch_end: Callable[[int], dict] | None = None) -> tuple[list, list]:
    """Train ``model`` in place (on ``device``); the final epoch is kept. ``max_steps`` > 0 stops after that many
    optimizer steps (smoke). ``on_epoch_end(epoch) -> dict`` is logged with the epoch. Returns (per-step log,
    per-epoch log)."""
    lengths = [len(x) for x in token_ids]
    term_t, gate_t = torch.tensor(term_targets), torch.tensor(gate_targets)
    cancer_t = gate_t  # 0/1: a case has term targets iff it is a gate positive
    kind = torch.device(device).type
    if recipe.gradient_checkpointing:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": recipe.lr},
        {"params": list(model.head.parameters()) + list(model.gate_head.parameters()), "lr": recipe.head_lr},
    ], weight_decay=recipe.weight_decay)
    steps_per_epoch = -(-len(token_ids) // recipe.batch_size)
    total_steps = max_steps or steps_per_epoch * recipe.epochs
    warmup = max(1, int(recipe.warmup_frac * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min((s + 1) / warmup, max(0.0, (total_steps - s) / max(1, total_steps - warmup))))

    steps_log, epochs_log, step = [], [], 0
    for epoch in range(recipe.epochs):
        model.train()
        for step_cases in batching.bucketed_steps(lengths, recipe.batch_size, recipe.seed + epoch):
            step_start, term_total, gate_total = time.time(), 0.0, 0.0
            n_cancer = int(cancer_t[step_cases].sum().item())
            for micro in batching.micro_batches(step_cases, lengths, recipe.token_budget):
                ids, mask = batching.collate(token_ids, micro, tokenizer.pad_token_id, device)
                with torch.autocast(kind, dtype=torch.bfloat16, enabled=kind == "cuda"):
                    term_logits, gate_logits = model(ids, mask)
                term, gate = step_loss(term_logits.float(), gate_logits.float(), term_t[micro].to(device),
                                       gate_t[micro].to(device), cancer_t[micro].to(device), n_cancer,
                                       len(step_cases))
                (term + recipe.gate_weight * gate).backward()
                term_total += term.item()
                gate_total += gate.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), recipe.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            steps_log.append({"step": step, "epoch": epoch, "term_loss": term_total, "gate_loss": gate_total,
                              "loss": term_total + recipe.gate_weight * gate_total, "seconds": time.time() - step_start,
                              "max_len": max(lengths[i] for i in step_cases)})
            if step % 20 == 0 or step == 1:
                peak = f" peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB" if kind == "cuda" else ""
                print(f"epoch {epoch} step {step}/{total_steps} term {term_total:.3f} gate {gate_total:.4f} "
                      f"{steps_log[-1]['seconds']:.1f}s{peak}", flush=True)
            if step >= total_steps:
                break
        this_epoch = [s for s in steps_log if s["epoch"] == epoch]
        entry = {"epoch": epoch, "steps": len(this_epoch),
                 **{f"train_{k}": float(np.mean([s[k] for s in this_epoch])) for k in ("loss", "term_loss", "gate_loss")}}
        if on_epoch_end is not None:
            entry.update(on_epoch_end(epoch))
        print(f"epoch {epoch} done: " + ", ".join(f"{k} {v:.4f}" for k, v in entry.items() if isinstance(v, float)),
              flush=True)
        epochs_log.append(entry)
        if step >= total_steps:
            break
    return steps_log, epochs_log
