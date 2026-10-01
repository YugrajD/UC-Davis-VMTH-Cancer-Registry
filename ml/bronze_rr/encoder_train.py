"""The one fine-tuning loop for both encoders (retriever and gate): length-bucketed steps cut into
token-budgeted micro-batches (gradient accumulation), bf16 autocast, gradient checkpointing, AdamW with
linear warmup then linear decay. Callers prepare token ids and targets and decide what to do after each epoch.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
import torch

from bronze_rr import batching


def fit(model, tokenizer, token_ids: list[list[int]], targets: np.ndarray, loss_fn, *, epochs: int, lr: float,
        head_lr: float, weight_decay: float, warmup_frac: float, batch_size: int, token_budget: int, seed: int,
        max_steps: int = 0, on_epoch_end: Callable[[int], dict] | None = None, device="cuda") -> tuple[list, list]:
    """Train ``model`` in place on (token_ids, targets); ``model`` must already be on ``device``.

    ``loss_fn(logits, targets) -> scalar`` is a mean over the cases it is given. ``on_epoch_end(epoch)`` runs after
    each epoch (model still in train mode) and returns a dict that is logged. ``max_steps`` > 0 is a smoke cut-off.
    Returns (per-step log, per-epoch log).
    """
    lengths = [len(x) for x in token_ids]
    targets_t = torch.tensor(targets)
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": lr},
        {"params": model.head.parameters(), "lr": head_lr},
    ], weight_decay=weight_decay)
    steps_per_epoch = -(-len(token_ids) // batch_size)
    total_steps = max_steps or steps_per_epoch * epochs
    warmup = max(1, int(warmup_frac * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min((s + 1) / warmup, max(0.0, (total_steps - s) / max(1, total_steps - warmup))))

    steps_log, epochs_log, step = [], [], 0
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        for step_cases in batching.bucketed_steps(lengths, batch_size, seed + epoch):
            step_start, total_loss = time.time(), 0.0
            for micro in batching.micro_batches(step_cases, lengths, token_budget):
                ids, mask = batching.collate(token_ids, micro, tokenizer.pad_token_id, device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(ids, mask)
                loss = loss_fn(logits.float(), targets_t[micro].to(device)) * len(micro) / len(step_cases)
                loss.backward()
                total_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            epoch_losses.append(total_loss)
            steps_log.append({"step": step, "epoch": epoch, "loss": total_loss, "seconds": time.time() - step_start,
                              "max_len": max(lengths[i] for i in step_cases)})
            if step % 20 == 0 or step == 1:
                print(f"epoch {epoch} step {step}/{total_steps} loss {total_loss:.4f} "
                      f"{steps_log[-1]['seconds']:.1f}s peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB",
                      flush=True)
            if step >= total_steps:
                break
        entry = {"epoch": epoch, "train_loss": float(np.mean(epoch_losses)), "steps": len(epoch_losses)}
        if on_epoch_end is not None:
            entry.update(on_epoch_end(epoch))
            print(f"epoch {epoch} done: " + ", ".join(f"{k} {v:.4f}" for k, v in entry.items() if isinstance(v, float)),
                  flush=True)
        epochs_log.append(entry)
        if step >= total_steps:
            break
    return steps_log, epochs_log
