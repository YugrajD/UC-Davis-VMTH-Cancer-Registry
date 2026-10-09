"""Length-bucketed batches: similar-length cases share an optimizer step, then the step is cut into
token-budgeted micro-batches (gradient accumulation) so 4096-token inputs fit in 16 GB."""

from __future__ import annotations

import random

import torch


def bucketed_steps(lengths: list[int], step_size: int, seed: int, block_steps: int = 32) -> list[list[int]]:
    """Shuffle, sort each block of ``block_steps`` steps by length, cut into steps, shuffle the steps."""
    rng = random.Random(seed)
    order = list(range(len(lengths)))
    rng.shuffle(order)
    block = step_size * block_steps
    steps = []
    for start in range(0, len(order), block):
        chunk = sorted(order[start:start + block], key=lambda i: lengths[i])
        steps += [chunk[s:s + step_size] for s in range(0, len(chunk), step_size)]
    rng.shuffle(steps)
    return steps


def micro_batches(indices: list[int], lengths: list[int], token_budget: int) -> list[list[int]]:
    """Cut one length-sorted group of cases into batches whose padded size stays under the budget."""
    indices = sorted(indices, key=lambda i: lengths[i])
    batches, current = [], []
    for i in indices:
        if current and lengths[i] * (len(current) + 1) > token_budget:
            batches.append(current)
            current = []
        current.append(i)
    if current:
        batches.append(current)
    return batches


def collate(token_ids: list[list[int]], indices: list[int], pad_id: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(token_ids[i]) for i in indices)
    ids = torch.full((len(indices), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(indices), width), dtype=torch.long)
    for row, i in enumerate(indices):
        ids[row, :len(token_ids[i])] = torch.tensor(token_ids[i])
        mask[row, :len(token_ids[i])] = 1
    return ids.to(device), mask.to(device)


def encode_reports(tokenizer, texts: list[str], max_length: int) -> tuple[list[list[int]], int]:
    """Token ids per text, cut to ``max_length`` keeping the closing [SEP]. Also the number of texts
    that were longer than ``max_length`` (counted before the cut)."""
    ids = tokenizer(texts, add_special_tokens=True, truncation=False)["input_ids"]
    over = sum(len(x) > max_length for x in ids)
    return [x if len(x) <= max_length else x[:max_length - 1] + [x[-1]] for x in ids], over
