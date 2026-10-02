"""Train ONE ModernBERT jointly for the term retriever and the gate (``joint.JointClassifier``).

Data: the gate's training universe (``data.gate_ids``) minus the gate's fixed 5% held-out set, for BOTH heads. The
term vocabulary is the retriever's (``data.vocab``, sorted train-partition silver terms): leaving the held-out
cases out of training does not change it, because the held-out cases are not removed from the vocabulary, only
from the loss; a term whose every train case is held out would just keep its frequency-prior bias (counted in the
summary as ``terms_without_train_positive``).

Loss per optimizer step: term loss (BCE summed over terms, averaged over the step's CANCER cases) + gate_weight *
gate loss (BCE averaged over all the step's cases); see ``joint.step_loss``. The recipe is the retriever's and the
gate's; the final epoch is kept. After each epoch the held-out set is scored and logged.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from bronze_rr import batching, encoder as enc, gate as gate_mod, joint, joint_scoring, report_text
from bronze_rr.data import BronzeData
from bronze_rr.train import Recipe
from report_mapping.training.recipe import seed_all

RECALL_AT = 20


@dataclass(frozen=True)
class JointRecipe(Recipe):
    gate_weight: float = 1.0   # total loss = term loss + gate_weight * gate loss


def heldout_metrics(term_logits: np.ndarray, gate_logits: np.ndarray, gate_targets: np.ndarray,
                    gold_idx: list[frozenset[int]]) -> dict:
    """Gate AUC and BCE over all held-out cases; term top-1 accuracy and pair recall@RECALL_AT over the held-out
    CANCER cases (``gate_targets`` == 1; ``gold_idx[i]`` = vocabulary indices of case i's silver terms)."""
    loss = float(F.binary_cross_entropy_with_logits(torch.tensor(gate_logits), torch.tensor(gate_targets)))
    cancer = np.flatnonzero(gate_targets == 1.0)
    top = np.argsort(-term_logits[cancer], axis=1, kind="stable")[:, :RECALL_AT]
    top1 = np.mean([top[k, 0] in gold_idx[i] for k, i in enumerate(cancer)])
    hits = sum(len(gold_idx[i].intersection(top[k].tolist())) for k, i in enumerate(cancer))
    pairs = sum(len(gold_idx[i]) for i in cancer)
    return {"heldout_gate_auc": gate_mod.roc_auc(gate_logits, gate_targets), "heldout_gate_loss": loss,
            "heldout_term_top1": float(top1), f"heldout_term_pair_recall_at_{RECALL_AT}": hits / pairs}


def fit(model, tokenizer, token_ids: list[list[int]], term_targets: np.ndarray, gate_targets: np.ndarray,
        recipe: JointRecipe, on_epoch_end, device="cuda") -> tuple[list, list]:
    """Train ``model`` in place (on ``device``). Same optimizer, schedule, bucketing, micro-batching, bf16 autocast
    and gradient checkpointing as ``encoder_train.fit``; the loss is ``joint.step_loss`` normalised by the step's
    counts across micro-batches. ``on_epoch_end(epoch) -> dict`` is logged. Returns (per-step log, per-epoch log)."""
    lengths = [len(x) for x in token_ids]
    term_t, gate_t = torch.tensor(term_targets), torch.tensor(gate_targets)
    cancer_t = gate_t  # 0/1: a case has term targets iff it is a gate positive (checked by the caller)
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": recipe.lr},
        {"params": list(model.head.parameters()) + list(model.gate_head.parameters()), "lr": recipe.head_lr},
    ], weight_decay=recipe.weight_decay)
    steps_per_epoch = -(-len(token_ids) // recipe.batch_size)
    total_steps = recipe.max_steps or steps_per_epoch * recipe.epochs
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
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    term_logits, gate_logits = model(ids, mask)
                term, gate = joint.step_loss(term_logits.float(), gate_logits.float(), term_t[micro].to(device),
                                             gate_t[micro].to(device), cancer_t[micro].to(device), n_cancer,
                                             len(step_cases))
                (term + recipe.gate_weight * gate).backward()
                term_total += term.item()
                gate_total += gate.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            steps_log.append({"step": step, "epoch": epoch, "term_loss": term_total, "gate_loss": gate_total,
                              "loss": term_total + recipe.gate_weight * gate_total, "seconds": time.time() - step_start,
                              "max_len": max(lengths[i] for i in step_cases)})
            if step % 20 == 0 or step == 1:
                print(f"epoch {epoch} step {step}/{total_steps} term {term_total:.3f} gate {gate_total:.4f} "
                      f"{steps_log[-1]['seconds']:.1f}s peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB",
                      flush=True)
            if step >= total_steps:
                break
        this_epoch = [s for s in steps_log if s["epoch"] == epoch]
        entry = {"epoch": epoch, "steps": len(this_epoch),
                 **{f"train_{k}": float(np.mean([s[k] for s in this_epoch])) for k in ("loss", "term_loss", "gate_loss")}}
        entry.update(on_epoch_end(epoch))
        print(f"epoch {epoch} done: " + ", ".join(f"{k} {v:.4f}" for k, v in entry.items() if isinstance(v, float)),
              flush=True)
        epochs_log.append(entry)
        if step >= total_steps:
            break
    return steps_log, epochs_log


def train_joint(data: BronzeData, out: Path, recipe: JointRecipe, device="cuda") -> dict:
    if (out / "train_log.json").exists():
        raise SystemExit(f"{out} already holds a trained model; choose another --out or delete it")
    seed_all(recipe.seed)
    n = recipe.limit_train or len(data.gate_ids)
    case_ids, gate_targets = data.gate_ids[:n], data.gate_targets[:n].astype(np.float32)
    inconsistent = [c for c, y in zip(case_ids, gate_targets) if (y == 1.0) != (c in data.gold)]
    if inconsistent:
        raise ValueError(f"{len(inconsistent)} gate case(s) disagree with the silver terms about being cancer")
    fit_idx, held_idx = gate_mod.split_holdout(gate_targets)

    term_index = {t: i for i, t in enumerate(data.vocab)}
    term_targets = np.zeros((len(case_ids), len(data.vocab)), dtype=np.float32)
    for row, case_id in enumerate(case_ids):
        for term in data.gold.get(case_id, ()):
            term_targets[row, term_index[term]] = 1.0
    fit_cancer = gate_targets[fit_idx] == 1.0
    term_freq = term_targets[fit_idx][fit_cancer].mean(0)

    tokenizer = enc.load_tokenizer()
    by_case = report_text.build_report_texts(data.reports)
    token_ids, over = batching.encode_reports(tokenizer, [by_case[case_ids[i]] for i in fit_idx], recipe.max_length)
    held_texts = [by_case[case_ids[i]] for i in held_idx]
    held_gate = gate_targets[held_idx]
    held_gold = [frozenset(term_index[t] for t in data.gold.get(case_ids[i], ())) for i in held_idx]
    print(f"joint: train {len(fit_idx)} ({int(fit_cancer.sum())} cancer), held-out {len(held_idx)} "
          f"({int(held_gate.sum())} cancer), terms {len(data.vocab)}, reports over {recipe.max_length} tokens: {over}",
          flush=True)

    model = joint.new_joint(data.vocab).to(device)
    model.init_biases(torch.tensor(term_freq), float(gate_targets[fit_idx].mean()))

    def after_epoch(epoch: int) -> dict:
        term_logits, gate_logits, _ = joint_scoring.logits_for(model, tokenizer, held_texts, recipe.max_length,
                                                               2 * recipe.token_budget, device)
        return heldout_metrics(term_logits, gate_logits, held_gate, held_gold)

    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    steps, epochs = fit(model, tokenizer, token_ids, term_targets[fit_idx], gate_targets[fit_idx], recipe, after_epoch,
                        device=device)
    model.save(out, tokenizer)   # the final epoch is kept
    summary = {"train_cases": len(fit_idx), "train_cancer_cases": int(fit_cancer.sum()), "heldout_cases": len(held_idx),
               "heldout_cancer_cases": int(held_gate.sum()), "terms": len(data.vocab),
               "terms_without_train_positive": int((term_targets[fit_idx][fit_cancer].sum(0) == 0).sum()),
               "reports_over_max_length": over, "steps": len(steps), "train_seconds": time.time() - start,
               "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30, "final_epoch": epochs[-1]}
    summary = {**summary, "args": {k: str(v) for k, v in asdict(recipe).items()}}
    (out / "train_log.json").write_text(json.dumps({"summary": summary, "steps": steps, "epochs": epochs}),
                                        encoding="utf-8")
    return summary
