"""Fine-tune the term retriever end to end on train cases that have >= 1 silver term.

  ml/.venv/Scripts/python.exe ml/experiments/retrieve_rerank/train.py --loss bce
  ml/.venv/Scripts/python.exe ml/experiments/retrieve_rerank/train.py --loss softmax

Saves to config.RR_PHASE0_DIR/<loss>/ (or --out). Smoke: --limit-train 640 --max-steps 20.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

import config
from experiments.retrieve_rerank import batching
from experiments.retrieve_rerank import report_text
from experiments.retrieve_rerank import retriever
from experiments.retrieve_rerank.data import load_phase0_data
from report_mapping.training.recipe import PRODUCTION_SEED, seed_all


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--loss", choices=sorted(retriever.LOSSES), required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=32, help="cases per optimizer step")
    parser.add_argument("--token-budget", type=int, default=16384, help="padded tokens per micro-batch")
    parser.add_argument("--lr", type=float, default=5e-5, help="encoder learning rate")
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=PRODUCTION_SEED)
    parser.add_argument("--limit-train", type=int, default=0, help="smoke: only the first N train cases")
    parser.add_argument("--max-steps", type=int, default=0, help="smoke: stop after N optimizer steps")
    parser.add_argument("--force-max-length-batch", action="store_true",
                        help="smoke: pad every input to --max-length (worst-case VRAM and step time)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out = args.out or config.RR_PHASE0_DIR / args.loss
    seed_all(args.seed)
    device = torch.device("cuda")

    data = load_phase0_data()
    train_ids = data.train_ids[:args.limit_train] if args.limit_train else data.train_ids
    terms = data.vocab
    term_index = {t: i for i, t in enumerate(terms)}
    targets = np.zeros((len(train_ids), len(terms)), dtype=np.float32)
    for row, case_id in enumerate(train_ids):
        for term in data.gold[case_id]:
            targets[row, term_index[term]] = 1.0

    tokenizer = retriever.load_tokenizer()
    by_case = report_text.build_report_texts(data.reports)
    token_ids, over = batching.encode_reports(tokenizer, [by_case[c] for c in train_ids], args.max_length)
    lengths = [len(x) for x in token_ids]
    print(f"train cases {len(train_ids)}, terms {len(terms)}, reports over {args.max_length} tokens: {over} "
          f"({100 * over / len(train_ids):.1f}%), mean tokens {np.mean(lengths):.0f}", flush=True)
    if args.force_max_length_batch:
        token_ids = [(x * (args.max_length // len(x) + 1))[:args.max_length] for x in token_ids]
        lengths = [len(x) for x in token_ids]

    model = retriever.TermRetriever(retriever.new_encoder(), terms).to(device)
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.init_bias_from_frequency(torch.tensor(targets.mean(0), device=device))
    loss_fn = retriever.LOSSES[args.loss]
    targets_t = torch.tensor(targets)

    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": args.lr},
        {"params": model.head.parameters(), "lr": args.head_lr},
    ], weight_decay=args.weight_decay)
    steps_per_epoch = -(-len(train_ids) // args.batch_size)
    total_steps = args.max_steps or steps_per_epoch * args.epochs
    warmup = max(1, int(args.warmup_frac * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min((s + 1) / warmup, max(0.0, (total_steps - s) / max(1, total_steps - warmup))))

    log, step, start = [], 0, time.time()
    torch.cuda.reset_peak_memory_stats()
    model.train()
    for epoch in range(args.epochs):
        for step_cases in batching.bucketed_steps(lengths, args.batch_size, args.seed + epoch):
            step_start = time.time()
            total_loss = 0.0
            for micro in batching.micro_batches(step_cases, lengths, args.token_budget):
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
            log.append({"step": step, "epoch": epoch, "loss": total_loss, "seconds": time.time() - step_start,
                        "max_len": max(lengths[i] for i in step_cases)})
            if step % 20 == 0 or step == 1:
                print(f"epoch {epoch} step {step}/{total_steps} loss {total_loss:.4f} "
                      f"{log[-1]['seconds']:.1f}s peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
            if step >= total_steps:
                break
        if step >= total_steps:
            break

    summary = {"loss": args.loss, "args": {k: str(v) for k, v in vars(args).items()}, "train_cases": len(train_ids),
               "terms": len(terms), "reports_over_max_length": over, "steps": step,
               "train_seconds": time.time() - start, "mean_step_seconds": float(np.mean([r["seconds"] for r in log])),
               "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30, "steps_per_epoch": steps_per_epoch}
    model.save(out, tokenizer)
    (out / "train_log.json").write_text(json.dumps({"summary": summary, "steps": log}), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
