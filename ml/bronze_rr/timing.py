"""CPU latency of the two deployment shapes of the gate, one report at a time (batch of 1, fp32, default threads):

  two models  retriever pass + a separate ModernBERT gate pass (two encoders)
  one model   one retriever pass + the shared head on its pooled vector

Each timed call covers tokenisation, the encoder forward and the head(s). The cases are the first N calibration
reports in report order. Only aggregates leave this module: report text is never printed or stored.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

import config
from bronze_rr import encoder as enc, report_text, retriever, scoring, shared_gate
from bronze_rr.data import BronzeData

CPU = "cpu"
BIG_BUDGET = 1 << 30   # one report per forward


def _seconds_per_report(model, tokenizer, texts: list[str], max_length: int, extra=None) -> tuple[list[float], list[int]]:
    """Per-report wall seconds and token counts. One untimed warm-up report comes first. ``extra(pooled)`` runs
    inside the timed region (the shared head)."""
    scoring.pooled_and_logits_for(model, tokenizer, texts[:1], max_length, BIG_BUDGET, CPU)
    seconds, tokens = [], []
    for text in texts:
        start = time.perf_counter()
        pooled, _, _ = scoring.pooled_and_logits_for(model, tokenizer, [text], max_length, BIG_BUDGET, CPU)
        if extra is not None:
            extra(pooled)
        seconds.append(time.perf_counter() - start)
        tokens.append(len(tokenizer(text, truncation=True, max_length=max_length)["input_ids"]))
    return seconds, tokens


def _stats(seconds: list[float]) -> dict:
    return {"mean": float(np.mean(seconds)), "median": float(np.median(seconds)), "p95": float(np.percentile(seconds, 95))}


def time_cpu(data: BronzeData, retriever_dir: Path, gate_dir: Path, shared_dir: Path | None, n: int) -> dict:
    """``gate_dir``: the separate ModernBERT gate. ``shared_dir``: the trained shared head, or None for an untrained
    stand-in head of the same shape (its cost is negligible next to an encoder pass)."""
    case_ids = data.partition_ids["calibration"][:n]
    by_case = report_text.build_report_texts(data.reports)
    texts = [by_case[c] for c in case_ids]
    max_length = scoring.trained_max_length(retriever_dir)

    ret_model, ret_tokenizer = retriever.load_retriever(retriever_dir)
    gate_model, gate_tokenizer = enc.load_saved(gate_dir, 1)
    if shared_dir is not None:
        shared_gate.verify_source(shared_dir, retriever_dir)
        head = shared_gate.load_head(shared_dir)
    else:
        head = shared_gate.SharedHead("linear", ret_model.head.in_features, torch.zeros(ret_model.head.in_features),
                                      torch.ones(ret_model.head.in_features))

    retriever_s, tokens = _seconds_per_report(ret_model, ret_tokenizer, texts, max_length)
    gate_s, _ = _seconds_per_report(gate_model, gate_tokenizer, texts, max_length)
    one_model_s, _ = _seconds_per_report(ret_model, ret_tokenizer, texts, max_length,
                                         extra=lambda pooled: shared_gate.head_probs(head, pooled))
    two_model_s = [a + b for a, b in zip(retriever_s, gate_s)]
    result = {"n_reports": len(texts), "torch_threads": torch.get_num_threads(), "device": CPU, "dtype": "float32",
              "mean_tokens_per_report": float(np.mean(tokens)), "shared_head": "trained" if shared_dir else "untrained stand-in",
              "two_model_seconds": _stats(two_model_s), "one_model_seconds": _stats(one_model_s),
              "retriever_pass_seconds": _stats(retriever_s), "gate_pass_seconds": _stats(gate_s)}
    result["two_over_one_mean_ratio"] = result["two_model_seconds"]["mean"] / result["one_model_seconds"]["mean"]
    return result


def save(result: dict, directory: Path = config.RR_PHASE1_DIR / "timing") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"cpu_n{result['n_reports']}.json"
    path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return path
