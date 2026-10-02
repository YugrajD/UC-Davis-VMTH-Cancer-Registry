"""Scoring a jointly trained model: ONE encoder pass per report gives the term logits and the gate logit, and fills
the two cache files the bake-off and the freeze already read, in their existing formats (``scoring.cache_path``):

  retriever_<tag>_<partition>.csv   top-49 term indices + probabilities
  gate_<tag>_<partition>.csv        case_id, prob
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import io_utils
from bronze_rr import batching, joint, report_text, retriever, scoring
from bronze_rr.data import BronzeData


@torch.inference_mode()
def logits_for(model: joint.JointClassifier, tokenizer, texts: list[str], max_length: int, token_budget: int, device
               ) -> tuple[np.ndarray, np.ndarray, int]:
    """(term logits (N, n_terms), gate logits (N,)) float32 in input order from one encoder pass, and the count of
    texts over max_length."""
    token_ids, over = batching.encode_reports(tokenizer, texts, max_length)
    lengths = [len(x) for x in token_ids]
    term = np.zeros((len(texts), model.head.out_features), dtype=np.float32)
    gate = np.zeros(len(texts), dtype=np.float32)
    kind = torch.device(device).type
    model.eval()
    for micro in batching.micro_batches(list(range(len(texts))), lengths, token_budget):
        ids, mask = batching.collate(token_ids, micro, tokenizer.pad_token_id, device)
        with torch.autocast(kind, dtype=torch.bfloat16, enabled=kind == "cuda"):
            term_logits, gate_logits = model(ids, mask)
        term[micro], gate[micro] = term_logits.float().cpu().numpy(), gate_logits.float().cpu().numpy()
    return term, gate, over


def _score_partition(model_dir: Path, data: BronzeData, partition: str, *, out_dir: Path, limit: int,
                     token_budget: int, device) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(retriever frame, gate frame). Both cached -> read; otherwise one pass writes whichever cache is missing."""
    retriever_csv = scoring.cache_path("retriever", model_dir, partition, out_dir)
    gate_csv = scoring.cache_path("gate", model_dir, partition, out_dir)
    if not (retriever_csv.is_file() and gate_csv.is_file()):
        case_ids = data.ids_for(partition)[:limit] if limit else data.ids_for(partition)
        model, tokenizer = joint.load_joint(model_dir)
        model.to(device)
        by_case = report_text.build_report_texts(data.reports)
        start = time.time()
        term, gate, over = logits_for(model, tokenizer, [by_case[c] for c in case_ids],
                                      scoring.trained_max_length(model_dir), token_budget, device)
        print(f"joint {scoring.model_tag(model_dir)} {partition}: {len(case_ids)} cases, {over} over max length, "
              f"{time.time() - start:.0f}s", flush=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        if not retriever_csv.is_file():
            io_utils.write_csv(scoring.retriever_frame(case_ids, term), retriever_csv)
        if not gate_csv.is_file():
            io_utils.write_csv(pd.DataFrame({"case_id": case_ids, "prob": scoring.sigmoid(gate)}), gate_csv)
        del model
        torch.cuda.empty_cache()
    return (io_utils.read_csv(retriever_csv, encoding="utf-8", dtype={"case_id": str}),
            io_utils.read_csv(gate_csv, encoding="utf-8", dtype={"case_id": str}))


def score(model_dir: Path, data: BronzeData, partition: str, *, out_dir: Path = scoring.SCORES_DIR, limit: int = 0,
          token_budget: int = 32768, device="cuda") -> tuple[scoring.RetrieverScores, dict[str, float]]:
    """(retriever scores, {case_id: gate probability}) of the joint model over a split partition or "all"."""
    pairs = [_score_partition(model_dir, data, p, out_dir=out_dir, limit=limit, token_budget=token_budget,
                              device=device) for p in scoring.partitions(partition)]
    retriever_frame = pd.concat([r for r, _ in pairs], ignore_index=True)
    gate_frame = pd.concat([g for _, g in pairs], ignore_index=True)
    gate_by_case = dict(zip(gate_frame["case_id"].astype(str), gate_frame["prob"].astype(float)))
    return scoring.retriever_scores_from_frame(retriever_frame, retriever.read_terms(model_dir)), gate_by_case
