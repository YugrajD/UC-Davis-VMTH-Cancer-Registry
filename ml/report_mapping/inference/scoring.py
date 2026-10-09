"""Batched inference: ONE encoder pass per report gives the term logits and the gate logit.

Numbers only: the ranked top ``TOP_K`` terms per case with their probabilities, and the gate probability.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from report_mapping import batching
from report_mapping.model import JointClassifier

TOP_K = 49
DEFAULT_TOKEN_BUDGET = 32768


@torch.inference_mode()
def logits_for(model: JointClassifier, tokenizer, texts: list[str], max_length: int, device,
               token_budget: int = DEFAULT_TOKEN_BUDGET) -> tuple[np.ndarray, np.ndarray, int]:
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


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x.astype(np.float64)))


@dataclass(frozen=True)
class RankedScores:
    terms: list[str]
    case_ids: list[str]
    term_idx: np.ndarray    # (N, TOP_K) int, best first
    probs: np.ndarray       # (N, TOP_K) float, descending
    gate_probs: np.ndarray  # (N,) float

    def ranked_terms(self, i: int, n: int = TOP_K) -> list[str]:
        return [self.terms[j] for j in self.term_idx[i][:n]]


def rank(case_ids: list[str], terms: list[str], term_logits: np.ndarray, gate_logits: np.ndarray) -> RankedScores:
    top = np.argsort(-term_logits, axis=1, kind="stable")[:, :TOP_K]
    probs = sigmoid(np.take_along_axis(term_logits, top, axis=1))
    return RankedScores(terms, list(case_ids), top, probs, sigmoid(gate_logits))
