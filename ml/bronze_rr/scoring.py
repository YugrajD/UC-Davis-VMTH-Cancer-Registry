"""Batched inference for the retriever and the gate, cached as numbers (case_id + floats, never text).

Cache files, one per (model, partition) under ``config.RR_PHASE1_DIR/scores/`` (or ``out_dir``):
  retriever_<tag>_<partition>.csv   case_id, t00..t48 (term index into the model's terms.json, best first),
                                    p00..p48 (sigmoid of that term's logit)
  gate_<tag>_<partition>.csv        case_id, prob
``tag`` is "<parent dir>-<dir name>" of the model dir. The bake-off and the freeze re-use these files.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import config
import io_utils
from bronze_rr import batching, encoder as enc, report_text, retriever
from bronze_rr.data import BronzeData

TOP_K = 49
DEFAULT_MAX_LENGTH = 4096
SCORES_DIR = config.RR_PHASE1_DIR / "scores"
TEST_FLAG = "--i-understand-this-spends-the-test-set"


def check_partition_allowed(partition: str, allow_test: bool) -> None:
    """Scoring on the test partition (or "all", which includes it) needs the explicit flag."""
    if partition in ("test", "all") and not allow_test:
        raise SystemExit(f"partition {partition!r} includes the test set; pass {TEST_FLAG} to spend it")


def model_tag(model_dir: Path) -> str:
    return f"{model_dir.parent.name}-{model_dir.name}"


def trained_max_length(model_dir: Path, default: int = DEFAULT_MAX_LENGTH) -> int:
    """The ``--max-length`` the model was trained with (train_log.json), else ``default``."""
    log = model_dir / "train_log.json"
    if not log.is_file():
        return default
    return int(json.loads(log.read_text(encoding="utf-8"))["summary"]["args"].get("max_length", default))


@torch.inference_mode()
def logits_for(model, tokenizer, texts: list[str], max_length: int, token_budget: int, device) -> tuple[np.ndarray, int]:
    """(len(texts), n_out) float32 logits in input order, and the count of texts over max_length."""
    token_ids, over = batching.encode_reports(tokenizer, texts, max_length)
    lengths = [len(x) for x in token_ids]
    logits = np.zeros((len(texts), model.head.out_features), dtype=np.float32)
    model.eval()
    for micro in batching.micro_batches(list(range(len(texts))), lengths, token_budget):
        ids, mask = batching.collate(token_ids, micro, tokenizer.pad_token_id, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits[micro] = model(ids, mask).float().cpu().numpy()
    return logits, over


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x.astype(np.float64)))


@dataclass(frozen=True)
class RetrieverScores:
    terms: list[str]
    case_ids: list[str]
    term_idx: np.ndarray   # (N, TOP_K) int, best first
    probs: np.ndarray      # (N, TOP_K) float, descending

    def subset(self, case_ids: list[str]) -> "RetrieverScores":
        position = {c: i for i, c in enumerate(self.case_ids)}
        rows = [position[c] for c in case_ids]
        return RetrieverScores(self.terms, list(case_ids), self.term_idx[rows], self.probs[rows])

    def ranked_terms(self, i: int, n: int = TOP_K) -> list[str]:
        return [self.terms[j] for j in self.term_idx[i][:n]]


def retriever_frame(case_ids: list[str], logits: np.ndarray) -> pd.DataFrame:
    top = np.argsort(-logits, axis=1, kind="stable")[:, :TOP_K]
    probs = sigmoid(np.take_along_axis(logits, top, axis=1))
    columns = {"case_id": case_ids}
    columns.update({f"t{k:02d}": top[:, k] for k in range(TOP_K)})
    columns.update({f"p{k:02d}": probs[:, k] for k in range(TOP_K)})
    return pd.DataFrame(columns)


def retriever_scores_from_frame(frame: pd.DataFrame, terms: list[str]) -> RetrieverScores:
    return RetrieverScores(terms, frame["case_id"].astype(str).tolist(),
                           frame[[f"t{k:02d}" for k in range(TOP_K)]].to_numpy(dtype=int),
                           frame[[f"p{k:02d}" for k in range(TOP_K)]].to_numpy(dtype=float))


def _cache_path(kind: str, model_dir: Path, partition: str, out_dir: Path) -> Path:
    return out_dir / f"{kind}_{model_tag(model_dir)}_{partition}.csv"


def _partitions(partition: str) -> tuple[str, ...]:
    return ("train", "calibration", "test") if partition == "all" else (partition,)


def _score_partition(kind: str, model_dir: Path, data: BronzeData, partition: str, *, out_dir: Path, limit: int,
                     token_budget: int, device) -> pd.DataFrame:
    """Cache hit -> the cached frame; miss -> run the model over the partition's cases and write the cache."""
    path = _cache_path(kind, model_dir, partition, out_dir)
    if path.is_file():
        return io_utils.read_csv(path, encoding="utf-8", dtype={"case_id": str})
    case_ids = data.ids_for(partition)[:limit] if limit else data.ids_for(partition)
    if kind == "retriever":
        model, tokenizer = retriever.load_retriever(model_dir)
    else:
        model, tokenizer = enc.load_saved(model_dir, 1)
    model.to(device)
    by_case = report_text.build_report_texts(data.reports)
    start = time.time()
    logits, over = logits_for(model, tokenizer, [by_case[c] for c in case_ids], trained_max_length(model_dir),
                              token_budget, device)
    print(f"{kind} {model_tag(model_dir)} {partition}: {len(case_ids)} cases, {over} over max length, "
          f"{time.time() - start:.0f}s", flush=True)
    frame = retriever_frame(case_ids, logits) if kind == "retriever" else pd.DataFrame(
        {"case_id": case_ids, "prob": sigmoid(logits[:, 0])})
    out_dir.mkdir(parents=True, exist_ok=True)
    io_utils.write_csv(frame, path)
    del model
    torch.cuda.empty_cache()
    return frame


def retriever_scores(model_dir: Path, data: BronzeData, partition: str, *, out_dir: Path = SCORES_DIR, limit: int = 0,
                     token_budget: int = 32768, device="cuda") -> RetrieverScores:
    frames = [_score_partition("retriever", model_dir, data, p, out_dir=out_dir, limit=limit,
                               token_budget=token_budget, device=device) for p in _partitions(partition)]
    return retriever_scores_from_frame(pd.concat(frames, ignore_index=True), retriever.read_terms(model_dir))


def gate_scores(model_dir: Path, data: BronzeData, partition: str, *, out_dir: Path = SCORES_DIR, limit: int = 0,
                token_budget: int = 32768, device="cuda") -> dict[str, float]:
    """{case_id: probability} from the ModernBERT gate in ``model_dir``."""
    frames = [_score_partition("gate", model_dir, data, p, out_dir=out_dir, limit=limit,
                               token_budget=token_budget, device=device) for p in _partitions(partition)]
    frame = pd.concat(frames, ignore_index=True)
    return dict(zip(frame["case_id"].astype(str), frame["prob"].astype(float)))
