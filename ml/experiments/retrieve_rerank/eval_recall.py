"""Score a trained retriever on the calibration partition: recall@N JSON + top-49 candidates CSV.

  ml/.venv/Scripts/python.exe ml/experiments/retrieve_rerank/eval_recall.py --loss bce

Reads config.RR_PHASE0_DIR/<loss>/ (or --model-dir) and writes recall.json and calibration_candidates.csv
there. Recall is over calibration cases with >= 1 silver term; the CSV covers EVERY calibration case
(cancer and non-cancer) for the pilot. Smoke: --limit 50.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch

import config
import io_utils
from experiments.retrieve_rerank import batching, recall_metrics, report_text, retriever
from experiments.retrieve_rerank.data import load_phase0_data

TOP_K = 49
NS = (5, 10, 20, 30, 49)
DEFAULT_MAX_LENGTH = 4096


@torch.inference_mode()
def score_cases(model, tokenizer, texts: list[str], max_length: int, token_budget: int, device) -> tuple[np.ndarray, int]:
    """(len(texts), n_terms) float32 logits, in input order, and the count of reports over max_length."""
    token_ids, over = batching.encode_reports(tokenizer, texts, max_length)
    lengths = [len(x) for x in token_ids]
    logits = np.zeros((len(texts), len(model.terms)), dtype=np.float32)
    model.eval()
    for micro in batching.micro_batches(list(range(len(texts))), lengths, token_budget):
        ids, mask = batching.collate(token_ids, micro, tokenizer.pad_token_id, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits[micro] = model(ids, mask).float().cpu().numpy()
    return logits, over


def trained_max_length(model_dir: Path, default: int = DEFAULT_MAX_LENGTH) -> int:
    """The ``--max-length`` the model was trained with (train_log.json), else ``default``."""
    log = model_dir / "train_log.json"
    if not log.is_file():
        return default
    return int(json.loads(log.read_text(encoding="utf-8"))["summary"]["args"].get("max_length", default))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--loss", choices=sorted(retriever.LOSSES), required=True)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--max-length", type=int, default=None, help="default: the value the model was trained with")
    parser.add_argument("--token-budget", type=int, default=32768)
    parser.add_argument("--limit", type=int, default=0, help="smoke: only the first N calibration cases")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    model_dir = args.model_dir or config.RR_PHASE0_DIR / args.loss
    out_dir = args.out_dir or model_dir
    args.max_length = args.max_length or trained_max_length(model_dir)
    device = torch.device("cuda")

    data = load_phase0_data()
    case_ids = data.calibration_ids[:args.limit] if args.limit else data.calibration_ids
    model, tokenizer = retriever.load_retriever(model_dir)
    model.to(device)
    by_case = report_text.build_report_texts(data.reports)

    start = time.time()
    logits, over = score_cases(model, tokenizer, [by_case[c] for c in case_ids], args.max_length,
                               args.token_budget, device)
    seconds = time.time() - start
    top = np.argsort(-logits, axis=1)[:, :TOP_K]
    ranked = {c: [model.terms[j] for j in top[i]] for i, c in enumerate(case_ids)}

    gold = {c: data.gold[c] for c in case_ids if c in data.gold}
    report = recall_metrics.recall_report(ranked, gold, data.term_group, data.train_freq, NS)
    report.update(loss=args.loss, model_dir=str(model_dir), calibration_cases_scored=len(case_ids),
                  calibration_cases_over_max_length=over, max_length=args.max_length, scoring_seconds=seconds)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "recall.json").write_text(json.dumps(report, indent=1), encoding="utf-8")

    rows = [{"case_id": c, "is_cancer": int(c in data.gold), "rank": r + 1, "term": model.terms[j],
             "group": data.term_group[model.terms[j]], "score": float(logits[i, j])}
            for i, c in enumerate(case_ids) for r, j in enumerate(top[i])]
    io_utils.write_csv(pd.DataFrame(rows), out_dir / "calibration_candidates.csv")

    print(f"{args.loss}: {len(case_ids)} cases ({len(gold)} cancer), {over} over {args.max_length} tokens, {seconds:.0f}s")
    for n in NS:
        print(f"  @{n}: pair {report['pair_recall'][n]:.3f}  case {report['case_recall'][n]:.3f}  "
              f"group {report['group_pair_recall'][n]:.3f}")


if __name__ == "__main__":
    main()
