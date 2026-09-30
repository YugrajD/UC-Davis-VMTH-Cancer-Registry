"""Ask OpenJev about a stratified sample of calibration cases; save raw probabilities (scoring is separate).

  ml/.venv/Scripts/python.exe ml/experiments/retrieve_rerank/openjev_pilot.py \\
      --candidates-csv ml/output/rr_phase0/<loss>/calibration_candidates.csv --n 30

OpenJev (community Qwen3.5 NLI cross-encoder, zero-shot) is used the way its card describes typed decisions:
premise = the report, each option = one hypothesis built with the card's decide() template, one
forward pass per hypothesis, P(entailment) read off. All hypotheses of a case share the report prefix
(predict_hypotheses). Per case: Q1 (yes / no hypotheses) and Q2 (one hypothesis per candidate term).

The candidate set is the retriever's top-n UNION the case's gold terms, so one pass serves both the
real run (top-n only) and the ORACLE run (gold forced in); pilot_score.py separates them by ``rank``.
Output: <out-dir>/cases.csv (the sample, fixed once written) and raw_<loss>_<tag>.jsonl (resumable; <loss> is
the candidates CSV's parent directory name, so two retrievers never share a raw file).
Smoke: --limit 10.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import torch
from huggingface_hub import hf_hub_download, snapshot_download

import config
import io_utils
from experiments.retrieve_rerank import report_text
from experiments.retrieve_rerank.data import load_phase0_data

# The community repo ships its own modeling code, so pin the revision that was read before use.
OPENJEV_REPO = "AlexWortega/openjev"
OPENJEV_REVISION = "26de23c44b67586b4bea31c0ef2e016e3068ae66"
OPENJEV_SUBFOLDER = "qwen3.5-4b-nli-v5"
HYPOTHESIS_RESERVE_TOKENS = 192  # room left for the template + hypothesis when the report is cut

# The card's decide() hypothesis template.
TEMPLATE = 'The answer to "{instr}" is {label}: {crit}'
Q1_INSTR = "Is a neoplasm diagnosed in this pathology report?"
Q1_YES = TEMPLATE.format(instr=Q1_INSTR, label="yes", crit="A neoplasm is diagnosed.")
Q1_NO = TEMPLATE.format(instr=Q1_INSTR, label="no", crit="No neoplasm is diagnosed.")
Q2_INSTR = "Which neoplasm diagnosis does this pathology report state?"


def candidate_hypothesis(term: str, group: str) -> str:
    return TEMPLATE.format(instr=Q2_INSTR, label=term, crit=group)


def load_openjev(device: str = "cuda", max_len: int = 4096):
    code = hf_hub_download(OPENJEV_REPO, "modeling_openjev.py", revision=OPENJEV_REVISION)
    sys.path.insert(0, str(Path(code).parent))
    from modeling_openjev import OpenJevCrossEncoder
    snapshot = snapshot_download(OPENJEV_REPO, revision=OPENJEV_REVISION, allow_patterns=[f"{OPENJEV_SUBFOLDER}/*"])
    return OpenJevCrossEncoder(snapshot, subfolder=OPENJEV_SUBFOLDER, device=device, max_len=max_len)


def sample_cases(data, per_stratum: int, seed: int) -> pd.DataFrame:
    """Half cancer / half non-cancer calibration cases, interleaved so any prefix stays balanced."""
    rng = random.Random(seed)
    cancer = rng.sample(sorted(data.calibration_cancer_ids()), per_stratum)
    non_cancer = rng.sample(sorted(set(data.calibration_ids) - set(data.gold)), per_stratum)
    rows = [{"case_id": c, "is_cancer": flag} for pair in zip(cancer, non_cancer)
            for c, flag in zip(pair, (1, 0))]
    return pd.DataFrame(rows)


def chunk_sizes(total: int, max_size: int) -> list[int]:
    """Balanced chunk sizes (each <= max_size): the shared-prefix path needs >= 3 hypotheses per call."""
    parts = -(-total // max_size)
    return [total // parts + (i < total % parts) for i in range(parts)]


def fit_premise(jev, text: str, budget: int) -> tuple[str, int, bool]:
    """(premise, n_tokens, truncated): the report cut to ``budget`` tokens if longer."""
    ids = jev.tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= budget:
        return text, len(ids), False
    return jev.tok.decode(ids[:budget]), len(ids), True


def read_done(raw_path: Path) -> set[str]:
    """Case ids already in the raw file. A torn (unparsable) LAST line from an interrupted run is dropped
    from the file so appends stay valid JSONL; an unparsable line anywhere else is an error."""
    if not raw_path.is_file():
        return set()
    lines = raw_path.read_text(encoding="utf-8").splitlines()
    records = []
    for k, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if k != len(lines) - 1:
                raise
            print(f"dropping 1 truncated last line of {raw_path.name}", flush=True)
    raw_path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8", newline="\n")
    return {r["case_id"] for r in records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates-csv", type=Path, required=True)
    parser.add_argument("--n", type=int, default=30, help="top-n retriever candidates per case")
    parser.add_argument("--per-stratum", type=int, default=500, help="cancer and non-cancer cases each")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="smoke: only the first N sampled cases")
    parser.add_argument("--max-len", type=int, default=4096, help="OpenJev input tokens (the card allows 8k)")
    parser.add_argument("--chunk", type=int, default=12, help="max hypotheses per forward call (memory)")
    parser.add_argument("--out-dir", type=Path, default=config.RR_PHASE0_DIR / "pilot")
    parser.add_argument("--tag", default=None, help="raw file suffix after the loss; default n<N>")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = load_phase0_data()
    cases_csv = args.out_dir / "cases.csv"
    if cases_csv.is_file():
        sample = io_utils.read_csv(cases_csv, encoding="utf-8", dtype={"case_id": str})
    else:
        sample = sample_cases(data, args.per_stratum, args.seed)
        io_utils.write_csv(sample, cases_csv)
    if args.limit:
        sample = sample.iloc[:args.limit]

    candidates = io_utils.read_csv(args.candidates_csv, encoding="utf-8", dtype={"case_id": str})
    candidates = candidates[candidates["rank"] <= args.n]
    ranked = {c: list(zip(g["term"], g["rank"])) for c, g in candidates.groupby("case_id")}

    loss = args.candidates_csv.resolve().parent.name
    raw_path = args.out_dir / f"raw_{loss}_{args.tag or f'n{args.n}'}.jsonl"
    done = read_done(raw_path)
    texts = report_text.build_report_texts(data.reports)
    jev = load_openjev(max_len=args.max_len)

    todo = [(c, int(f)) for c, f in zip(sample["case_id"], sample["is_cancer"]) if c not in done]
    print(f"{len(todo)} cases to run ({len(done)} already in {raw_path.name})", flush=True)
    with open(raw_path, "a", encoding="utf-8", newline="\n") as out:
        for k, (case_id, is_cancer) in enumerate(todo):
            start = time.time()
            gold = data.gold.get(case_id, frozenset())
            terms = dict(ranked[case_id])                      # term -> retriever rank
            for term in sorted(gold - set(terms)):
                terms[term] = None                             # oracle-only candidates
            premise, n_tokens, truncated = fit_premise(jev, texts[case_id], args.max_len - HYPOTHESIS_RESERVE_TOKENS)
            hypotheses = [Q1_YES, Q1_NO] + [candidate_hypothesis(t, data.term_group[t]) for t in terms]
            probs, pos = [], 0
            for size in chunk_sizes(len(hypotheses), args.chunk):
                probs += jev.predict_hypotheses(premise, hypotheses[pos:pos + size]).tolist()
                pos += size
            torch.cuda.synchronize()
            record = {
                "case_id": case_id, "is_cancer": is_cancer, "n_premise_tokens": n_tokens, "truncated": truncated,
                "seconds": time.time() - start, "q1_yes": probs[0], "q1_no": probs[1],
                "candidates": [{"term": t, "rank": r, "gold": t in gold, "probs": p}
                               for (t, r), p in zip(terms.items(), probs[2:])],
            }
            out.write(json.dumps(record) + "\n")
            out.flush()
            if (k + 1) % 25 == 0 or k == 0:
                print(f"{k + 1}/{len(todo)} last {record['seconds']:.2f}s", flush=True)
    print(f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB -> {raw_path}")


if __name__ == "__main__":
    main()
