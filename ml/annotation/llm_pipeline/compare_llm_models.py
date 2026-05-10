"""Compare LM-Studio-hosted LLMs on the Tier-3 annotation task.

Builds a stratified sample of rows that previously reached Tier 3 in
`llm_annotation.csv`, runs every requested model through `run_llm_scan` on
that focused input, and reports per-model latency, format-compliance,
inter-model agreement, and disagreement examples.

Usage:
  ml/.venv/Scripts/python.exe ml/annotation/llm_pipeline/compare_llm_models.py \
    --models medgemma-27b-text-it qwen/qwen3.6-27b meta/llama-3.3-70b google/gemma-4-31b \
    --sample-size 80
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd

import config
from annotation.llm_pipeline.pipeline import LLMConfig, run_llm_scan


def build_sample(annotation_csv: Path, diagnoses_csv: Path, n: int, seed: int) -> pd.DataFrame:
    ann = pd.read_csv(annotation_csv)
    llm_rows = ann[ann["method"] == "LLM"].copy()

    rng = pd.Series(range(len(llm_rows))).sample(frac=1, random_state=seed).index
    llm_rows = llm_rows.iloc[rng].reset_index(drop=True)

    per_group_cap = max(2, n // max(llm_rows["matched_group"].nunique(), 1))
    sampled = (
        llm_rows.groupby("matched_group", group_keys=False)
        .apply(lambda g: g.head(per_group_cap))
        .reset_index(drop=True)
    )
    if len(sampled) > n:
        sampled = sampled.head(n)

    diag = pd.read_csv(diagnoses_csv, encoding="latin-1")
    keys = sampled[["case_id", "diagnosis_number"]].drop_duplicates()
    test_df = diag.merge(keys, on=["case_id", "diagnosis_number"], how="inner")
    test_df = test_df.drop_duplicates(subset=["case_id", "diagnosis_number"]).reset_index(drop=True)
    return test_df


def run_one(model: str, test_csv: Path, out_root: Path, timeout: int) -> tuple[Path, float]:
    safe = model.replace("/", "_").replace(":", "_")
    out_dir = out_root / f"compare_{safe}"
    cfg = LLMConfig(
        csv_path=str(test_csv),
        id_col="case_id",
        diag_num_col="diagnosis_number",
        text_col="diagnosis",
        labels_csv_path=config.LABELS_CSV,
        out_dir=str(out_dir),
        max_rows=None,
        llm_timeout=timeout,
        llm_model=model,
    )
    t0 = time.perf_counter()
    outputs = run_llm_scan(cfg)
    elapsed = time.perf_counter() - t0
    return Path(outputs.predictions_csv), elapsed


def aggregate(per_model: dict[str, tuple[Path, float]], out_dir: Path) -> None:
    frames = {
        name: pd.read_csv(path).assign(_key=lambda d: d["case_id"].astype(str) + "#" + d["diagnosis_number"].astype(str))
        for name, (path, _) in per_model.items()
    }
    keys = next(iter(frames.values()))["_key"].tolist()

    summary_rows = []
    for name, (_, elapsed) in per_model.items():
        df = frames[name]
        n = len(df)
        n_match = int((~df["method"].isin(["No Match", "Uncertain"])).sum())
        n_uncert = int((df["method"] == "Uncertain").sum())
        n_nomatch = int((df["method"] == "No Match").sum())
        per_call = elapsed / max(n, 1)
        summary_rows.append({
            "model": name,
            "rows": n,
            "matched": n_match,
            "uncertain": n_uncert,
            "no_match": n_nomatch,
            "match_pct": round(100 * n_match / max(n, 1), 1),
            "wall_seconds": round(elapsed, 1),
            "sec_per_row": round(per_call, 2),
            "unique_terms": df.loc[df["method"] != "No Match", "matched_term"].nunique(),
        })
    summary = pd.DataFrame(summary_rows).sort_values("model").reset_index(drop=True)

    names = list(frames.keys())
    agree = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            df_a = frames[a].set_index("_key").reindex(keys)
            df_b = frames[b].set_index("_key").reindex(keys)
            same = (df_a["matched_term"].fillna("") == df_b["matched_term"].fillna("")).sum()
            agree.loc[a, b] = round(100 * same / len(keys), 1)

    disagreements = []
    base = next(iter(names))
    for k in keys:
        picks = {n: frames[n].set_index("_key").loc[k, "matched_term"] for n in names}
        if len(set(picks.values())) > 1:
            diag_text = frames[base].set_index("_key").loc[k, "diagnosis"]
            disagreements.append({"key": k, "diagnosis": str(diag_text)[:80], **picks})

    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "summary.csv", index=False)
    agree.to_csv(out_dir / "agreement_pct.csv")
    pd.DataFrame(disagreements).to_csv(out_dir / "disagreements.csv", index=False)

    print("\n" + "=" * 80)
    print("PER-MODEL SUMMARY")
    print("=" * 80)
    print(summary.to_string(index=False))

    print("\n" + "=" * 80)
    print("PAIRWISE AGREEMENT % (matched_term identical, blanks count as identical)")
    print("=" * 80)
    print(agree.to_string())

    print("\n" + "=" * 80)
    print(f"DISAGREEMENTS ({len(disagreements)} of {len(keys)} rows)")
    print("=" * 80)
    if disagreements:
        sample = pd.DataFrame(disagreements).head(15)
        for _, row in sample.iterrows():
            print(f"\n[{row['key']}] {row['diagnosis']}")
            for n in names:
                print(f"   {n:<32} -> {row[n] or '(no match)'}")

    print(f"\nFull artifacts written to: {out_dir}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True, help="LM-Studio model IDs to compare.")
    parser.add_argument("--sample-size", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--llm-timeout", type=int, default=120)
    parser.add_argument(
        "--out-root",
        default=os.path.join(config.LLM_ANNOTATION_DIR, "compare"),
    )
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    test_csv = out_root / "tier3_sample.csv"

    test_df = build_sample(
        annotation_csv=Path(config.LLM_ANNOTATION_DIR) / "llm_annotation.csv",
        diagnoses_csv=Path(config.DIAGNOSES_CSV),
        n=args.sample_size,
        seed=args.seed,
    )
    test_df.to_csv(test_csv, index=False)
    print(f"Built tier-3 sample: {len(test_df)} rows -> {test_csv}\n")

    per_model: dict[str, tuple[Path, float]] = {}
    for m in args.models:
        print(f"\n>>> Running {m}")
        pred_path, elapsed = run_one(m, test_csv, out_root, args.llm_timeout)
        per_model[m] = (pred_path, elapsed)
        print(f"    {m}: {elapsed:.1f}s wall ({elapsed/max(len(test_df),1):.2f}s/row)")

    aggregate(per_model, out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
