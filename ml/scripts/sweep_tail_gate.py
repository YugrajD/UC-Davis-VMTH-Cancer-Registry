"""Sweep (tail_max_predictions, tail_max_group_prob_gap) on the held-out test set.

Runs production + evaluation for each config and prints a comparison table.
Embeddings are cached so each iteration is fast (~2 min).

Usage:
  python ml/scripts/sweep_tail_gate.py
"""

import csv
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PY = str(REPO_ROOT / "ml" / ".venv" / "Scripts" / "python.exe")
TEST_CASES = str(REPO_ROOT / "ml" / "output" / "splits" / "test_cases.txt")
ANNOTATION = str(REPO_ROOT / "ml" / "output" / "annotation" / "llm" / "llm_annotation.csv")
SWEEP_ROOT_PROD = REPO_ROOT / "ml" / "output" / "production" / "tailgate_sweep"
SWEEP_ROOT_EVAL = REPO_ROOT / "ml" / "output" / "evaluation" / "tailgate_sweep"
RUN_PROD = str(REPO_ROOT / "ml" / "scripts" / "run_production.py")
RUN_EVAL = str(REPO_ROOT / "ml" / "scripts" / "run_evaluation.py")

CONFIGS: list[tuple[int, float]] = [
    (1, 1.00),  # cap=1, gap irrelevant
    (2, 0.02),
    (2, 0.05),
    (2, 0.08),  # production default (calibrated 2026-05-11)
    (2, 0.10),
    (3, 0.10),
    (5, 1.00),  # no gate — baseline
]


def run(K: int, gap: float) -> dict:
    tag = f"K{K}_g{gap:.2f}".replace(".", "p")
    prod_dir = SWEEP_ROOT_PROD / tag
    eval_dir = SWEEP_ROOT_EVAL / tag

    # 1. Production with the gate active
    subprocess.run([
        PY, RUN_PROD,
        "--group-classifier-threshold", "0.85",
        "--tail-max-predictions", str(K),
        "--tail-max-group-prob-gap", f"{gap:.3f}",
        "--out-dir", str(prod_dir),
        "--local-only",
    ], check=True, cwd=str(REPO_ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    # 2. Evaluation against held-out test set
    subprocess.run([
        PY, RUN_EVAL,
        "--prediction-csv", str(prod_dir / "petbert_predictions.csv"),
        "--annotation-csv", ANNOTATION,
        "--test-cases", TEST_CASES,
        "--out-dir", str(eval_dir),
        "--label", f"sweep {tag}",
    ], check=True, cwd=str(REPO_ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    # 3. Parse evaluation_summary.csv → OVERALL row
    summary_csv = eval_dir / "evaluation_summary.csv"
    with open(summary_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["scope"] == "OVERALL":
                return {
                    "K": K,
                    "gap": gap,
                    "total": int(row["total"]),
                    "good": float(row["good_pct"]),
                    "slight": float(row["slightly_off_pct"]),
                    "co": float(row["completely_off_pct"]),
                    "fp": float(row["false_positive_pct"]),
                    "fn": float(row["false_negative_pct"]),
                }
    raise RuntimeError(f"No OVERALL row in {summary_csv}")


def main() -> int:
    SWEEP_ROOT_PROD.mkdir(parents=True, exist_ok=True)
    SWEEP_ROOT_EVAL.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for K, gap in CONFIGS:
        print(f"=== K={K}, gap={gap:.2f} ===", flush=True)
        rows.append(run(K, gap))

    print()
    print(f"{'K':>3} {'gap':>5} {'total':>6} {'G%':>6} {'S%':>6} {'CO%':>6} {'FP%':>6} {'FN%':>6} {'G+S%':>6}")
    print("-" * 60)
    for r in rows:
        gs = r["good"] + r["slight"]
        print(
            f"{r['K']:>3} {r['gap']:>5.2f} {r['total']:>6} "
            f"{r['good']:>6.1f} {r['slight']:>6.1f} {r['co']:>6.1f} "
            f"{r['fp']:>6.1f} {r['fn']:>6.1f} {gs:>6.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
