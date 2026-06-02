"""Command-line interface for LLM-assisted label annotation (authoritative method).

Runs the three-tier annotation cascade and, by default, follows it with the
ensemble verification cleanup pass. Pass --skip-cleanup to stop after writing
llm_annotation.csv (useful for quick tests or when cleanup will be re-run
separately via ml/annotation/llm_pipeline/run_annotation_cleanup.py).
"""

import argparse
import os

import config
from .cleanup import CleanupConfig, run_cleanup
from .client import list_models
from .pipeline import LLMConfig, LLMOutputs, run_llm_scan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Annotate diagnosis text with Vet-ICD-O cancer labels using a tiered cascade "
                    "(keyword -> fuzzy -> LM Studio LLM), then verify confirmed matches with a "
                    "two-model ensemble cleanup pass. This is the authoritative annotation source."
    )
    parser.add_argument("--list-models", action="store_true", help="List available LM Studio models and exit.")
    parser.add_argument("--compare-models", action="store_true", help="Run all available models on --max-rows rows and print a comparison.")
    parser.add_argument("--csv", default=config.DIAGNOSES_CSV, help="Path to input diagnoses CSV.")
    parser.add_argument("--id-col", default="case_id", help="Case ID column name.")
    parser.add_argument("--diag-num-col", default="diagnosis_number", help="Diagnosis number column name.")
    parser.add_argument("--text-col", default="diagnosis", help="Diagnosis text column name.")
    parser.add_argument("--labels-csv", default=config.LABELS_CSV, help="Path to Vet-ICD-O taxonomy CSV.")
    parser.add_argument("--out-dir", default=config.ANNOTATION_DIR, help="Output directory.")
    parser.add_argument("--max-rows", type=int, default=None, help="Cap on input rows (for testing).")
    parser.add_argument("--llm-timeout", type=int, default=60, help="Seconds to wait for each LLM call.")
    parser.add_argument("--model", default=None, help="LM Studio model name to use for Tier-3 (overrides LLM_MODEL in .env).")
    parser.add_argument("--skip-cleanup", action="store_true", help="Skip the ensemble verification cleanup pass.")
    parser.add_argument(
        "--cleanup-models",
        default="google/gemma-4-31b,qwen/qwen3.6-27b",
        help="Comma-separated verifier models for the cleanup pass.",
    )
    parser.add_argument(
        "--cleanup-tiebreaker",
        default=None,
        help="Optional third model used when the verifier pair disagrees.",
    )
    parser.add_argument("--cleanup-timeout", type=int, default=60, help="Seconds to wait per cleanup LLM call.")
    return parser


def _run_compare(args: argparse.Namespace) -> int:
    import os, json
    models = [m["id"] for m in list_models()]
    if not models:
        print("No models found.")
        return 1

    max_rows = args.max_rows or 200
    print(f"Comparing {len(models)} model(s) on {max_rows} rows: {models}\n")

    results: dict[str, dict] = {}
    for model in models:
        print(f"--- {model} ---")
        out_dir = os.path.join(args.out_dir, f"compare_{model.replace(':', '_').replace('/', '_')}")
        config = LLMConfig(
            csv_path=args.csv,
            id_col=args.id_col,
            diag_num_col=args.diag_num_col,
            text_col=args.text_col,
            labels_csv_path=args.labels_csv,
            out_dir=out_dir,
            max_rows=max_rows,
            llm_timeout=args.llm_timeout,
            llm_model=model,
        )
        outputs = run_llm_scan(config)
        with open(outputs.summary_json) as f:
            summary = json.load(f)
        results[model] = summary
        print()

    print("=" * 60)
    print(f"{'Model':<30} {'Match%':>7}  Method counts")
    print("-" * 60)
    for model, s in results.items():
        print(f"{model:<30} {s['match_rate_pct']:>6.1f}%  {s['method_counts']}")

    return 0


def main() -> int:
    args = build_parser().parse_args()

    if args.list_models:
        models = list_models()
        if not models:
            print("No models found.")
        for m in models:
            print(m["id"])
        return 0

    if args.compare_models:
        return _run_compare(args)

    llm_config = LLMConfig(
        csv_path=args.csv,
        id_col=args.id_col,
        diag_num_col=args.diag_num_col,
        text_col=args.text_col,
        labels_csv_path=args.labels_csv,
        out_dir=args.out_dir,
        max_rows=args.max_rows,
        llm_timeout=args.llm_timeout,
        llm_model=args.model,
    )
    outputs: LLMOutputs = run_llm_scan(llm_config)
    print("Wrote:")
    print(f"  {outputs.predictions_csv}")
    print(f"  {outputs.summary_json}")

    if args.skip_cleanup:
        return 0

    print("\n=== Cleanup pass (ensemble verification) ===")
    cleanup_config = CleanupConfig(
        input_csv=outputs.predictions_csv,
        output_csv=os.path.join(args.out_dir, "llm_annotation_cleaned.csv"),
        diff_csv=os.path.join(args.out_dir, "cleanup_diff.csv"),
        summary_json=os.path.join(args.out_dir, "cleanup_summary.json"),
        labels_csv_path=args.labels_csv,
        models=[m.strip() for m in args.cleanup_models.split(",") if m.strip()],
        tiebreaker_model=args.cleanup_tiebreaker,
        timeout=args.cleanup_timeout,
        max_rows=None,
    )
    run_cleanup(cleanup_config)
    return 0
