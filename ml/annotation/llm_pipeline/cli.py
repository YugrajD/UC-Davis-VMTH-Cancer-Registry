"""Command-line interface for the LLM-assisted diagnosis annotation pipeline."""

import argparse

from .client import list_models
from .pipeline import LLMConfig, LLMOutputs, run_llm_scan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Map diagnosis text to Vet-ICD-O taxonomy labels using keyword + LLM matching."
    )
    parser.add_argument("--list-models", action="store_true", help="List available Ollama models and exit.")
    parser.add_argument("--compare-models", action="store_true", help="Run all available models on --max-rows rows and print a comparison.")
    parser.add_argument("--csv", default="ml/data/diagnoses.csv", help="Path to input diagnoses CSV.")
    parser.add_argument("--id-col", default="case_id", help="Case ID column name.")
    parser.add_argument("--diag-num-col", default="diagnosis_number", help="Diagnosis number column name.")
    parser.add_argument("--text-col", default="diagnosis", help="Diagnosis text column name.")
    parser.add_argument("--labels-csv", default="ml/ICD_labels/labels.csv", help="Path to Vet-ICD-O taxonomy CSV.")
    parser.add_argument("--out-dir", default="ml/output/annotation/llm", help="Output directory.")
    parser.add_argument("--max-rows", type=int, default=None, help="Cap on input rows (for testing).")
    parser.add_argument("--llm-timeout", type=int, default=60, help="Seconds to wait for each LLM call.")
    parser.add_argument("--model", default=None, help="Ollama model name to use (overrides OLLAMA_MODEL in .env).")
    parser.add_argument("--use-claude", action="store_true", help="Enable Tier 4: call Claude API for cases Tier 3 (Ollama) could not match.")
    parser.add_argument("--claude-timeout", type=int, default=30, help="Seconds to wait for each Claude API call (Tier 4).")
    return parser


def _run_compare(args: argparse.Namespace) -> int:
    import os, json
    models = [m["name"] for m in list_models()]
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
            print(m["name"])
        return 0

    if args.compare_models:
        return _run_compare(args)

    config = LLMConfig(
        csv_path=args.csv,
        id_col=args.id_col,
        diag_num_col=args.diag_num_col,
        text_col=args.text_col,
        labels_csv_path=args.labels_csv,
        out_dir=args.out_dir,
        max_rows=args.max_rows,
        llm_timeout=args.llm_timeout,
        llm_model=args.model,
        use_claude=args.use_claude,
        claude_timeout=args.claude_timeout,
    )
    outputs: LLMOutputs = run_llm_scan(config)
    print("Wrote:")
    print(f"  {outputs.predictions_csv}")
    print(f"  {outputs.summary_json}")
    return 0
