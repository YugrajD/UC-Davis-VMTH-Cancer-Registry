"""Diagnosis-to-taxonomy mapping: run the cascade, import the legacy silver
generation, and print coverage stats.

Replaces run_annotation.py, run_annotation_cleanup.py and run_data_analysis.py.

Usage:
  python ml/next/scripts/map_diagnoses.py run --id silver-1 [--no-llm] [--skip-cleanup]
  python ml/next/scripts/map_diagnoses.py import-legacy
  python ml/next/scripts/map_diagnoses.py stats --silver silver-0-legacy [--no-plots] [--allow-no-llm]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from diagnosis_mapping import silver, stats
from diagnosis_mapping.silver import NoLLMGenerationError
from generations.manifest import ManifestError

# Same default verifier pair as the old pipeline (see ml/documentation/label-annotation.md):
# selected from a 6-model bake-off on 26 Tier-3 rows for highest adjudicated correctness
# and architectural diversity, so unanimous votes carry signal.
DEFAULT_CLEANUP_MODELS = ["google/gemma-4-31b", "qwen/qwen3.6-27b"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run the cascade (and optionally cleanup) into a new silver generation.")
    run.add_argument("--id", required=True, help="New silver_id to write.")
    run.add_argument("--diagnoses-csv", default=None, help="Defaults to config.DIAGNOSES_CSV.")
    run.add_argument("--labels-csv", default=None, help="Defaults to config.LABELS_CSV.")
    run.add_argument("--no-llm", action="store_true", help="Skip Tier-3 LLM calls (every eligible row is recorded as declined).")
    run.add_argument("--model", default=None, help="Tier-3 LLM model name (overrides LLM_MODEL in .env).")
    run.add_argument("--llm-timeout", type=int, default=60, help="Seconds to wait for each Tier-3 LLM call.")
    run.add_argument("--skip-cleanup", action="store_true", help="Skip the ensemble verification cleanup pass (it runs by default).")
    run.add_argument("--cleanup-models", default=",".join(DEFAULT_CLEANUP_MODELS), help="Comma-separated verifier models for the cleanup pass.")
    run.add_argument("--cleanup-tiebreaker", default=None, help="Optional third model used when the verifier pair disagrees.")
    run.add_argument("--cleanup-timeout", type=int, default=60, help="Seconds to wait per cleanup LLM call.")

    sub.add_parser("import-legacy", help="Import the pre-versioning cleaned annotation.csv as silver-0-legacy.")

    stats_cmd = sub.add_parser("stats", help="Compute coverage statistics for a silver generation.")
    stats_cmd.add_argument("--silver", required=True, help="silver_id to analyse.")
    stats_cmd.add_argument("--out-dir", default=None, help="Defaults to config.DIAGNOSIS_MAPPING_STATS_DIR/<silver_id>.")
    stats_cmd.add_argument("--no-plots", action="store_true", help="Skip PNG plot generation.")
    stats_cmd.add_argument("--allow-no-llm", action="store_true", help="Allow analysing a --no-llm generation (see silver.load_silver).")

    args = parser.parse_args()

    if args.command == "run":
        if args.no_llm:
            print(
                "WARNING: --no-llm records every Tier-3-eligible row as a declined LLM match "
                '(tier3_llm/"No Match"), which the coding rule treats as decisive non-cancer. '
                "This generation will be refused by silver.load_silver() (and by consumers such "
                "as the coding rule and the Tier-3 sampler) unless allow_no_llm=True is passed "
                "explicitly. Do not adopt it as an authoritative silver source."
            )
        manifest = silver.run(
            args.id,
            diagnoses_csv=args.diagnoses_csv,
            labels_csv=args.labels_csv,
            llm_enabled=not args.no_llm,
            llm_model=args.model,
            llm_timeout=args.llm_timeout,
            cleanup_enabled=not args.skip_cleanup,
            cleanup_models=[m.strip() for m in args.cleanup_models.split(",") if m.strip()],
            cleanup_tiebreaker=args.cleanup_tiebreaker,
            cleanup_timeout=args.cleanup_timeout,
        )
        print(f"{manifest['silver_id']}: {manifest['decision_stage_counts']}")
    elif args.command == "import-legacy":
        manifest = silver.import_legacy()
        print(f"{manifest['silver_id']}: {manifest['decision_stage_counts']}")
    else:
        try:
            df = silver.load_silver(args.silver, allow_no_llm=args.allow_no_llm)
        except (ManifestError, NoLLMGenerationError) as violation:
            print(f"FAIL: {violation}")
            return 1
        out_dir = args.out_dir or str(config.DIAGNOSIS_MAPPING_STATS_DIR / args.silver)
        stats.run_analysis(df, labels_csv=str(config.LABELS_CSV), out_dir=out_dir, make_plots=not args.no_plots)
        print(f"\nWrote artifacts to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
