"""Command-line interface for running the PetBERT scan pipeline."""

import argparse

import config
from .pipeline import run_scan
from .types import ScanConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan a reportText CSV with PetBERT and produce categorizations (and optional nearest neighbors)."
    )
    parser.add_argument("--csv", default=config.REPORTS_CSV, help="Path to input CSV")
    parser.add_argument("--id-col", default="case_id", help="ID column name")
    parser.add_argument("--model", default="SAVSNET/PetBERT", help="HF model name or local path")
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Use only local cached model files (no network calls).",
    )
    parser.add_argument("--out-dir", default=config.PETBERT_SCAN_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap on rows")
    parser.add_argument("--batch-size", type=int, default=16, help="Embedding batch size")
    parser.add_argument("--max-length", type=int, default=512, help="Tokenizer max_length")
    parser.add_argument("--neighbors-k", type=int, default=3, help="Neighbors per row")
    parser.add_argument(
        "--task",
        default="categorize",
        choices=["categorize", "neighbors", "both"],
        help="Run categorization only, neighbors only, or both.",
    )
    parser.add_argument(
        "--embedding-min-sim",
        type=float,
        default=0.6,
        help="Minimum embedding similarity to accept embedding fallback category.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps", "xpu"],
        help="Compute device",
    )
    parser.add_argument(
        "--labels-csv",
        default=config.LABELS_CSV,
        help="Path to labels taxonomy CSV.",
    )
    parser.add_argument(
        "--embedding-cache",
        default=None,
        help=(
            "Path to an embedding cache npz file. "
            "If the cache exists and is valid, PetBERT embedding is skipped entirely. "
            "If it doesn't exist yet, embeddings are computed and saved here for future runs."
        ),
    )
    parser.add_argument(
        "--group-classifier",
        default=None,
        help=(
            "Path to a trained GroupClassifier checkpoint (.pt). "
            "Enables 3-stage categorization: "
            "(1) CasePresenceClassifier gates non-cancer cases (if --case-presence-classifier is set), "
            "(2) GroupClassifier predicts which cancer group(s) a case belongs to, "
            "(3) ICD-O behavior keyword matching selects the best term within each group."
        ),
    )
    parser.add_argument(
        "--group-classifier-threshold",
        type=float,
        default=0.3,
        help=(
            "Probability threshold for GroupClassifier group selection (default: 0.3). "
            "Lower = more predictions (higher recall, lower precision). "
            "Only used when --group-classifier is set."
        ),
    )
    parser.add_argument(
        "--case-presence-classifier",
        default=None,
        help=(
            "Path to a trained CasePresenceClassifier checkpoint (.pt). "
            "When set alongside --group-classifier, acts as a first-stage gate: "
            "cases whose cancer probability is below --case-presence-threshold are "
            "predicted Uncategorized without reaching the GroupClassifier."
        ),
    )
    parser.add_argument(
        "--case-presence-threshold",
        type=float,
        default=0.5,
        help=(
            "Probability threshold for CasePresenceClassifier gate (default: 0.5). "
            "Lower = more cases pass (higher recall, higher FP). "
            "Only used when --case-presence-classifier is set."
        ),
    )
    parser.add_argument(
        "--no-group-classifier-fallback-to-argmax",
        dest="group_classifier_fallback_to_argmax",
        action="store_false",
        default=True,
        help=(
            "Disable argmax fallback: cases where no group clears the threshold are "
            "predicted 'Unidentified Cancer' instead of using the top-1 group. "
            "Default: fallback is enabled."
        ),
    )
    parser.add_argument(
        "--label-presence-classifier-dir",
        default=None,
        help=(
            "Directory of per-group LabelPresenceClassifier checkpoints (one .pt per group). "
            "When set, enables Stage 3a: per-group within-group label scoring. "
            "Pass an empty string to disable and fall back to KW correction directly."
        ),
    )
    parser.add_argument(
        "--label-presence-threshold",
        type=float,
        default=0.5,
        help=(
            "Probability threshold for LabelPresenceClassifier within-group label selection "
            "(default: 0.5). Only used when --label-presence-classifier-dir is set. "
            "Acts as the fallback when --label-presence-thresholds-json is set but a group "
            "is missing from the mapping."
        ),
    )
    parser.add_argument(
        "--label-presence-thresholds-json",
        default=None,
        help=(
            "Optional path to a JSON file mapping group_name -> threshold (float). "
            "When set, overrides --label-presence-threshold on a per-group basis; "
            "groups not in the mapping fall back to --label-presence-threshold."
        ),
    )
    parser.add_argument(
        "--tail-max-predictions",
        type=int,
        default=2,
        help=(
            "Cap the number of group predictions emitted per case (default: 2). "
            "Set to 1 to keep only the top group; 2 keeps the top group plus one "
            "tail prediction that also clears --tail-max-group-prob-gap. Lowers CO "
            "at the cost of recall on multi-label cases. Calibrated 2026-05-11 "
            "(see sweep_tail_gate.py): K=2 with gap=0.08 gives +0.9 pp G+S on test."
        ),
    )
    parser.add_argument(
        "--tail-max-group-prob-gap",
        type=float,
        default=0.08,
        help=(
            "Drop tail group predictions whose probability is below "
            "(top_group_prob - this_value). Default 0.08 trims wrong-group tail "
            "predictions whose Stage-2 score is far below the top group. Set to "
            "1.0 to disable. Calibrated 2026-05-11 on the held-out test set — see "
            "ml/scripts/sweep_tail_gate.py for the trade-off curve."
        ),
    )
    parser.add_argument(
        "--rerank-stage3",
        action="store_true",
        default=False,
        help=(
            "Re-rank Stage-3 winners across the top-K surviving groups by "
            "(lp_score - lp_threshold) * group_prob (margin-times-group-prob). "
            "Default off: labels stay in group-prob order. Only meaningful when "
            "--tail-max-predictions > 1."
        ),
    )
    parser.add_argument(
        "--embed-only",
        action="store_true",
        default=False,
        help=(
            "Stop after Step 3 (cache populated). Useful for building the cache "
            "before training downstream classifiers without running classification."
        ),
    )
    return parser


def build_config(args: argparse.Namespace) -> ScanConfig:
    label_presence_dir = getattr(args, "label_presence_classifier_dir", None)
    if label_presence_dir == "":
        label_presence_dir = None
    embed_only = bool(getattr(args, "embed_only", False))
    return ScanConfig(
        csv_path=args.csv,
        id_col=args.id_col,
        model_name=args.model,
        local_only=args.local_only,
        out_dir=args.out_dir,
        max_rows=args.max_rows,
        batch_size=args.batch_size,
        max_length=args.max_length,
        neighbors_k=args.neighbors_k,
        task=args.task,
        embedding_min_sim=args.embedding_min_sim,
        device=args.device,
        labels_csv_path=args.labels_csv,
        embedding_cache_path=args.embedding_cache,
        group_classifier_path=args.group_classifier,
        group_classifier_threshold=args.group_classifier_threshold,
        group_classifier_fallback_to_argmax=args.group_classifier_fallback_to_argmax,
        case_presence_classifier_path=args.case_presence_classifier,
        case_presence_threshold=args.case_presence_threshold,
        label_presence_classifier_dir=label_presence_dir,
        label_presence_threshold=args.label_presence_threshold,
        label_presence_thresholds_json=args.label_presence_thresholds_json,
        tail_max_predictions=args.tail_max_predictions,
        tail_max_group_prob_gap=args.tail_max_group_prob_gap,
        rerank_stage3=args.rerank_stage3,
        embed_only=embed_only,
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    outputs = run_scan(build_config(args))

    print("Wrote:")
    print(outputs.predictions_csv)
    print(outputs.provenance_csv)
    print(outputs.similarity_csv)
    print(outputs.visualization_csv)
    if outputs.neighbors_csv is not None:
        print(outputs.neighbors_csv)
    print(outputs.npz)
    print(outputs.summary_json)
    return 0
