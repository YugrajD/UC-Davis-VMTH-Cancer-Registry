"""Train and retrain classifiers against verified cancer labels.

No env PYTHONPATH needed — this script adds ml/ to sys.path automatically.

Modes
-----
  train-groups          Train the multi-label GroupClassifier (Stage 2).
                        Predicts cancer group from a single report embedding.
                        Run after adapt-backbone; requires --train-cases.

  train-case-presence   Train the CasePresenceClassifier (Stage 1 gate).

  train-label-presence  Train per-group LabelPresenceClassifier checkpoints
                        (Stage 3a), one .pt per group.

  adapt-backbone        Fine-tune the embedding model (PetBERT) so that report
                        text and cancer-label text land closer together in
                        vector space. Run once; then cold-start downstream
                        classifiers.

Usage
-----
  python ml/scripts/run_training.py --mode train-groups --device xpu
  python ml/scripts/run_training.py --mode adapt-backbone --device xpu --local-only
  python ml/scripts/run_training.py --mode train-case-presence --device xpu
  python ml/scripts/run_training.py --mode train-label-presence --device xpu
"""

import argparse
import sys
from pathlib import Path

# Add ml/ to sys.path so all packages are importable without setting PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from model.constants import DEFAULT_HIDDEN_DIM
from training.group.build_training_data import build_training_data
from training.group.train import train as train_group
from training.contrastive.build_contrastive_dataset import build_contrastive_pairs
from training.contrastive.train_contrastive import train as train_contrastive
from training.binary.build_case_presence_dataset import build_dataset as build_case_presence_dataset
from training.binary.train_case_presence import train as train_case_presence
from training.label_presence.build_training_pairs import build_label_presence_pairs
from training.label_presence.train import train_label_presence
from utils.encoding import safe_filename


def _train_groups(args: argparse.Namespace) -> None:
    epochs = args.epochs if args.epochs is not None else 50
    group_lr = args.lr if args.lr is not None else 5e-5

    print("\n=== Step 2a: Build group classifier training data ===")
    excluded = [g.strip() for g in args.excluded_groups.split("|")] if args.excluded_groups else []
    build_training_data(
        cache_path=config.EMBEDDING_CACHE_NPZ,
        expectation_csv_path=args.annotation_csv,
        out_path=config.GROUP_TRAINING_DATA_NPZ,
        train_cases_txt=args.train_cases,
        uncommon_threshold=args.uncommon_threshold,
        uncommon_groups_out=config.UNCOMMON_GROUPS_TXT,
        excluded_groups=excluded,
    )
    print("\n=== Step 2b: Train group classifier ===")
    train_group(
        training_data_path=config.GROUP_TRAINING_DATA_NPZ,
        out_path=f"{config.CHECKPOINT_GROUP_DIR}/group_classifier_current.pt",
        epochs=epochs,
        lr=group_lr,
        hidden_dim=DEFAULT_HIDDEN_DIM,
        val_frac=0.2,
        threshold=0.3,
        device_arg=args.device,
        weight_decay=args.weight_decay,
        max_class_weight=args.max_class_weight,
        min_group_cases=10,
        max_group_cases=0,
        dropout=args.dropout,
        lr_schedule=args.lr_schedule,
    )


def _adapt_backbone(args: argparse.Namespace) -> None:
    epochs = args.epochs if args.epochs is not None else 3
    backbone_lr = args.lr if args.lr is not None else 2e-5

    if not args.skip_pair_build:
        print("\n=== Step 2a: Build (report, label) training pairs ===")
        build_contrastive_pairs(
            reports_csv=args.reports_csv,
            annotation_csv=args.annotation_csv,
            out_csv=args.pairs_csv,
            train_cases_txt=args.train_cases,
        )
    else:
        print("\n=== Step 2a: Skipped — reusing existing pairs CSV ===")

    print("\n=== Step 2b: Adapt embedding backbone ===")
    train_contrastive(
        pairs_csv=args.pairs_csv,
        out_dir=args.backbone_out_dir,
        model_name=args.model,
        epochs=epochs,
        batch_size=args.batch_size,
        lr=backbone_lr,
        temperature=args.temperature,
        max_length=args.max_length,
        device_arg=args.device,
        local_only=args.local_only,
    )

    print("\n=== Cold-start required ===")
    print("The embedding space has changed. Before retraining downstream classifiers:")
    print(f"  rm -f {config.EMBEDDING_CACHE_NPZ}")
    print(
        f"Then retrain in order: --mode train-case-presence, --mode train-groups, "
        f"--mode train-label-presence (each --model {args.backbone_out_dir} --local-only)."
    )


def _train_case_presence(args: argparse.Namespace) -> None:
    epochs = args.epochs if args.epochs is not None else 20

    print("\n=== Step 2a: Build case-level presence dataset ===")
    build_case_presence_dataset(
        annotation_csv=args.annotation_csv,
        embedding_cache=args.embedding_cache,
        out=config.CASE_PRESENCE_DATASET_NPZ,
        train_cases_txt=args.train_cases,
    )
    print("\n=== Step 2b: Train case presence classifier ===")
    train_case_presence(
        dataset_npz=config.CASE_PRESENCE_DATASET_NPZ,
        out_dir=config.CHECKPOINT_CASE_PRESENCE_DIR,
        epochs=epochs,
        device=args.device,
        recall_weight=args.case_presence_recall_weight,
        pos_weight=args.case_presence_pos_weight,
    )
    print(
        f"\nCheckpoint: {config.CASE_PRESENCE_CLASSIFIER_PT}\n"
        "Use with: --case-presence-classifier "
        f"{config.CASE_PRESENCE_CLASSIFIER_PT} "
        "--case-presence-threshold 0.5"
    )


def _train_label_presence(args: argparse.Namespace) -> None:
    print("\n=== Step 2: Train per-group LabelPresenceClassifier ===")

    uncommon_group_names: list[str] = []
    if args.label_presence_groups_from_taxonomy:
        from ICD_labels import load_labels_taxonomy
        group_names = sorted({tl.group for tl in load_labels_taxonomy(config.LABELS_CSV)})
        print(f"Taxonomy groups ({len(group_names)}): {group_names}")
    else:
        from model.group_classifier import GroupClassifier
        _, group_names = GroupClassifier.load(args.group_classifier_path)
        print(f"GroupClassifier groups ({len(group_names)}): {group_names}")

        uncommon_path = Path(config.UNCOMMON_GROUPS_TXT)
        if uncommon_path.exists():
            uncommon_group_names = [
                l.strip() for l in uncommon_path.read_text(encoding="utf-8").splitlines()
                if l.strip()
            ]
            print(f"Uncommon groups ({len(uncommon_group_names)}): {uncommon_group_names}")

    out_dir = Path(args.label_presence_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    training_pairs_dir = Path(config.OUTPUT_TRAINING_DIR) / "label_presence"
    training_pairs_dir.mkdir(parents=True, exist_ok=True)

    allowed_groups: set[str] = set()
    if args.label_presence_groups:
        allowed_groups = {g.strip() for g in args.label_presence_groups.split("|")}
        print(f"Filtering to groups: {sorted(allowed_groups)}")

    trained, skipped = 0, 0

    common_groups = [g for g in group_names if g != "Uncommon"]
    if allowed_groups:
        common_groups = [g for g in common_groups if g in allowed_groups]
    for group_name in common_groups:
        print(f"\n--- Group: {group_name!r} ---")
        safe = safe_filename(group_name)
        pairs_csv = str(training_pairs_dir / f"{safe}_pairs.csv")
        out_pt = str(out_dir / f"{safe}.pt")

        n_rows = build_label_presence_pairs(
            annotation_csv=args.annotation_csv,
            labels_csv=config.LABELS_CSV,
            out_csv=pairs_csv,
            group_name=group_name,
            train_cases_txt=args.train_cases,
            within_group_negs_per_pos=args.label_presence_negs_per_pos,
        )
        if n_rows == 0:
            skipped += 1
            continue

        score = train_label_presence(
            pairs_csv=pairs_csv,
            embedding_cache=args.embedding_cache,
            out_path=out_pt,
            epochs=args.label_presence_epochs,
            recall_weight=args.label_presence_recall_weight,
            dropout=args.label_presence_dropout,
            weight_decay=args.label_presence_weight_decay,
            device=args.device,
            model_name=args.model,
            labels_csv=config.LABELS_CSV,
            report_csv=config.REPORTS_CSV,
            n_cols=args.label_presence_n_cols,
            col_pair_mode=args.label_presence_col_pair_mode,
            col_combine=args.label_presence_col_combine,
        )
        if score > 0:
            trained += 1
        else:
            skipped += 1

    if uncommon_group_names and (not allowed_groups or "Uncommon" in allowed_groups):
        print(f"\n--- Group: 'Uncommon' (union of {len(uncommon_group_names)} groups) ---")
        pairs_csv = str(training_pairs_dir / "uncommon_pairs.csv")
        out_pt = str(out_dir / "uncommon.pt")

        n_rows = build_label_presence_pairs(
            annotation_csv=args.annotation_csv,
            labels_csv=config.LABELS_CSV,
            out_csv=pairs_csv,
            group_name="Uncommon",
            uncommon_group_names=uncommon_group_names,
            train_cases_txt=args.train_cases,
            within_group_negs_per_pos=args.label_presence_negs_per_pos,
        )
        if n_rows > 0:
            score = train_label_presence(
                pairs_csv=pairs_csv,
                embedding_cache=args.embedding_cache,
                out_path=out_pt,
                epochs=args.label_presence_epochs,
                recall_weight=args.label_presence_recall_weight,
                dropout=args.label_presence_dropout,
                weight_decay=args.label_presence_weight_decay,
                device=args.device,
                model_name=args.model,
                labels_csv=config.LABELS_CSV,
                report_csv=config.REPORTS_CSV,
                n_cols=args.label_presence_n_cols,
                col_pair_mode=args.label_presence_col_pair_mode,
                col_combine=args.label_presence_col_combine,
            )
            if score > 0:
                trained += 1
            else:
                skipped += 1
        else:
            skipped += 1

    print(
        f"\n=== train-label-presence complete ===\n"
        f"  Trained: {trained}  Skipped: {skipped}\n"
        f"  Checkpoints: {out_dir}\n"
        f"Use with: --label-presence-classifier-dir {out_dir}"
    )


_MODE_DISPATCH = {
    "train-groups":         _train_groups,
    "adapt-backbone":       _adapt_backbone,
    "train-case-presence":  _train_case_presence,
    "train-label-presence": _train_label_presence,
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train classifiers to predict cancer labels from veterinary pathology reports.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["train-groups", "adapt-backbone", "train-case-presence", "train-label-presence"],
        default="train-groups",
        help=(
            "What to train: "
            "train-groups (Stage 2 GroupClassifier — default), "
            "adapt-backbone (fine-tune embedding model), "
            "train-case-presence (Stage 1 gate), "
            "train-label-presence (Stage 3a per-group label scorer)."
        ),
    )
    # ------------------------------------------------------------------
    # Shared args
    # ------------------------------------------------------------------
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda", "mps", "xpu"],
                        help="Compute device (default: auto)")
    parser.add_argument("--local-only", action="store_true",
                        help="Use only locally cached model files (no HuggingFace download)")
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Training epochs. Defaults: adapt-backbone=3, train-groups=50.",
    )
    parser.add_argument(
        "--annotation-csv", default=config.ANNOTATION_CSV,
        help="Verified label annotations used as training supervision and evaluation "
             "ground truth. Accepts any annotation CSV in the shared format. "
             f"(default: {config.ANNOTATION_CSV})",
    )
    parser.add_argument("--train-cases", default="",
                        help="[all modes] Path to train_cases.txt. "
                             "When provided, only train cases are used during training. "
                             "Generate with ml/training/data/create_split.py.")
    parser.add_argument("--max-class-weight", type=float, default=50.0,
                        help="[train-groups] Cap per-group BCE pos_weight at this value (default: 50). "
                             "Prevents rare-group class weights (up to 3500x) from dominating training.")
    parser.add_argument("--weight-decay", type=float, default=1e-3,
                        help="[train-groups] Adam weight decay / L2 regularization (default: 1e-3).")
    parser.add_argument("--uncommon-threshold", type=int, default=200,
                        help="[train-groups] Groups with fewer training cases than this are merged "
                             "into a single 'Uncommon' output class (default: 200). Set 0 to disable.")
    parser.add_argument("--excluded-groups", default="Neoplasms, NOS",
                        help="[train-groups] Pipe-separated group names forced into 'Uncommon' "
                             "regardless of case count (use | not comma). Default: 'Neoplasms, NOS'.")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="[train-groups] MLP dropout probability (default: 0.3). "
                             "Try 0.1 or 0.05 — Phase 26 train/val gap suggests over-regularisation.")
    parser.add_argument("--lr-schedule", default="none", choices=["none", "cosine"],
                        help="[train-groups] LR schedule (default: none). "
                             "'cosine' uses CosineAnnealingWarmRestarts(T_0=100).")

    # ------------------------------------------------------------------
    # train-case-presence args
    # ------------------------------------------------------------------
    parser.add_argument(
        "--embedding-cache",
        default=config.EMBEDDING_CACHE_NPZ,
        help="[train-case-presence] Path to embedding cache NPZ "
             f"(default: {config.EMBEDDING_CACHE_NPZ})",
    )
    parser.add_argument(
        "--case-presence-recall-weight",
        type=float,
        default=0.7,
        help="[train-case-presence] Recall weight for checkpoint selection (default: 0.7). "
             "Higher = prefer fewer missed cancer cases over fewer false positives.",
    )
    parser.add_argument(
        "--case-presence-pos-weight",
        type=float,
        default=1.0,
        help="[train-case-presence] BCEWithLogitsLoss pos_weight (default: 1.0). "
             "Increase if cancer-positive cases are heavily outnumbered.",
    )

    # ------------------------------------------------------------------
    # train-label-presence args
    # ------------------------------------------------------------------
    parser.add_argument(
        "--group-classifier-path",
        default=f"{config.CHECKPOINT_GROUP_DIR}/group_classifier_best.pt",
        help="[train-label-presence] Path to trained GroupClassifier checkpoint. "
             "Used to read group_names and identify common vs. uncommon groups. "
             f"(default: {config.CHECKPOINT_GROUP_DIR}/group_classifier_best.pt)",
    )
    parser.add_argument(
        "--label-presence-out-dir",
        default=config.CHECKPOINT_LABEL_PRESENCE_DIR,
        help="[train-label-presence] Directory to save per-group LabelPresenceClassifier checkpoints. "
             f"(default: {config.CHECKPOINT_LABEL_PRESENCE_DIR})",
    )
    parser.add_argument(
        "--label-presence-epochs",
        type=int,
        default=25,
        help="[train-label-presence] Training epochs per group (default: 25).",
    )
    parser.add_argument(
        "--label-presence-negs-per-pos",
        type=int,
        default=5,
        help="[train-label-presence] Within-group negatives per positive pair (default: 5).",
    )
    parser.add_argument(
        "--label-presence-recall-weight",
        type=float,
        default=0.5,
        help="[train-label-presence] Recall weight for checkpoint selection (default: 0.5 = F1). "
             "Score = (1-rw)*P + rw*R.",
    )
    parser.add_argument(
        "--label-presence-dropout",
        type=float,
        default=0.3,
        help="[train-label-presence] MLP dropout (default: 0.3 = Phase 28).",
    )
    parser.add_argument(
        "--label-presence-weight-decay",
        type=float,
        default=1e-4,
        help="[train-label-presence] AdamW weight decay (default: 1e-4 = Phase 28).",
    )
    parser.add_argument(
        "--label-presence-groups",
        default="",
        help="[train-label-presence] Pipe-separated group names to train. "
             "Empty = train all groups. Include 'Uncommon' to retrain the Uncommon bucket. "
             "Example: 'Thymic epithelial neoplasms|Myxomatous neoplasms|Uncommon'",
    )
    parser.add_argument(
        "--label-presence-groups-from-taxonomy",
        action="store_true",
        help="[train-label-presence] Source group names directly from the taxonomy CSV "
             "(all 52 groups) instead of from the GroupClassifier checkpoint (24 common + "
             "'Uncommon'). Use with --label-presence-groups to train heads for groups "
             "that the GroupClassifier merged into 'Uncommon' or excluded.",
    )
    parser.add_argument(
        "--label-presence-n-cols",
        type=int,
        default=3,
        help="[train-label-presence] Number of per-row sections in the report "
             "embedding (default: 3 for concat_3). Each section's 768-dim view "
             "scores the label independently when col-pair-mode is on.",
    )
    parser.add_argument(
        "--label-presence-col-pair-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="[train-label-presence] Per-section pair architecture (default: True). "
             "Off = legacy single-MLP concat [report | label] path.",
    )
    parser.add_argument(
        "--label-presence-col-combine",
        choices=["max", "mean", "learned"],
        default="learned",
        help="[train-label-presence] How per-section logits combine when "
             "col-pair-mode is on (default: learned - Linear(n_cols -> 1)).",
    )

    # ------------------------------------------------------------------
    # adapt-backbone args
    # ------------------------------------------------------------------
    parser.add_argument("--reports-csv", default=config.REPORTS_CSV,
                        help=f"[adapt-backbone] Report text CSV (default: {config.REPORTS_CSV})")
    parser.add_argument("--pairs-csv", default=config.CONTRASTIVE_PAIRS_CSV,
                        help="[adapt-backbone] Output/input path for (report, label) training pairs CSV")
    parser.add_argument("--backbone-out-dir", default=config.CHECKPOINT_CONTRASTIVE_DIR,
                        help="[adapt-backbone] Directory for the adapted model checkpoint")
    parser.add_argument(
        "--model", default="SAVSNET/PetBERT",
        help="HuggingFace model name or local checkpoint path. "
             "[adapt-backbone] starting weights; "
             "[train-label-presence] embedding backbone for label/report encoding. "
             "Default: SAVSNET/PetBERT",
    )
    parser.add_argument("--batch-size", type=int, default=32,
                        help="[adapt-backbone] Batch size / number of in-batch negatives (default: 32)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Peak learning rate. "
                             "[adapt-backbone] default: 2e-5; [train-groups] default: 5e-5")
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="[adapt-backbone] Contrastive loss temperature (default: 0.07)")
    parser.add_argument("--max-length", type=int, default=256,
                        help="[adapt-backbone] Max BERT token length (default: 256)")
    parser.add_argument("--skip-pair-build", action="store_true",
                        help="[adapt-backbone] Skip building training pairs (reuse existing pairs CSV)")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Step 1: Ensure verified labels are available
    # ------------------------------------------------------------------
    annotation_path = Path(args.annotation_csv)
    if annotation_path.exists():
        print(f"\n=== Step 1: Verified labels found — skipping annotation ({args.annotation_csv}) ===")
    else:
        print(f"\nError: annotation file not found: {args.annotation_csv}")
        print("Run annotation first:")
        print("  python ml/scripts/run_annotation.py")
        return 1

    # ------------------------------------------------------------------
    # Step 2: Mode-specific training
    # ------------------------------------------------------------------
    _MODE_DISPATCH[args.mode](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
