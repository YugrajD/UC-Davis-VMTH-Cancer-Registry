"""MLM domain adaptation — fine-tune PetBERT on cancer registry reports.

Fine-tunes PetBERT using the masked language modeling (MLM) objective on the
unlabeled report corpus.  No labels are required.  The resulting checkpoint
produces improved embeddings for all downstream tasks (PresenceClassifier,
GroupClassifier) by adapting the model weights to UC Davis VMTH oncology
terminology.

The fine-tuned model is a drop-in replacement for the base PetBERT weights:
pass its path as --model to run_pipeline.py or run_training.py binary/group modes.

IMPORTANT — cold start required after fine-tuning:
  The embedding cache and CO bank are keyed to the old model's embedding space.
  Delete both before running any downstream training cycle with the new model:
    rm ml/data/embedding_cache.npz
    rm ml/output/evaluation/evaluation_co_bank.csv

Usage (via run_training.py):
  python ml/scripts/run_training.py --mode finetune --epochs 3 --local-only

Direct usage:
  python ml/training/finetune/train.py --epochs 3 --local-only
  python ml/training/finetune/train.py --include-diagnoses --epochs 5 --device mps
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from training.finetune.dataset import ReportMLMDataset

_DEFAULT_REPORT_CSV = "ml/data/report.csv"
_DEFAULT_DIAGNOSES_CSV = "ml/data/diagnoses.csv"
_DEFAULT_TEXT_COLS = ["HISTOPATHOLOGICAL SUMMARY", "FINAL COMMENT", "ANCILLARY TESTS"]
_DEFAULT_DIAGNOSES_COL = ["diagnosis"]
_DEFAULT_OUTPUT_DIR = "ml/model/checkpoints/petbert_mlm"
_DEFAULT_MODEL = "SAVSNET/PetBERT"

_COLD_START_WARNING = """
╔══════════════════════════════════════════════════════════════════╗
║  COLD START REQUIRED                                             ║
║                                                                  ║
║  The embedding space has changed. Before running a downstream    ║
║  training cycle with this model, delete the stale cache and CO   ║
║  bank — they are anchored to the old model's embedding space:    ║
║                                                                  ║
║    rm ml/data/embedding_cache.npz                                ║
║    rm ml/output/evaluation/evaluation_co_bank.csv                ║
╚══════════════════════════════════════════════════════════════════╝
"""


def train(
    *,
    report_csv: str = _DEFAULT_REPORT_CSV,
    diagnoses_csv: str = _DEFAULT_DIAGNOSES_CSV,
    text_cols: list[str] = _DEFAULT_TEXT_COLS,
    model_name: str = _DEFAULT_MODEL,
    output_dir: str = _DEFAULT_OUTPUT_DIR,
    epochs: int = 3,
    lr: float = 2e-5,
    batch_size: int = 16,
    mlm_probability: float = 0.15,
    val_frac: float = 0.1,
    include_diagnoses: bool = False,
    local_only: bool = False,
    device_arg: str = "auto",
) -> None:
    """Fine-tune PetBERT with MLM on the report corpus.

    TRAINING FLOW
    =============
    report.csv + (optionally) diagnoses.csv
        │
        ▼ ReportMLMDataset (tokenize, no padding)
        │
        ▼ 90/10 train/val split (random)
        │
        ▼ DataCollatorForLanguageModeling (dynamic 15% masking per batch)
        │
        ▼ HuggingFace Trainer (AutoModelForMaskedLM)
        │  eval_strategy="epoch", save best by eval_loss
        │
        ▼ output_dir/  (best checkpoint + tokenizer)
    """
    # --- 1. Load tokenizer and model ---
    print(f"[finetune] Loading tokenizer and model: {model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_only)
    model = AutoModelForMaskedLM.from_pretrained(
        model_name, local_files_only=local_only
    )

    # --- 2. Build dataset ---
    sources: list[tuple[str, list[str]]] = [(report_csv, text_cols)]
    if include_diagnoses:
        sources.append((diagnoses_csv, _DEFAULT_DIAGNOSES_COL))

    full_dataset = ReportMLMDataset(tokenizer, sources=sources)

    # --- 3. Train / val split ---
    n = len(full_dataset)
    n_val = max(1, math.floor(n * val_frac))
    n_train = n - n_val

    import torch

    train_ds, val_ds = torch.utils.data.random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    print(
        f"[finetune] Split: {n_train:,} train / {n_val:,} val  (val_frac={val_frac})",
        flush=True,
    )

    # --- 4. Data collator (handles masking + padding per batch) ---
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability,
    )

    # --- 5. Training arguments ---
    # no_cuda / use_mps_device flags are inferred from device_arg so that
    # TrainingArguments still receives the standard keyword args it expects.
    training_kwargs: dict = {}
    if device_arg == "cpu":
        training_kwargs["no_cuda"] = True
        training_kwargs["use_mps_device"] = False
    elif device_arg == "mps":
        training_kwargs["no_cuda"] = True
        training_kwargs["use_mps_device"] = True
    # For "auto", "cuda", "xpu" — let Trainer detect.

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        report_to="none",  # no wandb / tensorboard
        **training_kwargs,
    )

    # --- 6. Train ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    print(
        f"[finetune] Starting MLM training: {epochs} epoch(s), "
        f"batch_size={batch_size}, lr={lr}, mlm_prob={mlm_probability}",
        flush=True,
    )
    trainer.train()

    # --- 7. Save best checkpoint + tokenizer to output_dir root ---
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[finetune] Checkpoint saved to: {output_dir}", flush=True)

    print(_COLD_START_WARNING, flush=True)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MLM domain adaptation — fine-tune PetBERT on cancer registry reports."
    )
    p.add_argument(
        "--csv",
        default=_DEFAULT_REPORT_CSV,
        help=f"Report CSV (default: {_DEFAULT_REPORT_CSV})",
    )
    p.add_argument(
        "--text-cols",
        nargs="+",
        default=_DEFAULT_TEXT_COLS,
        help="Text columns to concatenate from the report CSV",
    )
    p.add_argument(
        "--model",
        default=_DEFAULT_MODEL,
        help=f"Base model to fine-tune (default: {_DEFAULT_MODEL})",
    )
    p.add_argument(
        "--output-dir",
        default=_DEFAULT_OUTPUT_DIR,
        help=f"Where to save the fine-tuned checkpoint (default: {_DEFAULT_OUTPUT_DIR})",
    )
    p.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3)")
    p.add_argument(
        "--lr", type=float, default=2e-5, help="Learning rate (default: 2e-5)"
    )
    p.add_argument(
        "--batch-size", type=int, default=16, help="Per-device batch size (default: 16)"
    )
    p.add_argument(
        "--mlm-probability",
        type=float,
        default=0.15,
        help="Fraction of tokens to mask per example (default: 0.15)",
    )
    p.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        help="Fraction of corpus held out for validation (default: 0.1)",
    )
    p.add_argument(
        "--include-diagnoses",
        action="store_true",
        help=f"Also include {_DEFAULT_DIAGNOSES_CSV} `diagnosis` column in the corpus",
    )
    p.add_argument(
        "--local-only",
        action="store_true",
        help="Use only locally cached model files (no HuggingFace Hub download)",
    )
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps", "xpu"],
        help="Compute device (default: auto)",
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()
    train(
        report_csv=args.csv,
        text_cols=args.text_cols,
        model_name=args.model,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        mlm_probability=args.mlm_probability,
        val_frac=args.val_frac,
        include_diagnoses=args.include_diagnoses,
        local_only=args.local_only,
        device_arg=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
