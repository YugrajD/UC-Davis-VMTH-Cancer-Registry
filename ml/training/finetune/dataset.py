"""PyTorch Dataset for MLM domain adaptation on cancer registry reports.

Reads one or more CSV files, concatenates specified text columns per row, and
tokenizes the results.  Masking is intentionally left to DataCollatorForLanguageModeling
so that a different random mask is drawn on every epoch pass.

Usage:
    tokenizer = AutoTokenizer.from_pretrained("SAVSNET/PetBERT")
    ds = ReportMLMDataset(
        tokenizer,
        sources=[
            ("ml/data/report.csv", ["HISTOPATHOLOGICAL SUMMARY", "FINAL COMMENT", "ANCILLARY TESTS"]),
            ("ml/data/diagnoses.csv", ["diagnosis"]),
        ],
    )
"""

from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class ReportMLMDataset(Dataset):
    """Tokenized veterinary report texts for masked language model training.

    Each item is a dict with ``input_ids`` and ``attention_mask`` tensors.
    No padding is applied here — the HuggingFace DataCollatorForLanguageModeling
    handles per-batch dynamic padding and masking.

    Args:
        tokenizer: A HuggingFace tokenizer (e.g. AutoTokenizer for PetBERT).
        sources: List of (csv_path, text_cols) pairs.  For each CSV the listed
            columns are concatenated with ``"\\n\\n"`` to form one text per row.
            Rows where all columns are empty are silently dropped.
        max_length: Maximum token count per example (truncation applied).

    Raises:
        FileNotFoundError: If any csv_path does not exist.
        ValueError: If the combined corpus is empty after filtering blank rows.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        sources: list[tuple[str, list[str]]],
        max_length: int = 512,
    ) -> None:
        texts = _load_texts(sources)
        if not texts:
            raise ValueError(
                "All rows across all CSV sources produced empty text — "
                "check that the specified column names exist and contain data."
            )

        print(f"[MLM dataset] {len(texts):,} texts loaded. Tokenizing...", flush=True)
        encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,  # collator pads per-batch
        )
        self._input_ids: list[list[int]] = encodings["input_ids"]
        self._attention_mask: list[list[int]] = encodings["attention_mask"]
        print(f"[MLM dataset] Tokenization complete.", flush=True)

    def __len__(self) -> int:
        return len(self._input_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor(self._input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self._attention_mask[idx], dtype=torch.long),
        }


def _load_texts(sources: list[tuple[str, list[str]]]) -> list[str]:
    """Load and concatenate text columns from each CSV source.

    DATA FLOW
    =========
    sources[0]: (report.csv, [col_A, col_B, col_C])
        row 0 → "text_A\n\ntext_B\n\ntext_C"  (non-empty parts only)
        row 1 → "text_A\n\ntext_C"             (col_B was NaN)
        ...
    sources[1]: (diagnoses.csv, [diagnosis])
        row 0 → "SKIN: SQUAMOUS CELL CARCINOMA"
        ...
    result: all non-blank strings, concatenated into one list
    """
    from pathlib import Path

    all_texts: list[str] = []
    for csv_path, text_cols in sources:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)

        missing = [c for c in text_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"{csv_path}: columns not found: {missing}. "
                f"Available: {list(df.columns)}"
            )

        for _, row in df.iterrows():
            parts = []
            for col in text_cols:
                val = str(row[col]).strip()
                if val and val.lower() != "nan":
                    parts.append(val)
            if parts:
                all_texts.append("\n\n".join(parts))

    return all_texts
