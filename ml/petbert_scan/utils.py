"""Small common helpers for cleaning text, selecting device, and making directories."""

import math
import os
import re

import torch


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def split_numbered_diagnoses(text: str) -> list[str]:
    """Split a numbered clinical diagnosis string into individual sub-diagnoses.

    Veterinary clinical entries often contain multiple diagnoses in a single
    field, formatted as::

        1) Osteosarcoma: right proximal femur 2) Chronic cystitis

    This function detects that pattern and splits the text into separate
    strings -- one per diagnosis.  If no numbered pattern is found the full
    text is returned as a single-element list so callers can always iterate
    over the result.

    Args:
        text: A cleaned clinical diagnosis string (may be empty).

    Returns:
        A list of sub-diagnosis strings.  Always contains at least one element.
        For empty input the result is ``[""]``.
    """
    if not text:
        return [""]

    # Only split when the text starts with "1)" -- this avoids false positives
    # from parenthetical usage like "Grade 3)" inside a regular sentence.
    if not re.match(r"\s*1\)\s", text):
        return [text]

    # Split on numbered markers: "1) ", "2) ", "10) ", etc.
    parts = re.split(r"\s*\d+\)\s*", text)

    # The first element is the (empty) string before "1)".  Filter blanks.
    diagnoses = [part.strip() for part in parts if part.strip()]

    return diagnoses if diagnoses else [text]


def merge_report_columns(row: object, columns: list[str]) -> str:
    """Merge multiple named report text columns into one labelled string.

    Each non-empty column value is prefixed with ``[COLUMN NAME]`` so the
    model can distinguish between sections.  Empty / NaN cells are skipped.

    Args:
        row: A single DataFrame row (pd.Series).
        columns: Ordered list of column names to merge.

    Returns:
        A single string with all non-empty sections joined by a space.
    """
    parts = []
    for col in columns:
        val = clean_text(row.get(col, ""))  # type: ignore[union-attr]
        if val:
            parts.append(f"[{col}] {val}")
    return " ".join(parts)


def device_from_arg(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
