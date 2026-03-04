"""Reads and normalizes the Vet-ICD-O taxonomy CSV into typed label records.

The taxonomy file (ml/labels/labels.csv) uses the Vet-ICD-O-canine-1 coding
system.  Each row defines one diagnosis term with:
  - code:  ICD-O morphology code (e.g. "9120/3")
  - group: Tumor category (e.g. "Blood vessel tumors")
  - term:  Specific diagnosis name (e.g. "Hemangiosarcoma, NOS")

This module also provides ``build_taxonomy_label_texts`` which converts each
label into a natural-language sentence that PetBERT can embed, so that
diagnosis text and label text live in the same embedding space.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass


@dataclass(frozen=True)
class TaxonomyLabel:
    """A single entry from the Vet-ICD-O taxonomy."""
    code: str
    group: str
    term: str


def load_labels_taxonomy(labels_csv_path: str) -> list[TaxonomyLabel]:
    """Parse the taxonomy CSV and return deduplicated TaxonomyLabel records.

    The CSV has a non-standard layout: row 0 is a title row, row 1 is the
    actual header, and data begins at row 2.  Duplicate (code, group, term)
    tuples are dropped.
    """
    with open(labels_csv_path, newline="", encoding="utf-8-sig") as file:
        rows = list(csv.reader(file))

    if len(rows) < 3:
        raise ValueError(f"labels csv appears empty or malformed: {labels_csv_path}")

    # Row 1 is the real header (row 0 is a title/comment row).
    header = list(rows[1])
    while header and header[-1] == "":
        header.pop()
    if not header:
        raise ValueError(f"labels csv missing header row: {labels_csv_path}")

    expected = [
        "Vet-ICD-O-canine-1 code",
        "Group",
        "Term",
        "level",
        "Topography",
        "obs",
    ]
    if header[: len(expected)] != expected:
        raise ValueError(f"Unexpected labels header: {header!r}")

    records: list[TaxonomyLabel] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows[2:]:
        values = row[: len(header)] + [""] * max(0, len(header) - len(row))
        if not any(cell.strip() for cell in values):
            continue
        record = dict(zip(header, values))
        code = record["Vet-ICD-O-canine-1 code"].strip()
        group = record["Group"].strip()
        term = record["Term"].strip()
        if not code or not group or not term:
            continue
        dedupe_key = (code, group, term)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        records.append(
            TaxonomyLabel(
                code=code,
                group=group,
                term=term,
            )
        )
    return records


def build_taxonomy_label_texts(labels: list[TaxonomyLabel]) -> list[str]:
    """Convert each taxonomy label into a short text string for embedding.

    Example output:
      "Hemangiosarcoma, NOS Blood vessel tumors"

    These strings are passed through PetBERT to produce label embeddings
    that can be compared (via cosine similarity) against diagnosis text
    embeddings in the same 768-dim vector space.
    """
    return [
        f"{label.term} {label.group}"
        for label in labels
    ]
