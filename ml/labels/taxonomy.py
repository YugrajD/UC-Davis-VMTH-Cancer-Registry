"""Reads and normalizes the Vet-ICD-O taxonomy CSV into typed label records."""

from __future__ import annotations

import csv
from dataclasses import dataclass


@dataclass(frozen=True)
class TaxonomyLabel:
    code: str
    group: str
    term: str


def load_labels_taxonomy(labels_csv_path: str) -> list[TaxonomyLabel]:
    with open(labels_csv_path, newline="", encoding="utf-8-sig") as file:
        rows = list(csv.reader(file))

    if len(rows) < 3:
        raise ValueError(f"labels csv appears empty or malformed: {labels_csv_path}")

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
    return [
        f"Veterinary diagnosis term: {label.term}. Group: {label.group}. Code: {label.code}."
        for label in labels
    ]
