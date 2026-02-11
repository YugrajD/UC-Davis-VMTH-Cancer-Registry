"""Builds the taxonomy-backed label catalog used for embedding comparisons."""

from __future__ import annotations

from dataclasses import dataclass

from petbert_scan.types import ScanConfig

from .taxonomy import TaxonomyLabel, build_taxonomy_label_texts, load_labels_taxonomy


@dataclass(frozen=True)
class LabelCatalog:
    labels: list[str]
    label_texts: list[str]
    taxonomy_labels: list[TaxonomyLabel] | None
    include_score_columns: bool


def label_catalog_for_config(config: ScanConfig) -> LabelCatalog:
    taxonomy_labels = load_labels_taxonomy(config.labels_csv_path)
    labels = [label.term for label in taxonomy_labels]
    return LabelCatalog(
        labels=labels,
        label_texts=build_taxonomy_label_texts(taxonomy_labels),
        taxonomy_labels=taxonomy_labels,
        include_score_columns=False,
    )
