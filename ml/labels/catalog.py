"""Builds the taxonomy-backed label catalog used for embedding comparisons.

The LabelCatalog bundles three things the pipeline needs:
  1. ``labels`` -- plain term strings used for display ("Hemangiosarcoma, NOS").
  2. ``label_texts`` -- descriptive sentences used as PetBERT input for
     embedding each label (see taxonomy.build_taxonomy_label_texts).
  3. ``taxonomy_labels`` -- the full TaxonomyLabel objects so that the
     final index can be resolved back to a code and group.
"""

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
    """Load the taxonomy CSV and build the label catalog for the pipeline.

    Steps:
      1. Parse ml/labels/labels.csv into TaxonomyLabel records.
      2. Extract the plain term string for each label.
      3. Build descriptive sentences (used as PetBERT input for label embedding).
    """
    taxonomy_labels = load_labels_taxonomy(config.labels_csv_path)
    labels = [label.term for label in taxonomy_labels]
    return LabelCatalog(
        labels=labels,
        label_texts=build_taxonomy_label_texts(taxonomy_labels),
        taxonomy_labels=taxonomy_labels,
        include_score_columns=False,
    )
