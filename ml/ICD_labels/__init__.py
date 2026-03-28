"""Label-domain modules: taxonomy loading, catalog building, and projection."""

from .taxonomy import TaxonomyLabel, load_labels_taxonomy
from .catalog import LabelCatalog, label_catalog_for_config
from .projection import resolve_taxonomy_matches
from .behavior_keywords import best_behavior
from .enrichment import compute_enriched_label_embeddings

__all__ = [
    "TaxonomyLabel",
    "load_labels_taxonomy",
    "LabelCatalog",
    "label_catalog_for_config",
    "resolve_taxonomy_matches",
    "best_behavior",
    "compute_enriched_label_embeddings",
]
