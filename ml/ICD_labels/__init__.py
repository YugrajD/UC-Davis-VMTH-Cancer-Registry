"""Label-domain modules: taxonomy loading, catalog building, and projection."""

from .taxonomy import TaxonomyLabel, load_labels_taxonomy
from .catalog import LabelCatalog, label_catalog_for_config
from .projection import resolve_taxonomy_matches
from .behavior_keywords import best_behavior, ranked_behaviors
from .subtype_keywords import filter_by_subtype

__all__ = [
    "TaxonomyLabel",
    "load_labels_taxonomy",
    "LabelCatalog",
    "label_catalog_for_config",
    "resolve_taxonomy_matches",
    "best_behavior",
    "ranked_behaviors",
    "filter_by_subtype",
]
