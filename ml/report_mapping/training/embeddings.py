"""Get-or-build the content-hash embedding cache a training run needs.

Every head trainer consumes the *same* cache ``predict.py`` reads and writes
(``report_mapping.inference.embedding_cache``), keyed on the resolved
backbone's fingerprint (``report_mapping.model.generation.
compute_embedding_fingerprint``). A matching cache is reused untouched, so
heads-only training never re-embeds. The embedding loop itself is
``report_mapping.inference.predict.build_fresh_cache`` — the one place it's
implemented, shared with ``predict.py``'s own cache-miss path — so this
module only resolves the report frame and the cache key.
"""

from __future__ import annotations

import config
import io_utils
from report_mapping import sections
from report_mapping.inference import embedding_cache as embedding_cache_mod
from report_mapping.inference.predict import build_fresh_cache
from report_mapping.model.generation import compute_embedding_fingerprint
from taxonomy.taxonomy import TaxonomyLabel


def get_or_build(
    backbone_dir: str,
    labels_csv: str,
    taxonomy_labels: list[TaxonomyLabel],
    *,
    local_only: bool,
    device,
) -> embedding_cache_mod.EmbeddingCache:
    """Load the cache for (report.csv, labels_csv, backbone_dir); build and
    save it on a miss. Never re-embeds when a matching cache already exists."""
    fingerprint = compute_embedding_fingerprint(backbone_dir)
    key = embedding_cache_mod.content_key(config.REPORT_CSV, labels_csv, fingerprint)
    cache = embedding_cache_mod.load(key)
    if cache is not None:
        return cache

    dataframe = io_utils.read_csv(config.REPORT_CSV, encoding="latin-1")
    ids = dataframe["case_id"].map(sections.clean_text).tolist()
    dataframe = sections.build_section_frame(dataframe)

    cache = build_fresh_cache(backbone_dir, taxonomy_labels, local_only=local_only, device=device,
                               dataframe=dataframe, ids=ids)
    embedding_cache_mod.save(key, cache)
    return cache
