"""Stage 3b — ICD-O behavior + subtype keyword correction.

Refines a candidate label pool by:
  1. Picking the highest-priority ICD-O behavior digit found in the report text
     and keeping only labels whose codes carry that behavior.
  2. Applying group-specific subtype keyword filters via ``filter_by_subtype``.

Used both as a post-filter on Stage 3a output and as the standalone Stage 3
when no LabelPresenceClassifier is available for the group.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ICD_labels import filter_by_subtype, ranked_behaviors

if TYPE_CHECKING:
    from ICD_labels import TaxonomyLabel


def _behavior_digit(code: str) -> str:
    """Return the ICD-O behavior digit from a code string like '8000/3' -> '3'."""
    parts = code.split("/")
    return parts[-1][0] if len(parts) > 1 and parts[-1] else ""


def apply_keyword_correction(
    *,
    text: str,
    pool: list[int],
    taxonomy_labels: list["TaxonomyLabel"],
    labels: list[str],
    group_name: str,
) -> list[int]:
    if not pool:
        return pool
    behavior_rank = ranked_behaviors(text)
    filtered_pool = pool
    for b in behavior_rank:
        filtered = [j for j in pool if _behavior_digit(taxonomy_labels[j].code) == b]
        if filtered:
            filtered_pool = filtered
            break
    return filter_by_subtype(group_name, filtered_pool, labels, text)
