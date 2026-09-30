"""Stage 3b — behavior + subtype keyword correction, and the Lipoma rescue.

Ported from ``production/petbert_pipeline/stages/keyword_correction.py``
(behavior-digit filter + subtype filter) and the Lipoma RESCUE that legacy
keeps in ``stages/__init__.py`` (Intervention 1, 2026-05-25) — both live here
per this rewrite's file boundary (WP4 deliverable 3).
"""

from __future__ import annotations

import re

from taxonomy.behavior import ranked_behaviors
from taxonomy.subtype import filter_by_subtype
from taxonomy.taxonomy import TaxonomyLabel

# Lipoma keyword RESCUE (Intervention 1, 2026-05-25): appended to the
# prediction list, after Stage 3a/3b, when the report has fatty-tissue
# vocabulary, no liposarcoma mention, and the GroupClassifier already gives
# Lipomatous a non-trivial probability — even if it didn't clear the group
# threshold. Targets multi-condition cases where a dominant malignancy
# captures top-1 and lipoma is a one-line incidental finding. Unconditional:
# no flag disables it.
LIPOMA_RESCUE_RE = re.compile(r"\b(?:lipoma|adipocyte|fatty mass)\b", re.I)
LIPOMA_RESCUE_EXCLUDE_RE = re.compile(r"\bliposarcoma\b", re.I)
LIPOMA_RESCUE_GROUP = "Lipomatous neoplasms"
LIPOMA_RESCUE_TERM = "Lipoma, NOS"
LIPOMA_RESCUE_MIN_GROUP_PROB = 0.5


def _behavior_digit(code: str) -> str:
    """The ICD-O behavior digit from a code string like '8000/3' -> '3'."""
    parts = code.split("/")
    return parts[-1][0] if len(parts) > 1 and parts[-1] else ""


def apply_keyword_correction(
    *, text: str, pool: list[int], taxonomy_labels: list[TaxonomyLabel], labels: list[str], group_name: str
) -> list[int]:
    """Narrow ``pool`` by the report's highest-ranked behavior digit, then by
    group-specific subtype keywords. Passes through unchanged on no signal."""
    if not pool:
        return pool
    filtered_pool = pool
    for digit in ranked_behaviors(text):
        filtered = [j for j in pool if _behavior_digit(taxonomy_labels[j].code) == digit]
        if filtered:
            filtered_pool = filtered
            break
    return filter_by_subtype(group_name, filtered_pool, labels, text)


def find_lipoma_rescue_index(taxonomy_labels: list[TaxonomyLabel], labels: list[str]) -> int | None:
    """The global label index of "Lipoma, NOS" in the Lipomatous group, or None
    if the taxonomy doesn't have it."""
    for j, tl in enumerate(taxonomy_labels):
        if tl.group == LIPOMA_RESCUE_GROUP and labels[j] == LIPOMA_RESCUE_TERM:
            return j
    return None


def lipoma_rescue_applies(*, text: str, lipomatous_group_prob: float) -> bool:
    """True when the report text and the (uncapped) Lipomatous group probability
    together justify appending the Lipoma, NOS rescue."""
    return (
        lipomatous_group_prob >= LIPOMA_RESCUE_MIN_GROUP_PROB
        and bool(LIPOMA_RESCUE_RE.search(text))
        and not LIPOMA_RESCUE_EXCLUDE_RE.search(text)
    )
