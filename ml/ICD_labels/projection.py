"""Maps chosen label indices to taxonomy term/group/code fields.

After the categorization step picks a label index for each row, this module
resolves that integer index back to the taxonomy's human-readable fields:
  - term:  e.g. "Hemangiosarcoma, NOS"
  - group: e.g. "Blood vessel tumors"
  - code:  e.g. "9120/3" (Vet-ICD-O-canine-1 morphology code)
"""

from __future__ import annotations

from .taxonomy import TaxonomyLabel


def resolve_taxonomy_matches(
    final_indices: list[int],
    labels: list[str],
    taxonomy_labels: list[TaxonomyLabel] | None,
) -> tuple[list[str], list[str], list[str]]:
    """Convert each row's chosen label index into (term, group, code) strings.

    Args:
        final_indices:   Per-row index into the taxonomy list (-1 if empty).
        labels:          Plain term strings (used as fallback if taxonomy_labels is None).
        taxonomy_labels: Full TaxonomyLabel objects with code/group/term.

    Returns:
        Three parallel lists: (terms, groups, codes).
    """
    if taxonomy_labels is None:
        # Fallback: no taxonomy metadata available, just use the plain label.
        terms = []
        groups = []
        codes = []
        for idx in final_indices:
            if idx < 0 or idx >= len(labels):
                terms.append("")
                groups.append("")
                codes.append("")
            else:
                terms.append(labels[idx])
                groups.append("")
                codes.append("")
        return terms, groups, codes

    # Normal path: look up the full taxonomy record for each index.
    terms = []
    groups = []
    codes = []
    for idx in final_indices:
        if idx < 0 or idx >= len(taxonomy_labels):
            terms.append("")
            groups.append("")
            codes.append("")
            continue
        label = taxonomy_labels[idx]
        terms.append(label.term)
        groups.append(label.group)
        codes.append(label.code)
    return terms, groups, codes
