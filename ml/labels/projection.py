"""Maps chosen label indices to taxonomy term/group/code fields."""

from __future__ import annotations

from .taxonomy import TaxonomyLabel


def resolve_taxonomy_matches(
    final_indices: list[int],
    labels: list[str],
    taxonomy_labels: list[TaxonomyLabel] | None,
) -> tuple[list[str], list[str], list[str]]:
    if taxonomy_labels is None:
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
