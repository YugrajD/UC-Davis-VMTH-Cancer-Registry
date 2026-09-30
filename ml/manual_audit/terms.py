"""Resolve a reviewer-typed taxonomy term to its code and group.

Used by
``gold.py`` (case-level gold terms), so a term typed once is validated the
same way everywhere.

**Gold is collected as terms, not codes.** A code alone is ambiguous: of the
534 distinct codes in the taxonomy, 186 resolve to more than one term (always
within the same group — 0 of 534 codes span more than one group). A blank
term would make ``evaluation/verdicts.score`` treat a real cancer case as
non-cancer (it keys "good"/"slightly_off" off the term, and false negatives
off an empty label set), so the reverse direction — term to code/group — must
be unambiguous, which it is for all but one of the 845 (code, group, term)
rows: **"Papillary adenocarcinoma"** is the one term that exists in two
groups ("Epithelial neoplasms, NOS" and "Adenomas and adenocarcinomas").
"""

from __future__ import annotations

from collections import defaultdict

from taxonomy.taxonomy import TaxonomyLabel

TermIndex = dict[str, list[TaxonomyLabel]]


class TermResolutionError(Exception):
    """A reviewer-typed term could not be resolved against the taxonomy."""


def build_term_index(labels: list[TaxonomyLabel]) -> TermIndex:
    """term.lower() -> the taxonomy label(s) with that term (usually one)."""
    index: TermIndex = defaultdict(list)
    for label in labels:
        index[label.term.lower()].append(label)
    return index


def resolve_term(raw: str, index: TermIndex) -> tuple[str, str, str]:
    """Resolve reviewer-typed text to (code, term, group) in the taxonomy's own spelling.

    Case-insensitive. Accepts a plain term when it names exactly one taxonomy
    row, or the ``Group: Term`` form to disambiguate the one term that names
    two (or, harmlessly, to confirm any other term's group). Raises
    ``TermResolutionError`` — never returns a partial result — when ``raw``
    does not resolve to exactly one row.
    """
    raw = raw.strip()
    group_hint, term_part = None, raw
    if ":" in raw:
        maybe_group, maybe_term = raw.split(":", 1)
        group_hint, term_part = maybe_group.strip(), maybe_term.strip()

    candidates = index.get(term_part.lower(), [])
    if group_hint is not None:
        match = next((label for label in candidates if label.group.lower() == group_hint.lower()), None)
        if match:
            return match.code, match.term, match.group
        # Fall through: maybe the whole `raw` string (colon included) IS the
        # term, e.g. a term that happens to contain a colon. Re-check as-is.
        candidates = index.get(raw.lower(), [])
        term_part = raw

    if len(candidates) == 1:
        label = candidates[0]
        return label.code, label.term, label.group
    if len(candidates) > 1:
        groups = sorted({label.group for label in candidates})
        raise TermResolutionError(
            f"{term_part!r} is ambiguous — it exists in more than one group ({groups}); "
            f"write it as 'Group: Term' to disambiguate"
        )
    raise TermResolutionError(f"{raw!r} is not a term in the taxonomy — it must be copied exactly from the Term column")
