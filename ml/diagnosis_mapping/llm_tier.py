"""Tier 3: signal fallback + LLM resolution.

Only reached when ``keyword_tiers.has_signal`` says the masked diagnosis still
contains a cancer-indicating token and neither Tier 1 nor Tier 2 matched. A
group token index picks the most likely taxonomy group (or an -oma/-emia
suffix index is used as a fallback); its terms become the LLM's candidate
list. ``decision_stage``/``method`` values are unchanged from the old
pipeline: tier3_llm/{LLM, No Match, Uncertain}, tier3_no_candidates/No Match.

Ported from ``ml/annotation/llm_pipeline/pipeline.py``.
"""

from __future__ import annotations

import re
from dataclasses import replace
from difflib import get_close_matches

import requests

from taxonomy.taxonomy import TaxonomyLabel
from diagnosis_mapping import llm_client
from diagnosis_mapping.keyword_tiers import (
    NO_MATCH,
    OMA_RE,
    QUALIFIER_RE,
    UNCERTAIN_RESULT,
    MatchResult,
    extract_site,
    normalize,
)

# LLM Tier 3: maximum number of candidate terms passed to the model.
LLM_MAX_CANDIDATES = 30

# Stopwords dropped from group-name tokens so they don't drive false-positive
# overlaps with diagnosis text (e.g. "and" matching every diagnosis with "and").
_GROUP_TOKEN_STOPWORDS = frozenset({
    "and", "the", "for", "of", "or", "with", "in", "on", "at", "by",
    "nos", "nec", "diffuse",
})

# 1-letter tokens that are discriminating in pathology context (B-cell vs T-cell
# vs NK-cell lymphomas). Kept despite the len>=3 filter.
_GROUP_TOKEN_KEEP = frozenset({"b", "t", "nk"})


# ---------------------------------------------------------------------------
# Candidate assembly: group identification, -oma/-emia suffix fallback
# ---------------------------------------------------------------------------


def build_oma_index(taxonomy_labels: list[TaxonomyLabel]) -> dict[str, int]:
    """Map each single -oma word in any taxonomy term to its label index. First wins."""
    oma_index: dict[str, int] = {}
    for i, label in enumerate(taxonomy_labels):
        norm = normalize(label.term)
        for word in norm.split():
            if (word.endswith("oma") or word.endswith("emia")) and word not in oma_index:
                oma_index[word] = i
    return oma_index


def build_group_token_index(taxonomy_labels: list[TaxonomyLabel]) -> dict[str, set[str]]:
    """Map each unique group name to its set of distinct content tokens.

    Used by `identify_group` to score groups by token overlap with the
    diagnosis, preferring more specific group matches (e.g. 'Mature T-cell
    lymphomas' over 'Malignant lymphomas, NOS or diffuse'). Tokens shorter
    than 3 chars and English/taxonomy stop-words are dropped to reduce noise.
    """
    idx: dict[str, set[str]] = {}
    for label in taxonomy_labels:
        if label.group in idx:
            continue
        norm = normalize(label.group)
        core = QUALIFIER_RE.sub("", norm).strip()
        # Strip trailing "s" as a poor-man's stem so "lymphomas" matches
        # diagnosis tokens spelt "lymphoma".
        tokens = {
            t.rstrip("s") for t in core.split()
            if (len(t) >= 3 or t in _GROUP_TOKEN_KEEP) and t not in _GROUP_TOKEN_STOPWORDS
        }
        if tokens:
            idx[label.group] = tokens
    return idx


def identify_group(
    norm_text: str,
    group_token_index: dict[str, set[str]],
) -> str | None:
    """Return the group whose name has the most distinct tokens in the diagnosis.

    Tiebreak by longest group name. Returns None if no group has any token
    overlap. This replaces a first-match approach, which collapsed T-cell /
    B-cell lymphomas into the broader "Malignant lymphomas, NOS" bucket
    because that group's keywords were sorted earlier.
    """
    diag_tokens = {t.rstrip("s") for t in norm_text.split()}
    best_group: str | None = None
    best_score: tuple[int, int] = (0, 0)
    for group_name, tokens in group_token_index.items():
        match_count = len(tokens & diag_tokens)
        if match_count == 0:
            continue
        score = (match_count, len(group_name))
        if score > best_score:
            best_score = score
            best_group = group_name
    return best_group


def _candidates_for(
    masked_text: str,
    group_token_index: dict[str, set[str]],
    taxonomy_labels: list[TaxonomyLabel],
    oma_index: dict[str, int],
) -> list[TaxonomyLabel]:
    """Group's full term list, or an -oma/-emia suffix fallback when no group matches.

    Uses masked_text so a "granuloma" inside a negated span, or a "non-B cell"
    mention, doesn't pull the group selection toward the wrong taxonomy or
    surface candidates the LLM would have to argue against.
    """
    group = identify_group(masked_text, group_token_index)
    if group:
        return [l for l in taxonomy_labels if l.group == group]
    raw_words = OMA_RE.findall(masked_text)
    indices = {oma_index[w] for w in set(raw_words) if w in oma_index}
    return [taxonomy_labels[i] for i in sorted(indices)]


# ---------------------------------------------------------------------------
# Prompt building + response parsing
# ---------------------------------------------------------------------------


def build_llm_prompt(original_text: str, candidates: list[TaxonomyLabel]) -> str:
    term_list = "\n".join(f"{i + 1}. {c.term}" for i, c in enumerate(candidates))
    site = extract_site(original_text)
    site_hint = ""
    if site:
        site_hint = (
            f"\nAnatomic site context: {site}.\n"
            "Use this to disambiguate site-specific subtypes — e.g. "
            '"Plasmacytoma, extramedullary (cutaneous)" applies to skin sites only; '
            "use a different variant for mucosal, nodal, or visceral sites.\n"
        )
    return (
        "You are a veterinary oncology classifier. "
        "Map the diagnosis below to the best matching ICD term.\n\n"
        f'Diagnosis: "{original_text}"\n'
        f"{site_hint}"
        f"\nCandidate ICD terms:\n{term_list}\n\n"
        "Rules:\n"
        "- Reply with ONLY the exact text of the best matching candidate "
        "(copy it character-for-character).\n"
        "- If the diagnosis is negated anywhere in the text — e.g. contains phrases like "
        '"no evidence of", "no metastasis", "not observed", "rule out", '
        '"not consistent with", "negative for", "cannot exclude" — reply with: no match\n'
        "- If the diagnosis is uncertain or hedged — e.g. contains phrases like "
        '"presumed", "suspect", "possible", "suspected", "consistent with", "compatible with", '
        '"likely", "probable", "versus", "vs." — reply with: uncertain\n'
        "- If no candidate fits the diagnosis, reply with: no match\n\n"
        "Your answer:"
    )


def parse_llm_response(
    response: str,
    candidates: list[TaxonomyLabel],
    method: str = "LLM",
) -> MatchResult | None:
    """Match LLM response text back to a taxonomy label. Returns None on no match."""
    response = response.strip()
    if not response or response.lower() == "no match":
        return None
    if response.lower() == "uncertain":
        return UNCERTAIN_RESULT
    for label in candidates:
        if label.term.lower() == response.lower():
            return MatchResult(
                term=label.term, group=label.group, code=label.code,
                keyword=response, method=method, confidence=1.0, stage="tier3_llm",
            )
    term_names = [c.term for c in candidates]
    close = get_close_matches(response, term_names, n=1, cutoff=0.8)
    if close:
        label = next(c for c in candidates if c.term == close[0])
        return MatchResult(
            term=label.term, group=label.group, code=label.code,
            keyword=response, method=method, confidence=0.9, stage="tier3_llm",
        )
    return None


# ---------------------------------------------------------------------------
# Tier 3 entry point
# ---------------------------------------------------------------------------


def run_tier3(
    original_text: str,
    masked_text: str,
    group_token_index: dict[str, set[str]],
    taxonomy_labels: list[TaxonomyLabel],
    oma_index: dict[str, int],
    *,
    llm_enabled: bool,
    model: str | None,
    timeout: int,
    counters: dict,
) -> MatchResult:
    candidates = _candidates_for(masked_text, group_token_index, taxonomy_labels, oma_index)
    if not candidates:
        return replace(NO_MATCH, stage="tier3_no_candidates")

    candidates = candidates[:LLM_MAX_CANDIDATES]

    if not llm_enabled:
        # No local LLM server for this run (e.g. --no-llm, or the keyword-only
        # replay). Every tier3-eligible row is recorded exactly like a
        # declined/failed call below - the existing tier3_llm/"No Match"
        # combination - never a new decision_stage value.
        counters["tier3_no_match"] = counters.get("tier3_no_match", 0) + 1
        return replace(NO_MATCH, stage="tier3_llm")

    prompt = build_llm_prompt(original_text, candidates)
    counters["tier3_calls"] = counters.get("tier3_calls", 0) + 1
    # A request failure and a genuine "no match" reply are both counted as
    # tier3_no_match, so stage="tier3_llm" is an upper bound on real declines.
    try:
        response = llm_client.chat(prompt, model=model, timeout=timeout)
    except (requests.RequestException, KeyError, ValueError):
        counters["tier3_no_match"] = counters.get("tier3_no_match", 0) + 1
        return replace(NO_MATCH, stage="tier3_llm")

    result = parse_llm_response(response, candidates)
    if result is None:
        counters["tier3_no_match"] = counters.get("tier3_no_match", 0) + 1
        return replace(NO_MATCH, stage="tier3_llm")
    if result.method == "Uncertain":
        counters["tier3_uncertain"] = counters.get("tier3_uncertain", 0) + 1
    else:
        counters["tier3_matched"] = counters.get("tier3_matched", 0) + 1
    return result
