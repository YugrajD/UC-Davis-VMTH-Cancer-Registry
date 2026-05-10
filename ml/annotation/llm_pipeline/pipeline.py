"""LLM-assisted pipeline for mapping diagnosis text to Vet-ICD-O taxonomy labels.

Flow:
  Pre-pass : normalize text + expand abbreviations & synonyms + mask negations
  Tier 1   : exact match  (keyword index: full term, core term, permutations, plurals)
  Tier 2   : fuzzy match  (token-overlap on core terms; behavior-code aware)
  Tier 3   : signal fallback + LLM  (diagnoses that contain cancer-indicating terms)
  No match : row labeled non-cancer
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from itertools import permutations as _permutations

import pandas as pd
import requests

from ICD_labels import TaxonomyLabel, load_labels_taxonomy
from annotation.llm_pipeline.client import chat
from utils.csv_io import strip_bom_from_columns


# ---------------------------------------------------------------------------
# Text normalization + keyword indexing
# (previously imported from annotation.keyword_pipeline; inlined when that
# package was removed)
# ---------------------------------------------------------------------------

_OMA_RE = re.compile(r"\b(\w+(?:oma|emia))s?\b")

_QUALIFIER_RE = re.compile(
    r",?\s*\b("
    r"nos|nec|conventional|well differentiated|spindle cell|kaposiform|"
    r"epithelioid|inflammatory lobular capillary|mixed capillary cavernous|"
    r"retiform|malignant|benign|presumptive|adult type|juvenile type|atypical|"
    r"gist"
    r")\b.*$",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    """Lowercase, collapse punctuation to spaces, normalize whitespace, apply synonyms."""
    text = text.lower()
    text = re.sub(r"[-_/]", " ", text)
    text = re.sub(r"[,();:]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\bneoplasia\b", "neoplasm", text)
    text = re.sub(r"\bplasma cell tumor\b", "plasmacytoma", text)
    return text


def _build_keyword_index(
    taxonomy_labels: list[TaxonomyLabel],
) -> list[tuple[str, re.Pattern, int]]:
    """Build (core_keyword, pattern, taxonomy_index) sorted by keyword length descending.

    Two keyword candidates per label: full normalized term and core term with
    qualifiers stripped. Longer keywords are tried first so more specific terms
    take priority. Duplicate keywords are skipped — first label to define a
    keyword wins (Preferred terms appear first in the taxonomy CSV).
    """
    entries: list[tuple[str, re.Pattern, int]] = []
    seen: set[str] = set()
    for i, label in enumerate(taxonomy_labels):
        norm = _normalize(label.term)
        core = _QUALIFIER_RE.sub("", norm).strip().strip(",").strip()
        candidates: set[str] = set()
        for kw in {norm, core}:
            kw = kw.strip()
            candidates.add(kw)
            words = kw.split()
            if 2 <= len(words) <= 3:
                for perm in _permutations(words):
                    candidates.add(" ".join(perm))
        for kw in candidates:
            if len(kw) < 6 or kw in seen:
                continue
            seen.add(kw)
            pat = re.compile(r"\b" + re.escape(kw) + r"s?\b")
            entries.append((kw, pat, i))
    entries.sort(key=lambda x: len(x[0]), reverse=True)
    return entries


def _build_oma_index(taxonomy_labels: list[TaxonomyLabel]) -> dict[str, int]:
    """Map each single -oma word in any taxonomy term to its label index. First wins."""
    oma_index: dict[str, int] = {}
    for i, label in enumerate(taxonomy_labels):
        norm = _normalize(label.term)
        for word in norm.split():
            if (word.endswith("oma") or word.endswith("emia")) and word not in oma_index:
                oma_index[word] = i
    return oma_index


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Abbreviation expansions and synonym mappings applied after normalization.
_ABBREVIATIONS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bcpnet\b"), "central primitive neuroectodermal tumor"),
    (re.compile(r"\bpnet\b"), "primitive neuroectodermal tumor"),
    (re.compile(r"\bdlbcl\b"), "diffuse large b cell lymphoma"),
    (re.compile(r"\bgist\b"), "gastrointestinal stromal tumor"),
    (re.compile(r"\btvt\b"), "transmissible venereal tumor"),
    (re.compile(r"\bhsa\b"), "hemangiosarcoma"),
    (re.compile(r"\bosa\b"), "osteosarcoma"),
    (re.compile(r"\bhcc\b"), "hepatocellular carcinoma"),
    (re.compile(r"\bscc\b"), "squamous cell carcinoma"),
    (re.compile(r"\bmct\b"), "mast cell tumor"),
    (re.compile(r"\bangiosarcoma\b"), "hemangiosarcoma"),
    (re.compile(r"\bplasmacell\b"), "plasmacytoma"),
    (re.compile(r"\bperivascular wall tumor\b"), "canine perivascular wall tumor"),
]

_RE_METASTASIS = re.compile(r"\bmetastasis\b")

# Signal terms that suggest a neoplastic diagnosis and trigger Tier 3.
_SIGNAL_RE = re.compile(
    r"\b(?:"
    r"tumor|tumour|leukemia|leukaemia|neoplasm|cancer|malignancy|malignant|metastatic|"
    r"carcinoid|mycosis|fungoides|polycythemia|paget|mastocytosis|"
    r"ganglioneuromatosis|gliomatosis|lipomatosis|refractory\s+anemia|"
    r"acanthomatous|fibromatosis"
    r")\b",
    re.IGNORECASE,
)

# Negation phrases consume the phrase + up to N following tokens. Catches
# "no evidence of neoplasia", "negative for malignancy", "rule out lymphoma".
_NEGATION_PHRASE_RE = re.compile(
    r"\b(?:"
    r"no\s+(?:histo(?:patho)?logic\s+)?evidence\s+of|"
    r"negative\s+for|"
    r"absence\s+of|"
    r"without\s+evidence\s+of|"
    r"not\s+consistent\s+with|"
    r"cannot\s+be\s+confirmed|"
    r"rule[ds]?\s+out|"
    r"no\s+signs?\s+of"
    r")\b(?:\s+\w+){0,6}",
    re.IGNORECASE,
)

# "non-X" / "non X" / "non X cell" — masks the negated phrase plus 1-2 tokens.
# Catches "non-B cell", "non-T cell", "non-neoplastic". Limit of 2 trailing
# tokens prevents over-consumption past sentence boundaries.
_NON_X_RE = re.compile(r"\bnon(?:[-\s]\w+){1,2}\b", re.IGNORECASE)

# Anatomic site detection for LLM prompt enrichment (Fix 5). First match wins.
_SITE_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("skin",        re.compile(r"\b(?:haired\s+skin|cutaneous|subcutis|subcutaneous|dermis|dermal|epidermis|skin)\b", re.IGNORECASE)),
    ("mucosal",     re.compile(r"\b(?:gingiva|tongue|buccal|palate|oral\s+mucos\w*|mucosal|\blip\b)\b", re.IGNORECASE)),
    ("lymph node",  re.compile(r"\blymph\s*nodes?\b|\bnodal\b", re.IGNORECASE)),
    ("spleen",      re.compile(r"\b(?:spleen|splenic)\b", re.IGNORECASE)),
    ("liver",       re.compile(r"\b(?:liver|hepatic)\b", re.IGNORECASE)),
    ("bone",        re.compile(r"\b(?:bone|osseous|skeletal)\b", re.IGNORECASE)),
    ("eye",         re.compile(r"\b(?:eye|ocular|cornea|retina|conjunctiv\w*|uveal|iris)\b", re.IGNORECASE)),
    ("nasal",       re.compile(r"\bnasal\b", re.IGNORECASE)),
    ("kidney",      re.compile(r"\b(?:kidney|renal)\b", re.IGNORECASE)),
    ("lung",        re.compile(r"\b(?:lung|pulmonary)\b", re.IGNORECASE)),
    ("mammary",     re.compile(r"\b(?:mammary|breast)\b", re.IGNORECASE)),
    ("intestinal",  re.compile(r"\b(?:intestin\w*|colon|rectum|cecum|jejun\w*|ileum|duoden\w*)\b", re.IGNORECASE)),
    ("brain",       re.compile(r"\b(?:brain|cerebr\w*|cortex|spinal\s+cord|meninges|meningeal)\b", re.IGNORECASE)),
    ("oral cavity", re.compile(r"\boral\b", re.IGNORECASE)),
]

# Fuzzy Tier 2: minimum fraction of core-term tokens that must appear in diagnosis.
_FUZZY_THRESHOLD = 0.85
# Behavior-filtered fuzzy: lower threshold because the candidate set is already
# narrowed to ICD-O behavior-matching labels, so false-positive risk is lower.
_FUZZY_THRESHOLD_BEHAVIOR = 0.7

# LLM Tier 3: maximum number of candidate terms passed to the model.
_LLM_MAX_CANDIDATES = 30


# ---------------------------------------------------------------------------
# Config / output types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LLMConfig:
    csv_path: str
    id_col: str
    diag_num_col: str
    text_col: str
    labels_csv_path: str
    out_dir: str
    max_rows: int | None
    llm_timeout: int = 60
    llm_model: str | None = None  # None uses the default from .env


@dataclass(frozen=True)
class LLMOutputs:
    predictions_csv: str
    summary_json: str
    summary_md: str


@dataclass(frozen=True)
class _MatchResult:
    term: str
    group: str
    code: str
    keyword: str
    method: str        # "Exact" | "Fuzzy" | "LLM" | "No Match" | "Uncertain"
    confidence: float  # 1.0=Exact/LLM, 0.0–1.0=Fuzzy, 0.0=No Match/Uncertain


_NO_MATCH = _MatchResult(term="", group="", code="", keyword="", method="No Match", confidence=0.0)
_UNCERTAIN_RESULT = _MatchResult(term="", group="", code="", keyword="", method="Uncertain", confidence=0.0)


# ---------------------------------------------------------------------------
# Pre-pass: normalization + negation masking
# ---------------------------------------------------------------------------

def _normalize_llm(text: str) -> str:
    """Normalize text using the keyword pipeline's rules, then expand abbreviations."""
    text = _normalize(text)
    text = _RE_METASTASIS.sub("metastatic neoplasm", text)
    for pattern, expansion in _ABBREVIATIONS:
        text = pattern.sub(expansion, text)
    return text


def _mask_negation(text: str) -> str:
    """Blank out negated regions so Tier 1/2 keyword matching skips them.

    Tier 3 (LLM) keeps the original text — the prompt has its own negation rules
    and benefits from full context. Negated regions are replaced with spaces, not
    deleted, so token offsets and word boundaries are preserved.
    """
    text = _NEGATION_PHRASE_RE.sub(" ", text)
    text = _NON_X_RE.sub(" ", text)
    return text


# ---------------------------------------------------------------------------
# Behavior-code helpers (Fix 2)
# ---------------------------------------------------------------------------

def _detect_behavior(text: str) -> str | None:
    """Detect ICD-O behavior digit from explicit modifiers in the diagnosis.

    Returns '0' (benign), '2' (in situ), '3' (malignant/metastatic), or None.
    Order matters: 'in situ' is checked before 'benign' since "carcinoma in situ"
    contains neither benign nor malignant.
    """
    if re.search(r"\bin\s*situ\b", text):
        return "2"
    if re.search(r"\bbenign\b", text):
        return "0"
    if re.search(r"\b(?:malignant|metastatic)\b", text):
        return "3"
    return None


def _behavior_digit(code: str) -> str | None:
    """Extract the behavior digit from an ICD-O code like '8830/3' or '8410.1/0'."""
    if "/" not in code:
        return None
    after_slash = code.split("/")[-1]
    return after_slash[0] if after_slash else None


# ---------------------------------------------------------------------------
# Tier 1: Exact match (keyword index)
# ---------------------------------------------------------------------------

def _tier1_exact(
    norm_text: str,
    keyword_index: list[tuple[str, re.Pattern, int]],
    taxonomy_labels: list[TaxonomyLabel],
) -> _MatchResult | None:
    for kw, pattern, label_idx in keyword_index:
        if pattern.search(norm_text):
            label = taxonomy_labels[label_idx]
            return _MatchResult(
                term=label.term, group=label.group, code=label.code,
                keyword=kw, method="Exact", confidence=1.0,
            )
    return None


# ---------------------------------------------------------------------------
# Tier 2: Fuzzy match (token overlap on core terms, behavior-aware)
# ---------------------------------------------------------------------------

def _core_term(norm_term: str) -> str:
    """Strip qualifier words from a normalized term to get its core tokens."""
    return _QUALIFIER_RE.sub("", norm_term).strip().strip(",").strip()


def _token_overlap(core: str, norm_text: str) -> float:
    """Fraction of core tokens found in the diagnosis tokens."""
    core_tokens = core.split()
    if len(core_tokens) < 2:
        return 0.0
    text_tokens = set(norm_text.split())
    matched = sum(1 for t in core_tokens if t in text_tokens)
    return matched / len(core_tokens)


def _tier2_fuzzy(
    norm_text: str,
    taxonomy_labels: list[TaxonomyLabel],
) -> _MatchResult | None:
    """Token-overlap fuzzy match. Behavior-aware: prefers candidates whose ICD-O
    behavior digit (/0, /2, /3) matches an explicit benign / in-situ / malignant
    modifier in the diagnosis. Falls back to unfiltered match if no behavior-
    matching candidate clears the threshold.

    When filter_behavior=True the comparison uses the full normalized term
    rather than the qualifier-stripped core, so benign/malignant tokens stay
    in the comparison and a diagnosis like "mixed mammary tumor, benign"
    actually matches "Benign mixed tumor, NOS" (whose core would otherwise
    be empty after qualifier stripping).
    """
    behavior_pref = _detect_behavior(norm_text)

    def _scan(filter_behavior: bool) -> _MatchResult | None:
        threshold = _FUZZY_THRESHOLD_BEHAVIOR if (filter_behavior and behavior_pref) else _FUZZY_THRESHOLD
        best: tuple[float, _MatchResult] | None = None
        for label in taxonomy_labels:
            if filter_behavior and behavior_pref:
                digit = _behavior_digit(label.code)
                if digit is not None and digit != behavior_pref:
                    continue
            term_str = _normalize(label.term) if filter_behavior and behavior_pref else _core_term(_normalize(label.term))
            if not term_str:
                continue
            score = _token_overlap(term_str, norm_text)
            if score >= threshold:
                if best is None or score > best[0]:
                    best = (score, _MatchResult(
                        term=label.term, group=label.group, code=label.code,
                        keyword=term_str, method="Fuzzy", confidence=round(score, 2),
                    ))
        return best[1] if best else None

    if behavior_pref:
        result = _scan(filter_behavior=True)
        if result:
            return result
    return _scan(filter_behavior=False)


# ---------------------------------------------------------------------------
# Tier 3: Signal fallback + LLM
# ---------------------------------------------------------------------------

def _has_signal(norm_text: str) -> bool:
    """Return True if the diagnosis contains any cancer-indicating term."""
    return bool(_OMA_RE.search(norm_text) or _SIGNAL_RE.search(norm_text))


# Stopwords dropped from group-name tokens so they don't drive false-positive
# overlaps with diagnosis text (e.g. "and" matching every diagnosis with "and").
_GROUP_TOKEN_STOPWORDS = frozenset({
    "and", "the", "for", "of", "or", "with", "in", "on", "at", "by",
    "nos", "nec", "diffuse",
})

# 1-letter tokens that are discriminating in pathology context (B-cell vs T-cell
# vs NK-cell lymphomas). Kept despite the len>=3 filter.
_GROUP_TOKEN_KEEP = frozenset({"b", "t", "nk"})


def _build_group_token_index(
    taxonomy_labels: list[TaxonomyLabel],
) -> dict[str, set[str]]:
    """Map each unique group name to its set of distinct content tokens.

    Used by `_identify_group` to score groups by token overlap with the
    diagnosis, preferring more specific group matches (e.g. 'Mature T-cell
    lymphomas' over 'Malignant lymphomas, NOS or diffuse'). Tokens shorter
    than 3 chars and English/taxonomy stop-words are dropped to reduce noise.
    """
    idx: dict[str, set[str]] = {}
    for label in taxonomy_labels:
        if label.group in idx:
            continue
        norm = _normalize(label.group)
        core = _QUALIFIER_RE.sub("", norm).strip()
        # Strip trailing "s" as a poor-man's stem so "lymphomas" matches
        # diagnosis tokens spelt "lymphoma".
        tokens = {
            t.rstrip("s") for t in core.split()
            if (len(t) >= 3 or t in _GROUP_TOKEN_KEEP) and t not in _GROUP_TOKEN_STOPWORDS
        }
        if tokens:
            idx[label.group] = tokens
    return idx


def _identify_group(
    norm_text: str,
    group_token_index: dict[str, set[str]],
) -> str | None:
    """Return the group whose name has the most distinct tokens in the diagnosis.

    Tiebreak by longest group name. Returns None if no group has any token
    overlap. This replaces the previous first-match approach, which collapsed
    T-cell / B-cell lymphomas into the broader "Malignant lymphomas, NOS"
    bucket because that group's keywords were sorted earlier.
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


def _extract_site(text: str) -> str | None:
    """Return the first matching anatomic site category, or None."""
    for site_name, pat in _SITE_PATTERNS:
        if pat.search(text):
            return site_name
    return None


def _build_llm_prompt(original_text: str, candidates: list[TaxonomyLabel]) -> str:
    term_list = "\n".join(f"{i + 1}. {c.term}" for i, c in enumerate(candidates))
    site = _extract_site(original_text)
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


def _parse_llm_response(
    response: str,
    candidates: list[TaxonomyLabel],
    method: str = "LLM",
) -> _MatchResult | None:
    """Match LLM response text back to a taxonomy label. Returns None on no match."""
    response = response.strip()
    if not response or response.lower() == "no match":
        return None
    if response.lower() == "uncertain":
        return _UNCERTAIN_RESULT
    for label in candidates:
        if label.term.lower() == response.lower():
            return _MatchResult(
                term=label.term, group=label.group, code=label.code,
                keyword=response, method=method, confidence=1.0,
            )
    from difflib import get_close_matches
    term_names = [c.term for c in candidates]
    close = get_close_matches(response, term_names, n=1, cutoff=0.8)
    if close:
        label = next(c for c in candidates if c.term == close[0])
        return _MatchResult(
            term=label.term, group=label.group, code=label.code,
            keyword=response, method=method, confidence=0.9,
        )
    return None


def _tier3_llm(
    original_text: str,
    norm_text: str,
    masked_text: str,
    group_token_index: dict[str, set[str]],
    taxonomy_labels: list[TaxonomyLabel],
    oma_index: dict[str, int],
    timeout: int,
    model: str | None,
    counters: dict,
) -> _MatchResult | None:
    # Use masked text for group identification so "non-B cell" / negated
    # mentions don't pull the group selection toward the wrong taxonomy.
    group = _identify_group(masked_text, group_token_index)

    if group:
        candidates = [l for l in taxonomy_labels if l.group == group]
    else:
        # Suffix-based fallback: collect candidates via oma/emia words.
        # Use masked_text so a "granuloma" inside a negated span doesn't
        # surface candidates the LLM has to argue against.
        raw_words = _OMA_RE.findall(masked_text)
        indices = {oma_index[w] for w in set(raw_words) if w in oma_index}
        candidates = [taxonomy_labels[i] for i in sorted(indices)]

    if not candidates:
        return None

    candidates = candidates[:_LLM_MAX_CANDIDATES]
    prompt = _build_llm_prompt(original_text, candidates)

    counters["tier3_calls"] += 1
    try:
        response = chat(prompt, model=model, timeout=timeout)
    except (requests.RequestException, KeyError, ValueError):
        counters["tier3_no_match"] += 1
        return None

    result = _parse_llm_response(response, candidates)
    if result is None:
        counters["tier3_no_match"] += 1
    elif result.method == "Uncertain":
        counters["tier3_uncertain"] += 1
    else:
        counters["tier3_matched"] += 1
    return result


# ---------------------------------------------------------------------------
# Per-row matching
# ---------------------------------------------------------------------------

def _match_diagnosis(
    original_text: str,
    keyword_index: list[tuple[str, re.Pattern, int]],
    taxonomy_labels: list[TaxonomyLabel],
    oma_index: dict[str, int],
    group_token_index: dict[str, set[str]],
    llm_timeout: int,
    llm_model: str | None,
    counters: dict | None = None,
) -> _MatchResult:
    if counters is None:
        counters = {}
    norm_text = _normalize_llm(original_text)
    masked_text = _mask_negation(norm_text)

    result = _tier1_exact(masked_text, keyword_index, taxonomy_labels)
    if result:
        return result

    result = _tier2_fuzzy(masked_text, taxonomy_labels)
    if result:
        return result

    if _has_signal(masked_text):
        counters["signal_rows"] = counters.get("signal_rows", 0) + 1
        result = _tier3_llm(
            original_text, norm_text, masked_text, group_token_index, taxonomy_labels,
            oma_index, llm_timeout, llm_model, counters,
        )
        if result:
            return result

    return _NO_MATCH


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _write_summary_md(summary: dict, path: str) -> None:
    mc = summary["method_counts"]
    total = summary["total_rows"]
    lines = [
        "# LLM Pipeline Summary",
        "",
        "## Overview",
        f"| | |",
        f"|---|---|",
        f"| Total diagnoses | {total:,} |",
        f"| Matched (confirmed) | {summary['matched_rows']:,} ({summary['match_rate_pct']}%) |",
        f"| Uncertain | {summary['uncertain_rows']:,} ({round(100*summary['uncertain_rows']/max(total,1),1)}%) |",
        f"| No match | {mc.get('No Match', mc.get('no_match', 0)):,} ({round(100*mc.get('No Match', mc.get('no_match', 0))/max(total,1),1)}%) |",
        "",
        "## Cases",
        f"| | |",
        f"|---|---|",
        f"| Total unique cases | {summary['total_cases']:,} |",
        f"| Cases with confirmed ICD match | {summary['cases_with_confirmed_match']:,} ({summary['case_match_rate_pct']}%) |",
        f"| Cases with uncertain diagnosis | {summary['cases_with_uncertain']:,} |",
        "",
        "## Method Breakdown",
        "| Method | Count | % of total |",
        "|--------|------:|----------:|",
    ]
    for method, count in sorted(mc.items(), key=lambda x: -x[1]):
        lines.append(f"| {method} | {count:,} | {round(100*count/max(total,1),1)}% |")

    ts = summary.get("tier_stats", {})
    if ts:
        lines += [
            "",
            "## Tier Statistics",
            "| | |",
            "|---|---|",
            f"| Rows with cancer signal (Tier 3 eligible) | {ts.get('signal_rows', 0):,} |",
            f"| Ollama calls made | {ts.get('tier3_calls', 0):,} |",
            f"| Ollama matched | {ts.get('tier3_matched', 0):,} |",
            f"| Ollama uncertain | {ts.get('tier3_uncertain', 0):,} |",
            f"| Ollama no match | {ts.get('tier3_no_match', 0):,} |",
        ]

    top_terms = summary["top_matched_terms"]
    most_matched = (
        f"{next(iter(top_terms))} ({next(iter(top_terms.values()))} matches)"
        if top_terms else "—"
    )
    lines += [
        "",
        "## Taxonomy Coverage",
        f"| | |",
        f"|---|---|",
        f"| Unique terms matched | {summary['unique_terms_matched']} of 846 total |",
        f"| Unique groups matched | {summary['unique_groups_matched']} of 52 total |",
        "",
        "## Imbalance",
        f"| | |",
        f"|---|---|",
        f"| Most matched term | {most_matched} |",
        f"| Terms with only 1 match | {summary['imbalance']['terms_with_1_match']} |",
        f"| Terms with < 5 matches | {summary['imbalance']['terms_with_lt5_matches']} |",
        f"| Terms with ≥ 100 matches | {summary['imbalance']['terms_with_gte100_matches']} |",
        "",
        "## Group Distribution (all 52 groups)",
        "| Group | Count | % of matched |",
        "|-------|------:|-------------:|",
    ]
    for group, vals in sorted(summary["group_distribution"].items(), key=lambda x: -x[1]["count"]):
        lines.append(f"| {group} | {vals['count']:,} | {vals['pct']}% |")

    lines += [
        "",
        "## Top 20 Matched Terms",
        "| Term | Count |",
        "|------|------:|",
    ]
    for term, count in summary["top_matched_terms"].items():
        lines.append(f"| {term} | {count:,} |")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _run_matching_pass(
    df: pd.DataFrame,
    config: LLMConfig,
    taxonomy_labels: list[TaxonomyLabel],
) -> tuple[list[dict], dict]:
    """Run the three-tier match per row; return (result rows, tier counters)."""
    keyword_index = _build_keyword_index(taxonomy_labels)
    oma_index = _build_oma_index(taxonomy_labels)
    group_token_index = _build_group_token_index(taxonomy_labels)

    total = len(df)
    counters: dict = {
        "signal_rows": 0,
        "tier3_calls": 0,
        "tier3_matched": 0,
        "tier3_uncertain": 0,
        "tier3_no_match": 0,
    }
    results: list[dict] = []
    for i, (_, row) in enumerate(df.iterrows(), 1):
        text = str(row[config.text_col]) if pd.notna(row[config.text_col]) else ""
        match = _match_diagnosis(
            text, keyword_index, taxonomy_labels, oma_index, group_token_index,
            config.llm_timeout, config.llm_model,
            counters=counters,
        )

        if match.method == "LLM":
            print(f"[{i}/{total}] {match.method:<6}: {text[:60]!r}  ->  {match.term!r}")
        elif i % 500 == 0:
            print(f"[{i}/{total}] ...")

        entry: dict = {config.id_col: row[config.id_col]}
        if config.diag_num_col in df.columns:
            entry[config.diag_num_col] = row[config.diag_num_col]
        entry.update({
            config.text_col: text,
            "matched_term": match.term,
            "matched_group": match.group,
            "matched_code": match.code,
            "matched_keyword": match.keyword,
            "method": match.method,
            "confidence": match.confidence,
        })
        results.append(entry)

    return results, counters


def _compute_summary(
    out_df: pd.DataFrame,
    counters: dict,
    config: LLMConfig,
    taxonomy_labels: list[TaxonomyLabel],
) -> dict:
    """Aggregate per-row results into the summary dict written to llm_summary.json."""
    total = len(out_df)
    method_counts = out_df["method"].value_counts().to_dict()
    confirmed_df = out_df[~out_df["method"].isin(["No Match", "Uncertain"])]

    total_cases = out_df[config.id_col].nunique()
    confirmed_cases = confirmed_df[config.id_col].nunique()
    uncertain_cases = out_df[out_df["method"] == "Uncertain"][config.id_col].nunique()

    term_counts = confirmed_df["matched_term"].value_counts()
    group_counts = confirmed_df["matched_group"].value_counts()

    return {
        "csv_path": config.csv_path,
        "total_rows": total,
        "matched_rows": len(confirmed_df),
        "uncertain_rows": int((out_df["method"] == "Uncertain").sum()),
        "match_rate_pct": round(100 * len(confirmed_df) / max(total, 1), 1),
        "method_counts": method_counts,
        "tier_stats": counters,
        "total_cases": total_cases,
        "cases_with_confirmed_match": confirmed_cases,
        "cases_with_uncertain": uncertain_cases,
        "case_match_rate_pct": round(100 * confirmed_cases / max(total_cases, 1), 1),
        "unique_terms_matched": int(term_counts.nunique()),
        "unique_groups_matched": int(group_counts.nunique()),
        "imbalance": {
            "top_term_count": int(term_counts.iloc[0]) if len(term_counts) else 0,
            "bottom_term_count": int(term_counts.iloc[-1]) if len(term_counts) else 0,
            "terms_with_1_match": int((term_counts == 1).sum()),
            "terms_with_lt5_matches": int((term_counts < 5).sum()),
            "terms_with_gte100_matches": int((term_counts >= 100).sum()),
        },
        "term_distribution": term_counts.to_dict(),
        "group_distribution": {
            g: {
                "count": int(group_counts.get(g, 0)),
                "pct": round(100 * group_counts.get(g, 0) / max(len(confirmed_df), 1), 2),
            }
            for g in sorted({label.group for label in taxonomy_labels})
        },
        "top_matched_terms": term_counts.head(20).to_dict(),
        "top_matched_groups": group_counts.head(10).to_dict(),
    }


def _write_outputs(out_df: pd.DataFrame, summary: dict, outputs: LLMOutputs) -> None:
    out_df.to_csv(outputs.predictions_csv, index=False)
    with open(outputs.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    _write_summary_md(summary, outputs.summary_md)


def run_llm_scan(config: LLMConfig) -> LLMOutputs:
    """Execute the LLM-assisted diagnosis annotation pipeline."""
    os.makedirs(config.out_dir, exist_ok=True)
    outputs = LLMOutputs(
        predictions_csv=os.path.join(config.out_dir, "llm_annotation.csv"),
        summary_json=os.path.join(config.out_dir, "llm_summary.json"),
        summary_md=os.path.join(config.out_dir, "llm_summary.md"),
    )

    df = pd.read_csv(config.csv_path, encoding="latin-1")
    df.columns = strip_bom_from_columns(df.columns)
    if config.max_rows is not None:
        df = df.head(config.max_rows).copy()

    taxonomy_labels = load_labels_taxonomy(config.labels_csv_path)

    results, counters = _run_matching_pass(df, config, taxonomy_labels)
    out_df = pd.DataFrame(results)
    summary = _compute_summary(out_df, counters, config, taxonomy_labels)
    _write_outputs(out_df, summary, outputs)

    total = summary["total_rows"]
    confirmed_rows = summary["matched_rows"]
    print(f"\nDone. {confirmed_rows}/{total} rows matched ({summary['match_rate_pct']}%)")
    print(
        f"Cases: {summary['cases_with_confirmed_match']}/{summary['total_cases']} "
        f"have a confirmed ICD match ({summary['case_match_rate_pct']}%)"
    )
    top_term_count = summary["imbalance"]["top_term_count"]
    bot_term_count = summary["imbalance"]["bottom_term_count"]
    print(
        f"Unique terms: {summary['unique_terms_matched']} | "
        f"Imbalance: top={top_term_count}, bottom={bot_term_count}, "
        f"single-match terms={summary['imbalance']['terms_with_1_match']}"
    )
    print(f"Method counts: {summary['method_counts']}")
    return outputs
