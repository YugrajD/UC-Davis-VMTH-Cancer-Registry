"""Tier 1 (exact) and Tier 2 (fuzzy) diagnosis matching, plus the shared
normalization/negation-masking pre-pass and the cascade's result type.

Ported from ``ml/annotation/llm_pipeline/pipeline.py``. The cascade logic, the
``decision_stage`` values and the ``method`` values are unchanged from the old
pipeline: no_signal/No Match, tier1_exact/Exact, tier2_fuzzy/Fuzzy. (Tier 3's
own stages/methods live in ``llm_tier.py``.)

``extract_site`` also lives here: it is a plain text-analysis helper (no LLM
or cleanup specifics), so both ``llm_tier.py`` (prompt enrichment) and
``cleanup.py`` (verification prompt) import it from this shared, neutral home
instead of one module reaching into the other's internals.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from itertools import permutations as _permutations

from taxonomy.taxonomy import TaxonomyLabel

# ---------------------------------------------------------------------------
# Cascade result type + decision_stage values (shared by every tier)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatchResult:
    term: str
    group: str
    code: str
    keyword: str
    method: str        # "Exact" | "Fuzzy" | "LLM" | "No Match" | "Uncertain"
    confidence: float  # 1.0=Exact/LLM, 0.0-1.0=Fuzzy, 0.0=No Match/Uncertain
    stage: str         # which cascade stage decided the row; see STAGES


# Values of the `decision_stage` output column. `method` records only the winning
# tier, so a "No Match" row is ambiguous: the LLM may have been asked and declined,
# or never consulted at all. `stage` disambiguates - "tier3_llm" on a No Match row
# means the model was called and its answer stands.
STAGES = (
    "tier1_exact",          # keyword index matched
    "tier2_fuzzy",          # token-overlap matched
    "tier3_llm",            # LLM was called: matched, uncertain, or declined
    "tier3_no_candidates",  # cancer signal present but candidate build empty; LLM not called
    "no_signal",            # no cancer signal; Tier 3 never reached
)

NO_MATCH = MatchResult(term="", group="", code="", keyword="", method="No Match",
                        confidence=0.0, stage="no_signal")
UNCERTAIN_RESULT = MatchResult(term="", group="", code="", keyword="", method="Uncertain",
                                confidence=0.0, stage="tier3_llm")


# ---------------------------------------------------------------------------
# Text normalization + keyword indexing
# ---------------------------------------------------------------------------

QUALIFIER_RE = re.compile(
    r",?\s*\b("
    r"nos|nec|conventional|well differentiated|spindle cell|kaposiform|"
    r"epithelioid|inflammatory lobular capillary|mixed capillary cavernous|"
    r"retiform|malignant|benign|presumptive|adult type|juvenile type|atypical|"
    r"gist"
    r")\b.*$",
    re.IGNORECASE,
)

OMA_RE = re.compile(r"\b(\w+(?:oma|emia))s?\b")


# Punctuation/whitespace collapse + synonym substitutions used by normalize().
# Named module-level constants (rather than inline re.sub calls) so
# cascade_fingerprint can include them - a change here should invalidate the
# fingerprint just like a change to the keyword index or the thresholds.
_PUNCT_TO_SPACE_RE = re.compile(r"[-_/]")
_DROP_PUNCT_RE = re.compile(r"[,();:]")
_WHITESPACE_RE = re.compile(r"\s+")
_NEOPLASIA_RE = re.compile(r"\bneoplasia\b")
_NEOPLASIA_EXPANSION = "neoplasm"
_PLASMA_CELL_TUMOR_RE = re.compile(r"\bplasma cell tumor\b")
_PLASMA_CELL_TUMOR_EXPANSION = "plasmacytoma"


def normalize(text: str) -> str:
    """Lowercase, collapse punctuation to spaces, normalize whitespace, apply synonyms."""
    text = text.lower()
    text = _PUNCT_TO_SPACE_RE.sub(" ", text)
    text = _DROP_PUNCT_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    text = _NEOPLASIA_RE.sub(_NEOPLASIA_EXPANSION, text)
    text = _PLASMA_CELL_TUMOR_RE.sub(_PLASMA_CELL_TUMOR_EXPANSION, text)
    return text


# Abbreviation expansions and synonym mappings applied after normalization.
ABBREVIATIONS: list[tuple[re.Pattern, str]] = [
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
_METASTASIS_EXPANSION = "metastatic neoplasm"


def normalize_llm(text: str) -> str:
    """Normalize text using the keyword pipeline's rules, then expand abbreviations."""
    text = normalize(text)
    text = _RE_METASTASIS.sub(_METASTASIS_EXPANSION, text)
    for pattern, expansion in ABBREVIATIONS:
        text = pattern.sub(expansion, text)
    return text


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

# "non-X" / "non X" / "non X cell" - masks the negated phrase plus 1-2 tokens.
# Catches "non-B cell", "non-T cell", "non-neoplastic". Limit of 2 trailing
# tokens prevents over-consumption past sentence boundaries.
_NON_X_RE = re.compile(r"\bnon(?:[-\s]\w+){1,2}\b", re.IGNORECASE)


def mask_negation(text: str) -> str:
    """Blank out negated regions so Tier 1/2 keyword matching skips them.

    Tier 3 (LLM) keeps the original text - the prompt has its own negation rules
    and benefits from full context. Negated regions are replaced with spaces, not
    deleted, so token offsets and word boundaries are preserved.
    """
    text = _NEGATION_PHRASE_RE.sub(" ", text)
    text = _NON_X_RE.sub(" ", text)
    return text


def build_keyword_index(taxonomy_labels: list[TaxonomyLabel]) -> list[tuple[str, re.Pattern, int]]:
    """Build (core_keyword, pattern, taxonomy_index) sorted by keyword length descending.

    Two keyword candidates per label: full normalized term and core term with
    qualifiers stripped. Longer keywords are tried first so more specific terms
    take priority. Duplicate keywords are skipped - first label to define a
    keyword wins (Preferred terms appear first in the taxonomy CSV).
    """
    entries: list[tuple[str, re.Pattern, int]] = []
    seen: set[str] = set()
    for i, label in enumerate(taxonomy_labels):
        norm = normalize(label.term)
        core = QUALIFIER_RE.sub("", norm).strip().strip(",").strip()
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


# ---------------------------------------------------------------------------
# Tier 1: Exact match (keyword index)
# ---------------------------------------------------------------------------


def tier1_exact(
    norm_text: str,
    keyword_index: list[tuple[str, re.Pattern, int]],
    taxonomy_labels: list[TaxonomyLabel],
) -> MatchResult | None:
    for kw, pattern, label_idx in keyword_index:
        if pattern.search(norm_text):
            label = taxonomy_labels[label_idx]
            return MatchResult(
                term=label.term, group=label.group, code=label.code,
                keyword=kw, method="Exact", confidence=1.0, stage="tier1_exact",
            )
    return None


# ---------------------------------------------------------------------------
# Tier 2: Fuzzy match (token overlap on core terms, behavior-aware)
# ---------------------------------------------------------------------------

# Fuzzy Tier 2: minimum fraction of core-term tokens that must appear in diagnosis.
FUZZY_THRESHOLD = 0.85
# Behavior-filtered fuzzy: lower threshold because the candidate set is already
# narrowed to ICD-O behavior-matching labels, so false-positive risk is lower.
FUZZY_THRESHOLD_BEHAVIOR = 0.7


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


def core_term(norm_term: str) -> str:
    """Strip qualifier words from a normalized term to get its core tokens."""
    return QUALIFIER_RE.sub("", norm_term).strip().strip(",").strip()


def token_overlap(core: str, norm_text: str) -> float:
    """Fraction of core tokens found in the diagnosis tokens."""
    core_tokens = core.split()
    if len(core_tokens) < 2:
        return 0.0
    text_tokens = set(norm_text.split())
    matched = sum(1 for t in core_tokens if t in text_tokens)
    return matched / len(core_tokens)


def tier2_fuzzy(
    norm_text: str,
    taxonomy_labels: list[TaxonomyLabel],
) -> MatchResult | None:
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

    def _scan(filter_behavior: bool) -> MatchResult | None:
        threshold = FUZZY_THRESHOLD_BEHAVIOR if (filter_behavior and behavior_pref) else FUZZY_THRESHOLD
        best: tuple[float, MatchResult] | None = None
        for label in taxonomy_labels:
            if filter_behavior and behavior_pref:
                digit = _behavior_digit(label.code)
                if digit is not None and digit != behavior_pref:
                    continue
            term_str = normalize(label.term) if filter_behavior and behavior_pref else core_term(normalize(label.term))
            if not term_str:
                continue
            score = token_overlap(term_str, norm_text)
            if score >= threshold:
                if best is None or score > best[0]:
                    best = (score, MatchResult(
                        term=label.term, group=label.group, code=label.code,
                        keyword=term_str, method="Fuzzy", confidence=round(score, 2),
                        stage="tier2_fuzzy",
                    ))
        return best[1] if best else None

    if behavior_pref:
        result = _scan(filter_behavior=True)
        if result:
            return result
    return _scan(filter_behavior=False)


# ---------------------------------------------------------------------------
# Cancer-signal gate (decides no_signal vs proceeding to Tier 3)
# ---------------------------------------------------------------------------

# Signal terms that suggest a neoplastic diagnosis and trigger Tier 3.
SIGNAL_RE = re.compile(
    r"\b(?:"
    r"tumor|tumour|leukemia|leukaemia|neoplasm|cancer|malignancy|malignant|metastatic|"
    r"carcinoid|mycosis|fungoides|polycythemia|paget|mastocytosis|"
    r"ganglioneuromatosis|gliomatosis|lipomatosis|refractory\s+anemia|"
    r"acanthomatous|fibromatosis"
    r")\b",
    re.IGNORECASE,
)


def has_signal(norm_text: str) -> bool:
    """Return True if the diagnosis contains any cancer-indicating term."""
    return bool(OMA_RE.search(norm_text) or SIGNAL_RE.search(norm_text))


# ---------------------------------------------------------------------------
# Anatomic site detection (LLM prompt enrichment - Tier 3 + cleanup)
# ---------------------------------------------------------------------------

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


def extract_site(text: str) -> str | None:
    """Return the first matching anatomic site category, or None."""
    for site_name, pat in _SITE_PATTERNS:
        if pat.search(text):
            return site_name
    return None


# ---------------------------------------------------------------------------
# Cascade fingerprint (for the silver manifest's "cascade version" field)
# ---------------------------------------------------------------------------


def cascade_fingerprint(taxonomy_labels: list[TaxonomyLabel]) -> str:
    """sha256 over the cascade's hardcoded constants plus the taxonomy-derived
    keyword index.

    The manifest's own ``git_sha`` already changes on every commit, including
    ones unrelated to matching behavior, so it can't by itself say whether the
    cascade actually changed between two silver generations. This fingerprint
    only moves when the no_signal/tier1/tier2 rules or the taxonomy they are
    built from actually change.
    """
    payload = {
        "punct_to_space_re": _PUNCT_TO_SPACE_RE.pattern,
        "drop_punct_re": _DROP_PUNCT_RE.pattern,
        "whitespace_re": _WHITESPACE_RE.pattern,
        "neoplasia_re": [_NEOPLASIA_RE.pattern, _NEOPLASIA_EXPANSION],
        "plasma_cell_tumor_re": [_PLASMA_CELL_TUMOR_RE.pattern, _PLASMA_CELL_TUMOR_EXPANSION],
        "metastasis_re": [_RE_METASTASIS.pattern, _METASTASIS_EXPANSION],
        "abbreviations": [[p.pattern, expansion] for p, expansion in ABBREVIATIONS],
        "qualifier_re": QUALIFIER_RE.pattern,
        "signal_re": SIGNAL_RE.pattern,
        "negation_phrase_re": _NEGATION_PHRASE_RE.pattern,
        "non_x_re": _NON_X_RE.pattern,
        "oma_re": OMA_RE.pattern,
        "fuzzy_threshold": FUZZY_THRESHOLD,
        "fuzzy_threshold_behavior": FUZZY_THRESHOLD_BEHAVIOR,
        "keyword_index": [[kw, idx] for kw, _, idx in build_keyword_index(taxonomy_labels)],
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()
