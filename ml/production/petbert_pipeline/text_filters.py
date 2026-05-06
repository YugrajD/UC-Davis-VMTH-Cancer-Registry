"""Pure text-level filters used by the rule-based gates.

Three functions, all stateless and side-effect free:

    strip_tissue_lists(text)
        Drop necropsy "tissues examined" list sentences from a report so the
        embedder sees the diagnostic prose instead of organ names. No-op when
        the report is a normal biopsy narrative.

    looks_non_neoplastic(final_comment, hp_summary)
        Heuristic: report's primary diagnosis is inflammation / hyperplasia /
        cyst / degeneration with no competing neoplastic diagnosis.

    qualifier_words_missing_from_text(predicted_term, source_text)
        For multi-word predicted terms (e.g., "Microcystic meningioma"),
        return the qualifier tokens that don't appear in the source text.
        Used to demote hallucinated subtypes to the NOS variant.

These functions encode rules derived from manual spot-checks of the model on
40 historical cases. See ml/documentation/fallback-model-notes.md for the
per-case evidence.
"""

from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# 1. Tissue-list filter
# ---------------------------------------------------------------------------

# Necropsy reports often start the HP summary with this exact prefix and
# follow it with a list of "(T1) organ, organ; (T2) organ, ..." segments
# that contains no diagnostic prose. The embedder picks up organ words
# instead of the diagnosis (which lives in FINAL COMMENT).
_TISSUE_LIST_HEADER = re.compile(
    r"the following tissues? (?:were|are)\s+examined(?:\s+microscopically)?\s*[:.]?",
    re.IGNORECASE,
)

# A "(T1)" or "(T1-T3)" or "T1:" marker — repeated occurrences of these with
# only short comma-separated tokens between them indicate a list segment.
_TISSUE_MARKER = re.compile(r"\(?\bT\d+(?:\s*-\s*T?\d+)?\b\)?\s*[:.]?", re.IGNORECASE)

# Verbs/nouns that indicate the sentence describes morphology rather than
# just listing tissues. If a candidate list sentence contains any of these,
# we keep it.
_NARRATIVE_TOKENS = re.compile(
    r"\b(examined|composed|consist(?:s|ing|ed)?|demarcat|"
    r"infiltrat|effac|express|mass|tumor|tumour|neoplas|"
    r"cells?|nucle|mitot|cytoplasm|stroma|epitheli|mesenchym|"
    r"hyperplas|carcinoma|sarcoma|adenoma|lymphoma|melanoma|"
    r"differentiat|grade|inflam|necrosis|metastas)",
    re.IGNORECASE,
)


def _is_tissue_list_sentence(sentence: str) -> bool:
    """True if the sentence is a list of tissues with no diagnostic content."""
    s = sentence.strip()
    if not s:
        return False
    # Must look list-shaped: at least one (T#) marker AND multiple commas/semicolons.
    has_marker = bool(_TISSUE_MARKER.search(s))
    delimiter_count = s.count(",") + s.count(";")
    if not (has_marker and delimiter_count >= 2):
        # Header-only sentence ("The following tissues were examined microscopically:")
        # also counts even without markers if it's short and matches the header.
        if _TISSUE_LIST_HEADER.fullmatch(s.rstrip(":. ")):
            return True
        return False
    # If the sentence contains diagnostic prose, keep it (e.g. "(T1-T3) Right
    # cranial mammary mass. Examined are several sections of mammary gland in
    # which there are multiple..." — that's a real biopsy narrative).
    if _NARRATIVE_TOKENS.search(s):
        return False
    return True


def strip_tissue_lists(text: str) -> str:
    """Remove tissue-list-only sentences. Returns possibly-empty string.

    The filter is conservative: only drops sentences that match the necropsy
    "list of tissues examined" pattern with no diagnostic prose. Biopsy reports
    (which describe morphology in the same paragraph) are left untouched.
    """
    if not text:
        return text

    # Find the necropsy header and split: everything up to the next narrative
    # sentence is treated as a list block to drop.
    cleaned = _TISSUE_LIST_HEADER.sub(" ", text)

    # Now split on sentence-ending punctuation. We use a simple split that
    # preserves the delimiter so we can re-join cleanly.
    parts = re.split(r"(?<=[.;])\s+", cleaned)
    kept = [p for p in parts if not _is_tissue_list_sentence(p)]
    out = " ".join(kept).strip()
    # Collapse whitespace introduced by removals.
    out = re.sub(r"\s+", " ", out)
    return out


# ---------------------------------------------------------------------------
# 2. Non-neoplastic detector
# ---------------------------------------------------------------------------

# Phrases that strongly indicate the *primary* diagnosis is not a neoplasm.
# Each is a regex matched case-insensitively. Patterns are deliberately
# conservative: they require the non-neoplastic term to appear as a
# diagnostic conclusion, not just incidentally.
_NON_NEOPLASTIC_PATTERNS = [
    r"consistent with\s+(?:\w+\s+){0,3}hyperplasi",
    r"\bis\s+hyperplasi",
    r"diagnos(?:is|ed)\s+(?:of\s+)?hyperplasi",
    r"hyperplastic process",
    r"hyperplastic nodul",
    r"\bis\s+a\s+(?:benign\s+)?(?:radicular|odontogenic|dermal|epidermal|sebaceous|follicular|epidermoid)?\s*cyst",
    r"consistent with\s+(?:\w+\s+){0,4}cyst\b",
    r"compatible with\s+(?:a\s+)?(?:radicular|odontogenic|dermal|sebaceous)?\s*cyst",
    r"\bleiomyositi", r"\bpancreatiti", r"\bhepatiti", r"\bdermatiti",
    r"\bcholangiti", r"\bnephriti", r"\bencephaliti", r"\bmyositi",
    r"\bcystiti", r"\bgastriti", r"\bcolitis", r"\bperitoniti",
    r"no evidence of neoplas",
    r"\bnon[- ]neoplastic\b",
    r"\bnon[- ]?tumorous\b",
    r"consistent with\s+(?:\w+\s+){0,4}inflammat",
    r"inflammatory process",
    r"hepatic (?:injury|dysfunction|necrosis|hepatopathy)",
    r"degenerative (?:process|change|lesion)",
    r"reactive\s+(?:process|change|lesion|hyperplasi)",
    r"\bgranulation tissue\b",
    r"\babscess\b",
]
_NON_NEOPLASTIC_RE = re.compile("|".join(_NON_NEOPLASTIC_PATTERNS), re.IGNORECASE)

# Phrases that indicate a competing neoplastic primary diagnosis. If any of
# these match, we do NOT suppress — pathologist is calling a tumor and a
# concurrent inflammatory process.
_NEOPLASTIC_PATTERNS = [
    r"consistent with\s+(?:\w+\s+){0,4}(?:carcinoma|sarcoma|adenoma|lymphoma|melanoma|tumou?r|neoplas|leukemia|myeloma|mesothelioma|blastoma|cytoma)\b",
    r"diagnos(?:is|ed)\s+(?:of\s+)?(?:\w+\s+){0,4}(?:carcinoma|sarcoma|adenoma|lymphoma|melanoma|tumou?r|neoplas|leukemia|myeloma|mesothelioma|blastoma|cytoma)\b",
    r"\bis\s+(?:a\s+)?(?:malignant\s+|benign\s+)?(?:carcinoma|sarcoma|adenoma|lymphoma|melanoma|neoplas|leukemia|myeloma|mesothelioma|blastoma|cytoma)\b",
    r"\bmalignant\s+(?:lymphoma|tumou?r|neoplas)",
    r"neoplastic (?:cells?|infiltrat|process|tissue|population|mass)",
    r"\bcell tumor\b",
    r"\bmast cell tumor\b",
    r"\bround cell tumor\b",
    r"\b(?:carcinoma|sarcoma|adenoma|lymphoma|melanoma|hepatoma|osteosarcoma|hemangiosarcoma|lipoma|fibroma|leiomyoma|leiomyosarcoma|fibrosarcoma|mast cell tumor|meningioma|glioma)\b",
]
_NEOPLASTIC_RE = re.compile("|".join(_NEOPLASTIC_PATTERNS), re.IGNORECASE)


_ANCILLARY_MARKER_POSITIVE_RE = re.compile(
    r"\b(positive|positivity|immunoreactive|immunolabel(?:ed|ing)?|"
    r"immunoreactivity|label(?:ed|ing)|stain(?:ed|ing)?|express(?:es|ed|ion)|"
    r"expression)\b",
    re.IGNORECASE,
)

_ANCILLARY_NEGATIVE_RE = re.compile(
    r"\b(negative|no immunoreactivity|not immunoreactive|"
    r"non[- ]immunoreactive|non[- ]specific|"
    r"failed to reveal|does not support|do not support|pending|inconclusive|"
    r"none performed|not performed|rule(?:s|d)? out|ruled out|rule-out|"
    r"exclude(?:s|d)?|without evidence|no evidence)\b",
    re.IGNORECASE,
)

_ANCILLARY_TUMOR_EVIDENCE_RE = re.compile(
    r"|".join(
        [
            # "neoplastic cells are positive/immunoreactive for ..."
            r"\b(?:neoplastic|tumou?r)\s+(?:cells?|population|lymphocytes?|"
            r"epithelial cells?|endothelial cells?|melanocytes?)\b.{0,100}"
            r"\b(?:positive|positivity|immunoreactive|immunolabel(?:ed|ing)?|"
            r"immunoreactivity|label(?:ed|ing)|stain(?:ed|ing)?|express(?:es|ed|ion)|"
            r"expression)\b",
            # "... positive/immunoreactive ... neoplastic/tumor cells"
            r"\b(?:positive|positivity|immunoreactive|immunolabel(?:ed|ing)?|"
            r"immunoreactivity|label(?:ed|ing)|stain(?:ed|ing)?|express(?:es|ed|ion)|"
            r"expression)\b.{0,100}"
            r"\b(?:neoplastic|tumou?r)\s+(?:cells?|population|lymphocytes?|"
            r"epithelial cells?|endothelial cells?|melanocytes?)\b",
            # Marker-positive neoplastic cells, e.g. "CD3-positive neoplastic lymphocytes".
            r"\b[A-Z0-9][A-Za-z0-9/\- ]{1,25}[- ]positive\b.{0,80}"
            r"\b(?:neoplastic|tumou?r)\s+(?:cells?|population|lymphocytes?|"
            r"epithelial cells?|endothelial cells?|melanocytes?)\b",
        ]
    ),
    re.IGNORECASE,
)


def ancillary_tests_support_neoplasia(ancillary_tests: str) -> bool:
    """True when ancillary tests provide strong, positive tumor evidence.

    This is intentionally conservative. Marker names alone do not count; the
    evidence must connect positive marker language directly to neoplastic/tumor
    cells. If the same ancillary section contains negation, rule-out, pending,
    or inconclusive language, it does not veto non-neoplastic suppression.
    """
    text = ancillary_tests or ""
    if not text.strip():
        return False
    if _ANCILLARY_NEGATIVE_RE.search(text):
        return False
    if not _ANCILLARY_MARKER_POSITIVE_RE.search(text):
        return False
    return bool(_ANCILLARY_TUMOR_EVIDENCE_RE.search(text))


def looks_non_neoplastic(
    final_comment: str,
    hp_summary: str = "",
    ancillary_tests: str = "",
) -> bool:
    """Heuristic: True if the report's primary diagnosis appears non-neoplastic.

    Logic: a non-neoplastic indicator must be present in FINAL COMMENT, AND
    no competing neoplastic primary indicator may be present. FINAL COMMENT and
    HP summary are primary evidence; ANCILLARY TESTS is used only as a narrow
    tumor-evidence veto so it can prevent unsafe suppression without creating a
    new tumor diagnosis.
    """
    fc = final_comment or ""
    hp = hp_summary or ""
    if not _NON_NEOPLASTIC_RE.search(fc):
        return False
    # Veto if the report also names a tumor type in FINAL COMMENT.
    if _NEOPLASTIC_RE.search(fc):
        return False
    # Soft veto from HP summary: if HP mentions a tumor explicitly we keep
    # the prediction. (HP often says "neoplastic cells infiltrate ..." even
    # when FINAL COMMENT focuses on the inflammatory secondary process.)
    if _NEOPLASTIC_RE.search(hp):
        return False
    # Ancillary tests can veto suppression when they strongly support a tumor,
    # but marker names alone are deliberately ignored.
    if ancillary_tests_support_neoplasia(ancillary_tests):
        return False
    return True


# ---------------------------------------------------------------------------
# 3. Subtype-qualifier check
# ---------------------------------------------------------------------------

# Words that frequently appear as anatomic/structural qualifiers in tumor
# names AND that the model has been observed to hallucinate. When the
# predicted term contains one of these but the source text does not, the
# prediction is demoted to the group's NOS variant.
#
# Each entry is the canonical qualifier; the matcher accepts the listed
# inflectional variants too.
_QUALIFIER_VARIANTS: dict[str, tuple[str, ...]] = {
    "subcutaneous": ("subcutaneous", "subcut"),
    "cutaneous": ("cutaneous",),
    "dermal": ("dermal", "dermis"),
    "intramuscular": ("intramuscular", "intra-muscular"),
    "infiltrative": ("infiltrative", "infiltrating", "infiltrate", "infiltration"),
    "microcystic": ("microcystic",),
    "transitional": ("transitional",),
    "meningothelial": ("meningothelial",),
    "psammomatous": ("psammomatous",),
    "fibrous": ("fibrous",),
    "angiomatous": ("angiomatous",),
    "papillary": ("papillary", "papilliform"),
    "surface": ("surface",),
    "periosteal": ("periosteal",),
    "parosteal": ("parosteal",),
    "extraskeletal": ("extraskeletal", "extra-skeletal"),
    "multilobular": ("multilobular", "multi-lobular"),
    "telangiectatic": ("telangiectatic",),
    "epitheliotropic": ("epitheliotropic",),
    "non-epitheliotropic": ("non-epitheliotropic", "nonepitheliotropic"),
    "follicular": ("follicular",),
    "marginal": ("marginal",),
    "centrocytic": ("centrocytic",),
    "centroblastic": ("centroblastic",),
    "lymphoblastic": ("lymphoblastic",),
    "anaplastic": ("anaplastic",),
    "undifferentiated": ("undifferentiated",),
    "spindle": ("spindle",),
    "round": ("round",),
    "giant": ("giant",),
    "clear": ("clear",),
    "small": ("small",),
    "large": ("large",),
    # NOTE: do not include "hepatoid" — it is a synonym for perianal/circumanal
    # gland, not a hallucinated qualifier. "Hepatoid gland adenoma" should pass.
}


def qualifier_words_missing_from_text(predicted_term: str, source_text: str) -> list[str]:
    """Return qualifier tokens in ``predicted_term`` that are absent from ``source_text``.

    Tokens are matched case-insensitively against the variant list above.
    Returns an empty list if every recognized qualifier in the predicted term
    has a corresponding mention (or variant) in the source.
    """
    if not predicted_term or not source_text:
        return []
    src = source_text.lower()
    missing: list[str] = []
    for word in re.findall(r"[A-Za-z][A-Za-z\-]+", predicted_term):
        canonical = word.lower()
        if canonical not in _QUALIFIER_VARIANTS:
            continue
        variants = _QUALIFIER_VARIANTS[canonical]
        if not any(re.search(rf"\b{re.escape(v)}", src) for v in variants):
            missing.append(canonical)
    return missing
