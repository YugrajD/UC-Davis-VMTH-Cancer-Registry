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
    r"\bpneumonia\b", r"\bbronchopneumonia\b",
    r"\bleiomyositi", r"\bpancreatiti", r"\bhepatiti", r"\bdermatiti",
    r"\bcholangiti", r"\bnephriti", r"\bencephaliti", r"\bmyositi",
    r"\bcystiti", r"\bgastriti", r"\bcolitis", r"\bperitoniti",
    r"\bcellulitis\b", r"\bpyogranulomatous\b",
    r"\b(?:fungal|bacterial|mycotic|protozoal)\s+(?:infection|pneumonia|dermatitis|inflammation)\b",
    r"no evidence of neoplas",
    r"\bnon[- ]neoplastic\b",
    r"\bnon[- ]?tumorous\b",
    r"consistent with\s+(?:\w+\s+){0,4}inflammat",
    r"inflammatory process",
    r"hepatic (?:injury|dysfunction|necrosis|hepatopathy)",
    r"degenerative (?:process|change|lesion)",
    r"\b(?:disk|disc)\s+herniation\b",
    r"\bmyelomalacia\b",
    r"\bcongenital\b", r"\bdevelopmental\b",
    r"reactive\s+(?:process|change|lesion|hyperplasi)",
    r"\bgranulation tissue\b",
    r"\babscess\b",
]
_NON_NEOPLASTIC_RE = re.compile("|".join(_NON_NEOPLASTIC_PATTERNS), re.IGNORECASE)

_DIRECT_TUMOR_TERMS = (
    r"neoplasia|neoplasm|tumou?r|carcinoma|adenocarcinoma|"
    r"cystadenocarcinoma|sarcoma|osteosarcoma|chondrosarcoma|"
    r"hemangiosarcoma|lymphoma|lymphosarcoma|mast cell tumou?r|"
    r"melanoma|meningioma|glioma|oligodendroglioma|ependymoma|"
    r"choroid plexus tumou?r|adenoma|cystadenoma|hemangioma|"
    r"pheochromocytoma|insulinoma|glucagonoma|neuroendocrine tumou?r|"
    r"carcinoid|plasmacytoma|plasma cell tumou?r|schwannoma|"
    r"peripheral nerve sheath tumou?r|neurofibroma|trichoblastoma|"
    r"trichoepithelioma|rhabdomyosarcoma|leiomyosarcoma|"
    r"histiocytic sarcoma|fibrosarcoma|lipoma"
)

_DIRECT_TUMOR_EVIDENCE_RE = re.compile(
    r"\b(?:" + _DIRECT_TUMOR_TERMS + r")(?:s|es)?\b|"
    r"(?<!non-)(?<!non )\bneoplastic\s+(?:cells?|population|mass|tissue|infiltrat)",
    re.IGNORECASE,
)

_NEGATED_TUMOR_SPAN_RE = re.compile(
    r"\b(?:no|without|absence of|free of)\s+(?:overt\s+)?(?:evidence of\s+)?"
    r"(?:\w+\s+){0,5}(?:" + _DIRECT_TUMOR_TERMS + r"|neoplastic\s+cells?)\b|"
    r"\b(?:negative for|not consistent with|not diagnostic for|does not support|"
    r"do not support|rule(?:s|d)? out|ruled out|exclude(?:s|d)?)\s+"
    r"(?:\w+\s+){0,5}(?:" + _DIRECT_TUMOR_TERMS + r"|neoplastic\s+cells?)\b|"
    r"\b(?:" + _DIRECT_TUMOR_TERMS + r"|neoplastic\s+cells?)\s+"
    r"(?:\w+\s+){0,4}(?:not\s+(?:seen|observed|identified|present)|absent)\b|"
    r"\bnon[- ]neoplastic\b",
    re.IGNORECASE,
)


def has_non_neoplastic_primary_diagnosis(final_comment: str) -> bool:
    """True when FINAL COMMENT contains a strong non-neoplastic diagnosis cue."""
    return bool(_NON_NEOPLASTIC_RE.search(final_comment or ""))


def _text_has_direct_tumor_evidence(text: str) -> bool:
    """True when text names tumor/neoplasia evidence outside negated spans."""
    if not text:
        return False
    cleaned = _NEGATED_TUMOR_SPAN_RE.sub(" ", text)
    return bool(_DIRECT_TUMOR_EVIDENCE_RE.search(cleaned))


def final_comment_has_tumor_evidence(final_comment: str) -> bool:
    """True when FINAL COMMENT directly names tumor/neoplasia evidence.

    Incidental, benign, unrelated, or completely excised tumors still count:
    for Vet-ICD-O extraction they remain neoplastic diagnoses and should veto
    suppression to Non-neoplastic.
    """
    return _text_has_direct_tumor_evidence(final_comment or "")


_ANCILLARY_MARKER_POSITIVE_RE = re.compile(
    r"\b(positive|positivity|(?<!non-)immunoreactive|immunolabel(?:ed|ing)?|"
    r"(?<!non-)immunoreactivity|label(?:ed|ing)|stain(?:ed|ing)?|"
    r"express(?:es|ed|ion)|expression|support(?:s|ed|ing)?|"
    r"confirm(?:s|ed|ing)?|consistent with|diagnostic for)\b",
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

_ANCILLARY_NEGATED_SPAN_RE = re.compile(
    r"[^.;\n\r|]{0,40}\b(?:negative|no immunoreactivity|not immunoreactive|"
    r"non[- ]immunoreactive|non[- ]specific|failed to reveal|does not support|"
    r"do not support|pending|inconclusive|none performed|not performed|"
    r"rule(?:s|d)? out|ruled out|rule-out|exclude(?:s|d)?|without evidence|"
    r"no evidence)\b[^.;\n\r|]{0,120}",
    re.IGNORECASE,
)

_ANCILLARY_TUMOR_EVIDENCE_RE = re.compile(
    r"|".join(
        [
            # "neoplastic cells are positive/immunoreactive for ..."
            r"\b(?:neoplastic|tumou?r)\s+(?:cells?|population|lymphocytes?|"
            r"epithelial cells?|endothelial cells?|melanocytes?|spindle cells?)\b.{0,100}"
            r"\b(?:positive|positivity|immunoreactive|immunolabel(?:ed|ing)?|"
            r"immunoreactivity|label(?:ed|ing)|stain(?:ed|ing)?|express(?:es|ed|ion)|"
            r"expression)\b",
            # "... positive/immunoreactive ... neoplastic/tumor cells"
            r"\b(?:positive|positivity|immunoreactive|immunolabel(?:ed|ing)?|"
            r"immunoreactivity|label(?:ed|ing)|stain(?:ed|ing)?|express(?:es|ed|ion)|"
            r"expression)\b.{0,100}"
            r"\b(?:neoplastic|tumou?r)\s+(?:cells?|population|lymphocytes?|"
            r"epithelial cells?|endothelial cells?|melanocytes?|spindle cells?)\b",
            # Marker-positive neoplastic cells, e.g. "CD3-positive neoplastic lymphocytes".
            r"\b[A-Z0-9][A-Za-z0-9/\- ]{1,25}[- ]positive\b.{0,80}"
            r"\b(?:neoplastic|tumou?r)\s+(?:cells?|population|lymphocytes?|"
            r"epithelial cells?|endothelial cells?|melanocytes?|spindle cells?)\b",
            # "IHC supports/confirms diagnosis of insulinoma / PNST / etc."
            r"\b(?:support(?:s|ed|ing)?|confirm(?:s|ed|ing)?|consistent with|"
            r"diagnostic for)\b.{0,140}\b(?:" + _DIRECT_TUMOR_TERMS + r"|"
            r"urothelial origin|skeletal muscle origin)\b",
            # "insulin/S100/uroplakin supports diagnosis/origin ..."
            r"\b(?:insulin|s100|uroplakin|olig-?2|desmin|pan[- ]?muscle actin|"
            r"cd3|cd20|cd31|factor viii|melan[- ]?a)\b.{0,100}"
            r"\b(?:support(?:s|ed|ing)?|confirm(?:s|ed|ing)?|consistent with|"
            r"diagnostic for)\b.{0,140}\b(?:" + _DIRECT_TUMOR_TERMS + r"|"
            r"urothelial origin|skeletal muscle origin)\b",
        ]
    ),
    re.IGNORECASE,
)


def ancillary_tests_support_neoplasia(ancillary_tests: str) -> bool:
    """True when ancillary tests provide strong, positive tumor evidence.

    This is intentionally conservative. Marker names alone do not count; the
    evidence must connect positive/supportive marker language to neoplastic or
    tumor cells, a tumor diagnosis, or tumor lineage/origin. Negated/rule-out
    spans are ignored so negative stains, nonspecific staining, pending tests,
    and infection-only stains do not veto non-neoplastic suppression.
    """
    text = ancillary_tests or ""
    if not text.strip():
        return False
    cleaned = _ANCILLARY_NEGATED_SPAN_RE.sub(" ", text)
    if not _ANCILLARY_MARKER_POSITIVE_RE.search(cleaned):
        return False
    return bool(_ANCILLARY_TUMOR_EVIDENCE_RE.search(cleaned))


def looks_non_neoplastic(
    final_comment: str,
    hp_summary: str = "",
    ancillary_tests: str = "",
) -> bool:
    """Heuristic: True if the report's primary diagnosis appears non-neoplastic.

    Logic: a strong non-neoplastic indicator must be present in FINAL COMMENT,
    and direct tumor evidence must be absent from FINAL COMMENT and ANCILLARY
    TESTS. HP summary remains a soft veto for explicit tumor morphology.
    """
    fc = final_comment or ""
    hp = hp_summary or ""
    if not has_non_neoplastic_primary_diagnosis(fc):
        return False
    # Veto if FINAL COMMENT names any neoplasm, including incidental/benign
    # tumors. Vet-ICD-O extraction should keep those diagnoses.
    if final_comment_has_tumor_evidence(fc):
        return False
    # Soft veto from HP summary: if HP mentions a tumor explicitly we keep
    # the prediction. (HP often says "neoplastic cells infiltrate ..." even
    # when FINAL COMMENT focuses on the inflammatory secondary process.)
    if _text_has_direct_tumor_evidence(hp):
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
