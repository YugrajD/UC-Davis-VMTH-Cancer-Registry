"""Keyword-based ICD-O behavior code scoring for within-group term selection.

The ICD-O behavior digit (after '/') encodes tumor behavior:
  0 = benign, 1 = borderline/uncertain, 2 = in situ,
  3 = malignant, 6 = metastatic, 9 = uncertain primary vs metastatic

This module maps report text to the most likely behavior digit using
weighted keyword matching, enabling Stage 2 term disambiguation after
a group has been selected in Stage 1.
"""

import re

# (keyword, weight) pairs per behavior digit.
# Weight 1.0 = strong signal; 0.5 = weaker/contextual signal.
BEHAVIOR_KEYWORDS: dict[str, list[tuple[str, float]]] = {
    "0": [  # Benign
        ("benign", 1.0),
        ("adenoma", 1.0),
        ("papilloma", 1.0),
        ("fibroma", 1.0),
        ("lipoma", 1.0),
        ("osteoma", 1.0),
        ("hemangioma", 1.0),
        ("leiomyoma", 1.0),
        ("myoma", 1.0),
        ("neurofibroma", 1.0),
        ("schwannoma", 1.0),
        ("trichoblastoma", 1.0),
        ("hamartoma", 1.0),
        ("well-differentiated", 0.5),
        ("well differentiated", 0.5),
        ("encapsulated", 0.5),
        ("non-invasive", 0.5),
        ("noninvasive", 0.5),
        ("low-grade", 0.5),
        ("low grade", 0.5),
    ],
    "1": [  # Borderline / Uncertain malignant potential
        ("uncertain", 1.0),
        ("borderline", 1.0),
        ("low malignant potential", 1.0),
        ("uncertain malignant potential", 1.0),
        ("uncertain biological behavior", 1.0),
        ("indeterminate", 1.0),
        ("intermediate", 0.5),
        ("atypical", 0.5),
        ("cannot be determined", 0.5),
        ("cannot exclude", 0.5),
        ("equivocal", 0.5),
    ],
    "2": [  # In situ
        ("in situ", 1.0),
        ("intraepithelial", 1.0),
        ("preinvasive", 1.0),
        ("pre-invasive", 1.0),
        ("carcinoma in situ", 1.0),
        ("noninvasive carcinoma", 1.0),
        ("non-invasive carcinoma", 1.0),
    ],
    "3": [  # Malignant
        ("malignant", 1.0),
        ("carcinoma", 1.0),
        ("sarcoma", 1.0),
        ("adenocarcinoma", 1.0),
        ("lymphoma", 1.0),
        ("leukemia", 1.0),
        ("invasive", 1.0),
        ("poorly differentiated", 1.0),
        ("undifferentiated", 1.0),
        ("anaplastic", 1.0),
        ("high-grade", 1.0),
        ("high grade", 1.0),
        ("infiltrating", 0.5),
        ("infiltrative", 0.5),
        ("destructive", 0.5),
        ("necrosis", 0.5),
        ("vascular invasion", 0.5),
        ("mitotic", 0.5),
    ],
    "6": [  # Metastatic
        ("metastatic", 1.0),
        ("metastasis", 1.0),
        ("metastases", 1.0),
        ("secondary", 1.0),
        ("distant metastasis", 1.0),
        ("regional metastasis", 1.0),
        ("lymph node metastasis", 1.0),
        ("pulmonary metastasis", 1.0),
        ("hepatic metastasis", 1.0),
        ("splenic metastasis", 1.0),
        ("mets", 0.5),
        ("disseminated", 0.5),
        ("stage iv", 0.5),
        ("stage 4", 0.5),
    ],
    "9": [  # Uncertain whether primary or metastatic
        ("uncertain whether primary or metastatic", 1.0),
        ("unknown primary", 1.0),
        ("primary vs metastatic", 1.0),
        ("cannot determine primary", 1.0),
        ("possible primary", 0.5),
    ],
}

# Pre-compile patterns once at import time.
_PATTERNS: dict[str, list[tuple[re.Pattern[str], float]]] = {
    digit: [(re.compile(r"\b" + re.escape(kw) + r"\b"), w) for kw, w in kws]
    for digit, kws in BEHAVIOR_KEYWORDS.items()
}


def score_behavior(text: str) -> dict[str, float]:
    """Return per-behavior-digit keyword scores for *text*.

    Only digits with a non-zero score are included in the result.
    """
    lower = text.lower()
    scores: dict[str, float] = {}
    for digit, patterns in _PATTERNS.items():
        total = sum(w for pat, w in patterns if pat.search(lower))
        if total > 0.0:
            scores[digit] = total
    return scores


def best_behavior(text: str) -> str | None:
    """Return the best-matching behavior digit, or *None* if no signal.

    Tie-break rule: /6 (metastatic) supersedes /3 (malignant) when both
    score >= 1.0, because "metastatic" is a more specific descriptor.
    """
    scores = score_behavior(text)
    if not scores:
        return None
    winner = max(scores, key=lambda d: scores[d])
    if winner == "3" and scores.get("6", 0.0) >= 1.0:
        winner = "6"
    return winner
