import re

DEFAULT_LABELS = [
    "Lymphoma",
    "Mast Cell Tumor",
    "Osteosarcoma",
    "Hemangiosarcoma",
    "Melanoma",
    "Squamous Cell Carcinoma",
    "Fibrosarcoma",
    "Transitional Cell Carcinoma",
]

LABEL_ANCHORS = {
    "Lymphoma": re.compile(r"\blymphoma\b|\blymphoid\b", re.IGNORECASE),
    "Mast Cell Tumor": re.compile(r"\bmast cell\b|\bmct\b", re.IGNORECASE),
    "Osteosarcoma": re.compile(r"\bosteosarcoma\b|\bosteoid\b", re.IGNORECASE),
    "Hemangiosarcoma": re.compile(r"\bhemangiosarcoma\b", re.IGNORECASE),
    "Melanoma": re.compile(r"\bmelanoma\b|\bmelanocyt", re.IGNORECASE),
    "Squamous Cell Carcinoma": re.compile(r"\bsquamous\b|\bscc\b", re.IGNORECASE),
    "Fibrosarcoma": re.compile(r"\bfibrosarcoma\b", re.IGNORECASE),
    "Transitional Cell Carcinoma": re.compile(
        r"\btransitional cell\b|\btcc\b|\burothelial\b", re.IGNORECASE
    ),
}

