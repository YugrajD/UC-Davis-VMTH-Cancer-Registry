"""Within-group histologic/topographic subtype keyword filtering.

Applied after behavior-code filtering in the Stage 3 KW correction step.
Narrows the candidate pool for groups where the main "Slightly off" source is
histologic or topographic ambiguity rather than behavior-code mismatch.

Groups covered (ordered by slightly_off rate on Phase 26 test set):
  - Mast cell neoplasms    (leukemia, subcutaneous, systemic/visceral, Kiupel grade)
  - Blood vessel tumors    (hemangiosarcoma vs hemangioma vs hemangioendothelioma)
  - Melanocytoma and Melanomas  (melanocytoma vs melanoma subtypes)
  - Meningiomas            (histologic subtype: meningothelial, fibrous, etc.)
  - Osseous and chondromatous neoplasms  (osteosarcoma vs chondrosarcoma)
  - Gliomas                (glioblastoma vs astrocytoma vs oligodendroglioma …)
"""

import re

# Per group: ordered list of (text_pattern, label_substr) pairs.
# Rules are tried in order; the first pattern that matches the report text AND
# produces a non-empty subset of the pool is applied.  If no rule matches, the
# pool is returned unchanged.
_RULES: dict[str, list[tuple[re.Pattern[str], str]]] = {
    "Mast cell neoplasms": [
        (re.compile(r"\bmast cell leukemia\b", re.I), "leukemia"),
        (re.compile(r"\bsubcutaneous\b", re.I), "Subcutaneous"),
        (re.compile(r"\bvisceral\b", re.I), "isceral"),
        (re.compile(r"\bsystemic\b|\bextracutaneous\b|\bmastocytosis\b", re.I), "ystemic"),
        (re.compile(r"\bkiupel\b.*\bhigh\b|\bhigh\b.*\bkiupel\b", re.I), "Kiupel high"),
        (re.compile(r"\bkiupel\b.*\blow\b|\blow\b.*\bkiupel\b", re.I), "Kiupel low"),
    ],
    "Blood vessel tumors": [
        (re.compile(r"\bhemangiosarcoma\b", re.I), "hemangiosarcoma"),
        (re.compile(r"\bhemangioendothelioma\b", re.I), "hemangioendothelioma"),
        (re.compile(r"\bhemangioma\b", re.I), "hemangioma"),
        (re.compile(r"\bpyogenic granuloma\b", re.I), "granuloma"),
        (re.compile(r"\bangiofibroma\b", re.I), "angiofibroma"),
        (re.compile(r"\bangiokeratoma\b", re.I), "angiokeratoma"),
    ],
    "Melanocytoma and Melanomas": [
        (re.compile(r"\bmelanoac", re.I), "melanoacanthoma"),
        (re.compile(r"\bamelano", re.I), "amelanotic"),
        (re.compile(r"\bsignet[\s-]ring\b", re.I), "signet ring"),
        (re.compile(r"\bballoon cell\b", re.I), "balloon"),
        (re.compile(r"\bclear cell melanoma\b", re.I), "clear cell"),
        (re.compile(r"\bmelanocy", re.I), "melanocytoma"),
        (re.compile(r"\bjunctional\b", re.I), "junctional"),
        (re.compile(r"\bcompound\b", re.I), "compound"),
        (re.compile(r"\bdermal melanoma\b", re.I), "Melanoma, dermal"),
    ],
    "Meningiomas": [
        (re.compile(r"\bmeningothelial\b", re.I), "meningothelial"),
        (re.compile(r"\bpsammomatous\b", re.I), "psammomatous"),
        (re.compile(r"\bmicrocystic meningioma\b", re.I), "microcystic"),
        (re.compile(r"\bsecretory meningioma\b", re.I), "secretory"),
        (re.compile(r"\bangiomatous\b|\bangioblastic meningioma\b", re.I), "angiomatous"),
        (re.compile(r"\bfibroblastic meningioma\b|\bfibrous meningioma\b", re.I), "fibrous"),
        (re.compile(r"\btransitional meningioma\b|\bmixed meningioma\b", re.I), "transitional"),
        (re.compile(r"\bclear cell meningioma\b", re.I), "clear cell"),
        (re.compile(r"\bchordoid meningioma\b", re.I), "chordoid"),
        (re.compile(r"\bpapillary meningioma\b", re.I), "papillary"),
        (re.compile(r"\brhabdoid meningioma\b", re.I), "rhabdoid"),
        (re.compile(r"\batypical meningioma\b|\bwho grade ii\b|\bwho grade 2\b|\bgrade ii meningioma\b", re.I), "atypical"),
    ],
    "Osseous and chondromatous neoplasms": [
        (re.compile(r"\bosteosarcoma\b", re.I), "osteosarcoma"),
        (re.compile(r"\bchondrosarcoma\b", re.I), "chondrosarcoma"),
        (re.compile(r"\bosteochondroma\b", re.I), "osteochondroma"),
        (re.compile(r"\bmultilobular\b", re.I), "multilobular"),
        (re.compile(r"\bchondroma\b", re.I), "chondroma"),
        (re.compile(r"\bosteoma\b", re.I), "osteoma"),
    ],
    "Gliomas": [
        (re.compile(r"\bglioblastoma\b|\bgbm\b", re.I), "glioblastoma"),
        (re.compile(r"\bgliosarcoma\b", re.I), "gliosarcoma"),
        (re.compile(r"\boligoastrocytoma\b", re.I), "oligoastrocytoma"),
        (re.compile(r"\boligodendroglioma\b", re.I), "oligodendroglioma"),
        (re.compile(r"\bependymoma\b", re.I), "ependymoma"),
        (re.compile(r"\bchoroid plexus\b", re.I), "choroid plexus"),
        (re.compile(r"\bmedulloblastoma\b", re.I), "medulloblastoma"),
        (re.compile(r"\bprimitive neuroectodermal\b|\bpnet\b|\bcpnet\b", re.I), "primitive neuroectodermal"),
        (re.compile(r"\bpilocytic astrocytoma\b", re.I), "pilocytic"),
        (re.compile(r"\bpleomorphic xanthoastrocytoma\b|\bpxa\b", re.I), "xanthoastrocytoma"),
        (re.compile(r"\bastrocytoma\b|\bastroglioma\b", re.I), "astrocytoma"),
    ],
}


def filter_by_subtype(
    group_name: str,
    pool: list[int],
    labels: list[str],
    text: str,
) -> list[int]:
    """Narrow pool to labels matching the first subtype keyword found in text.

    Returns pool unchanged if group_name has no rules, no keyword matches the
    report text, or the matched keyword produces an empty label subset.
    """
    rules = _RULES.get(group_name)
    if not rules or len(pool) <= 1:
        return pool
    for pattern, label_substr in rules:
        if pattern.search(text):
            filtered = [j for j in pool if label_substr.lower() in labels[j].lower()]
            if filtered:
                return filtered
    return pool
