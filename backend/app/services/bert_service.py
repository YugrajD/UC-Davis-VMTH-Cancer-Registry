"""
VetBERT cancer type classifier.
Uses a keyword-based mock classifier for development,
with an optional real BERT model for production.
"""

import re
from typing import List, Dict
from app.schemas.schemas import ClassifyResult


# Keyword patterns for each cancer type, ordered by specificity
CANCER_PATTERNS = {
    "Lymphoma": [
        r"lymphoma", r"lymphoid", r"lymphocyte", r"lymphoblastic",
        r"B-cell", r"T-cell", r"CD20", r"CD3", r"CD79a",
        r"multicentric", r"lymph node", r"immunoblastic",
    ],
    "Mast Cell Tumor": [
        r"mast cell", r"metachromatic", r"granul(e|ar)", r"Patnaik",
        r"Kiupel", r"toluidine blue", r"c-KIT", r"Ki-67",
    ],
    "Osteosarcoma": [
        r"osteosarcoma", r"osteoid", r"osteoblast", r"bone.?forming",
        r"alkaline phosphatase", r"appendicular",
    ],
    "Hemangiosarcoma": [
        r"hemangiosarcoma", r"vascular", r"endothelial",
        r"CD31", r"vWF", r"Factor VIII", r"erythrophagocytosis",
        r"angiosarcoma",
    ],
    "Melanoma": [
        r"melanoma", r"melanocyt", r"melanin", r"Melan-A", r"PNL2",
        r"S-100", r"amelanotic", r"pigment",
    ],
    "Squamous Cell Carcinoma": [
        r"squamous", r"keratin pearl", r"keratinization",
        r"intercellular bridge", r"solar elastosis", r"SCC",
    ],
    "Fibrosarcoma": [
        r"fibrosarcoma", r"spindle cell", r"herringbone",
        r"collagen(ous)?", r"vimentin", r"mesenchymal",
    ],
    "Transitional Cell Carcinoma": [
        r"transitional cell", r"urothelial", r"bladder",
        r"uroplakin", r"trigone", r"TCC", r"papillary.*transitional",
    ],
}


class BertClassifier:
    """Mock BERT classifier using keyword matching for development."""

    def __init__(self):
        self.patterns = {
            cancer: [re.compile(p, re.IGNORECASE) for p in patterns]
            for cancer, patterns in CANCER_PATTERNS.items()
        }

    def classify(self, text: str) -> ClassifyResult:
        scores: Dict[str, float] = {}

        for cancer_type, compiled_patterns in self.patterns.items():
            score = 0.0
            for pattern in compiled_patterns:
                matches = pattern.findall(text)
                score += len(matches) * (1.0 / len(compiled_patterns))
            scores[cancer_type] = score

        total_score = sum(scores.values())
        if total_score == 0:
            # Default with low confidence
            return ClassifyResult(
                predicted_cancer_type="Unknown",
                confidence=0.1,
                top_predictions=[
                    {"cancer_type": ct, "confidence": round(1.0 / len(scores), 4)}
                    for ct in list(scores.keys())[:3]
                ],
            )

        # Normalize to probabilities
        probabilities = {
            ct: round(score / total_score, 4) for ct, score in scores.items()
        }

        # Sort by probability
        sorted_predictions = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        predicted = sorted_predictions[0]

        return ClassifyResult(
            predicted_cancer_type=predicted[0],
            confidence=predicted[1],
            top_predictions=[
                {"cancer_type": ct, "confidence": conf}
                for ct, conf in sorted_predictions[:5]
            ],
        )
