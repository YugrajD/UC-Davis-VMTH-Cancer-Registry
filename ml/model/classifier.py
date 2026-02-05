"""
VetBERT Cancer Type Classifier

In production, this would load a fine-tuned BERT model for veterinary pathology
report classification. For development, it uses keyword-based matching that
mimics BERT inference behavior.

Production usage would be:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("./vetbert-finetuned")
"""

import re
from typing import Dict, List, Tuple


CANCER_LABELS = [
    "Lymphoma",
    "Mast Cell Tumor",
    "Osteosarcoma",
    "Hemangiosarcoma",
    "Melanoma",
    "Squamous Cell Carcinoma",
    "Fibrosarcoma",
    "Transitional Cell Carcinoma",
]

# Weighted keyword patterns per cancer type
KEYWORD_WEIGHTS: Dict[str, List[Tuple[str, float]]] = {
    "Lymphoma": [
        (r"lymphoma", 3.0), (r"lymphoid", 2.0), (r"lymphocyte", 2.0),
        (r"B-cell|T-cell", 2.5), (r"CD20|CD3|CD79a|CD4", 2.0),
        (r"multicentric", 1.5), (r"lymph node", 1.0),
        (r"immunoblastic|lymphoblastic", 2.0),
    ],
    "Mast Cell Tumor": [
        (r"mast cell", 3.0), (r"metachromatic", 2.5), (r"granul", 1.5),
        (r"Patnaik|Kiupel", 2.5), (r"toluidine blue", 2.0),
        (r"c-KIT", 2.0), (r"Ki-67", 1.5),
    ],
    "Osteosarcoma": [
        (r"osteosarcoma", 3.0), (r"osteoid", 2.5), (r"osteoblast", 2.0),
        (r"bone.?forming", 2.0), (r"alkaline phosphatase", 1.5),
        (r"appendicular", 1.5),
    ],
    "Hemangiosarcoma": [
        (r"hemangiosarcoma", 3.0), (r"vascular neoplasm", 2.5),
        (r"endothelial", 2.0), (r"CD31", 2.5), (r"vWF|Factor VIII", 2.0),
        (r"erythrophagocytosis", 2.0), (r"spleen|splenic", 1.0),
    ],
    "Melanoma": [
        (r"melanoma", 3.0), (r"melanocyt", 2.5), (r"melanin", 2.0),
        (r"Melan-A|PNL2", 2.5), (r"S-100", 2.0), (r"amelanotic", 2.0),
        (r"pigment", 1.0),
    ],
    "Squamous Cell Carcinoma": [
        (r"squamous", 3.0), (r"keratin pearl", 2.5),
        (r"keratinization", 2.0), (r"intercellular bridge", 2.0),
        (r"solar elastosis", 1.5), (r"\bSCC\b", 3.0),
    ],
    "Fibrosarcoma": [
        (r"fibrosarcoma", 3.0), (r"spindle cell", 2.0),
        (r"herringbone", 2.5), (r"collagen", 1.5),
        (r"vimentin", 2.0), (r"mesenchymal", 1.0),
    ],
    "Transitional Cell Carcinoma": [
        (r"transitional cell", 3.0), (r"urothelial", 2.5),
        (r"bladder", 2.0), (r"uroplakin", 2.5),
        (r"trigone", 2.0), (r"\bTCC\b", 3.0),
    ],
}


class VetBERTClassifier:
    """Mock VetBERT classifier using weighted keyword matching."""

    def __init__(self):
        self.compiled_patterns = {
            cancer: [(re.compile(pattern, re.IGNORECASE), weight)
                     for pattern, weight in patterns]
            for cancer, patterns in KEYWORD_WEIGHTS.items()
        }

    def predict(self, text: str) -> Dict:
        scores: Dict[str, float] = {}

        for cancer_type, patterns in self.compiled_patterns.items():
            score = 0.0
            for pattern, weight in patterns:
                matches = pattern.findall(text)
                score += len(matches) * weight
            scores[cancer_type] = score

        total = sum(scores.values())
        if total == 0:
            probs = {ct: 1.0 / len(CANCER_LABELS) for ct in CANCER_LABELS}
        else:
            probs = {ct: score / total for ct, score in scores.items()}

        sorted_preds = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        predicted = sorted_preds[0]

        return {
            "predicted_label": predicted[0],
            "confidence": round(predicted[1], 4),
            "all_probabilities": {ct: round(p, 4) for ct, p in sorted_preds},
        }


if __name__ == "__main__":
    classifier = VetBERTClassifier()

    test_texts = [
        "Histopathology reveals diffuse large B-cell lymphoma in the submandibular lymph node",
        "Excisional biopsy reveals dermal mast cell tumor, Patnaik grade II, with clean margins",
        "Core biopsy from distal radius shows osteoid-producing malignant osteoblasts consistent with osteosarcoma",
        "Splenectomy specimen with hemangiosarcoma, CD31 positive, Factor VIII positive",
    ]

    for text in test_texts:
        result = classifier.predict(text)
        print(f"\nText: {text[:80]}...")
        print(f"  Predicted: {result['predicted_label']} ({result['confidence']:.1%})")
