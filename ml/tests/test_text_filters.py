"""Tests for PetBERT post-processing text gates."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


def _load_text_filters():
    path = Path(__file__).resolve().parents[1] / "production" / "petbert_pipeline" / "text_filters.py"
    spec = importlib.util.spec_from_file_location("petbert_text_filters_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


text_filters = _load_text_filters()


class NonNeoplasticGateTests(unittest.TestCase):
    def assert_not_suppressed(self, final_comment: str, ancillary_tests: str = "") -> None:
        self.assertFalse(
            text_filters.looks_non_neoplastic(final_comment, "", ancillary_tests),
            msg=final_comment,
        )

    def assert_suppressed(self, final_comment: str, ancillary_tests: str = "") -> None:
        self.assertTrue(
            text_filters.looks_non_neoplastic(final_comment, "", ancillary_tests),
            msg=final_comment,
        )

    def test_final_comment_tumor_evidence_vetoes_suppression(self) -> None:
        examples = [
            "The dermal mass is consistent with a cavernous hemangioma with mild inflammation.",
            "The masses are consistent with cutaneous hemangiosarcomas and solar dermatitis.",
            "A pheochromocytoma is present; concurrent hepatitis is mild.",
            "The intestinal mass is an adenocarcinoma with secondary ulceration and inflammation.",
            "Sections contain neoplastic mast cells consistent with mast cell tumor.",
            "The bone lesion is a chondrosarcoma with regional necrosis.",
            "This is a benign tumor and appears completely excised.",
            "An adrenal cortical adenoma is incidental to the cystitis.",
            "A cystadenoma is incidental and unrelated to the pneumonia.",
            "The diagnosis is peripheral nerve sheath tumor; cellulitis is secondary.",
        ]
        for final_comment in examples:
            with self.subTest(final_comment=final_comment):
                self.assertTrue(text_filters.final_comment_has_tumor_evidence(final_comment))
                self.assert_not_suppressed(final_comment)

    def test_ancillary_positive_tumor_evidence_vetoes_suppression(self) -> None:
        non_neoplastic_fc = "The primary lesion is chronic inflammation with abscessation."
        examples = [
            "Desmin and pan-muscle actin show strong immunoreactivity of neoplastic spindle cells.",
            "Olig-2 positive neoplastic cells are present at the edge of the lesion.",
            "Insulin supports diagnosis of insulinoma.",
            "S100 supports peripheral nerve sheath tumor.",
            "Uroplakin supports urothelial origin.",
            "Tumor cells are immunoreactive for CD31.",
            "Immunohistochemistry confirms diagnosis of rhabdomyosarcoma.",
        ]
        for ancillary in examples:
            with self.subTest(ancillary=ancillary):
                self.assertTrue(text_filters.ancillary_tests_support_neoplasia(ancillary))
                self.assert_not_suppressed(non_neoplastic_fc, ancillary)

    def test_ancillary_negative_or_ruleout_language_does_not_veto(self) -> None:
        non_neoplastic_fc = "The lung lesion is fungal pyogranulomatous pneumonia."
        examples = [
            "PANCK is negative in the lesion.",
            "There is no immunoreactivity for CD3 or CD20.",
            "The stain does not support carcinoma.",
            "No evidence of neoplastic cells is identified.",
            "Only the internal positive control stained appropriately.",
            "GMS and PAS stains are negative for fungal organisms.",
            "Immunohistochemistry is pending to rule out melanoma.",
        ]
        for ancillary in examples:
            with self.subTest(ancillary=ancillary):
                self.assertFalse(text_filters.ancillary_tests_support_neoplasia(ancillary))
                self.assert_suppressed(non_neoplastic_fc, ancillary)

    def test_clear_non_neoplastic_cases_still_suppress(self) -> None:
        examples = [
            "The findings are consistent with aspiration pneumonia and sepsis.",
            "The main lesion is marked necrotizing pancreatitis.",
            "The lung contains fungal pyogranulomatous pneumonia.",
            "The spinal cord lesions are due to disk herniation and myelomalacia.",
            "The submitted tissue is a benign cyst with epithelial hyperplasia and inflammation.",
            "There is no evidence of neoplasia in the examined sections.",
        ]
        for final_comment in examples:
            with self.subTest(final_comment=final_comment):
                self.assertFalse(text_filters.final_comment_has_tumor_evidence(final_comment))
                self.assert_suppressed(final_comment)


class LowConfidenceRescueTests(unittest.TestCase):
    def assert_rescued(self, label_term: str, source_text: str) -> None:
        self.assertTrue(
            text_filters.low_confidence_label_supported_by_text(label_term, source_text),
            msg=f"{label_term}: {source_text}",
        )

    def assert_not_rescued(self, label_term: str, source_text: str) -> None:
        self.assertFalse(
            text_filters.low_confidence_label_supported_by_text(label_term, source_text),
            msg=f"{label_term}: {source_text}",
        )

    def test_rescues_only_text_supported_specific_tumors(self) -> None:
        examples = [
            ("Mast cell tumor, NOS", "Final comment: cutaneous mast cell tumor, completely excised."),
            ("Hemangiosarcoma, NOS", "Final comment: splenic hemangiosarcoma with hemorrhage."),
            ("Adenocarcinoma, NOS", "The intestinal mass is an adenocarcinoma."),
            ("Osteosarcoma, NOS", "The bone lesion is consistent with osteosarcoma."),
            ("Lymphosarcoma", "The mass is consistent with lymphoma."),
            ("Peripheral nerve sheath tumor, NOS", "S100 supports peripheral nerve sheath tumor."),
            ("Rhabdomyosarcoma, NOS", "IHC supports skeletal muscle origin of the neoplasm."),
        ]
        for label_term, source_text in examples:
            with self.subTest(label_term=label_term):
                self.assert_rescued(label_term, source_text)

    def test_does_not_rescue_generic_or_negated_tumor_language(self) -> None:
        examples = [
            ("Neoplasm, malignant", "There is a mass-like area of chronic inflammation."),
            ("Round cell tumor, NOS", "Round cells are reactive; no evidence of neoplasia."),
            ("Lymphosarcoma", "Immunohistochemistry is pending to rule out lymphoma."),
            ("Adenoma, NOS", "The submitted tissue is a cyst with epithelial hyperplasia."),
            ("Acute megakaryoblastic leukemia", "There is no evidence of leukemia or neoplasia."),
        ]
        for label_term, source_text in examples:
            with self.subTest(label_term=label_term):
                self.assert_not_rescued(label_term, source_text)


if __name__ == "__main__":
    unittest.main()
