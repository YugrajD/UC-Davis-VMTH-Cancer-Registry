"""Focused tests for PetBERT per-case categorization gate behavior."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ML_ROOT = Path(__file__).resolve().parents[1]
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from ICD_labels.taxonomy import TaxonomyLabel
from production.petbert_pipeline.stages import categorize_per_case


def test_gate_rejected_case_becomes_uncategorized_low_confidence():
    result = categorize_per_case(
        texts=["malignant mast cell tumor"],
        mean_embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
        lp_embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
        label_embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
        taxonomy_labels=[
            TaxonomyLabel(code="9740/3", group="Mast cell neoplasms", term="Mast cell tumor, NOS"),
        ],
        labels=["Mast cell tumor, NOS"],
        group_probs=np.array([[0.0]], dtype=np.float32),
        group_names=["Mast cell neoplasms"],
        threshold=0.3,
        max_predictions=2,
        presence_mask=np.array([False]),
        fallback_to_argmax=True,
    )

    assert result.final_labels == ["Uncategorized"]
    assert result.methods == ["low_confidence"]
    assert result.top_k_indices == [[-1]]
    assert result.top_k_methods == [["low_confidence"]]


def test_gate_passed_argmax_fallback_does_not_become_uncategorized():
    result = categorize_per_case(
        texts=["malignant tumor"],
        mean_embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
        lp_embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
        label_embeddings=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        taxonomy_labels=[
            TaxonomyLabel(code="8000/3", group="Test group", term="Cancer A"),
            TaxonomyLabel(code="8001/3", group="Other group", term="Cancer B"),
        ],
        labels=["Cancer A", "Cancer B"],
        group_probs=np.array([[0.2, 0.1]], dtype=np.float32),
        group_names=["Test group", "Other group"],
        threshold=0.3,
        max_predictions=2,
        presence_mask=np.array([True]),
        fallback_to_argmax=True,
    )

    assert result.final_labels == ["Cancer A"]
    assert result.methods == ["embedding"]
    assert result.top_k_indices == [[0]]
    assert result.top_k_methods == [["embedding"]]
