"""New keyword tiers (Tier 1/2 + the no_signal gate) against the pre-rewrite
``ml/annotation/llm_pipeline/pipeline.py`` cascade.

LEGACY COMPARISON — deleted at cutover (WP13), when the old ml/ tree
(including ml/annotation/) is removed. Everything below that imports
``annotation.llm_pipeline.pipeline`` exists only to prove new/old parity.

Runs against the full real taxonomy (labels.csv is public and safe to read in
full) so the built keyword index is identical to production; every diagnosis
string below is invented for this test file - none is real patient data.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# Old ml/annotation/ is outside the ml/next/ import root (pytest.ini's
# `pythonpath = .` only adds ml/next); reach it the same way
# tests/test_taxonomy.py does, since this file is deleted at cutover anyway.
_ML_DIR = Path(__file__).resolve().parents[2]
if str(_ML_DIR) not in sys.path:
    sys.path.append(str(_ML_DIR))

from ICD_labels.taxonomy import load_labels_taxonomy as old_load_labels_taxonomy
from annotation.llm_pipeline.pipeline import (
    _build_keyword_index as old_build_keyword_index,
    _has_signal as old_has_signal,
    _mask_negation as old_mask_negation,
    _normalize_llm as old_normalize_llm,
    _tier1_exact as old_tier1_exact,
    _tier2_fuzzy as old_tier2_fuzzy,
)

import config
from taxonomy.taxonomy import load_labels_taxonomy
from diagnosis_mapping.keyword_tiers import (
    build_keyword_index,
    has_signal,
    mask_negation,
    normalize_llm,
    tier1_exact,
    tier2_fuzzy,
)
from diagnosis_mapping.llm_tier import build_llm_prompt


def _load_legacy_pipeline():
    """Load ml/annotation/llm_pipeline/pipeline.py by file path, mirroring
    test_verdicts.py's _load_legacy_evaluate.

    pipeline.py's own top-level imports (ICD_labels, annotation.llm_pipeline.client,
    utils.csv_io) need ml/ on sys.path for the duration of the exec only.
    """
    def load(name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        # dataclasses' type-hint resolution looks the module up by name in
        # sys.modules; a module loaded via spec_from_file_location isn't
        # registered there by default.
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    saved_path = list(sys.path)
    sys.path.insert(0, str(_ML_DIR))
    try:
        return load("legacy_llm_pipeline", _ML_DIR / "annotation" / "llm_pipeline" / "pipeline.py")
    finally:
        sys.path[:] = saved_path
        del sys.modules["legacy_llm_pipeline"]


old_build_llm_prompt = _load_legacy_pipeline()._build_llm_prompt

OLD_LABELS_CSV = str(_ML_DIR / "ICD_labels" / "labels.csv")
NEW_LABELS_CSV = str(config.LABELS_CSV)

# A spread of invented diagnoses covering: Tier 1 (single- and multi-word
# core, permuted word order, abbreviation expansion), Tier 2 (scattered core
# tokens that can't hit Tier 1's contiguous-phrase match), negated / hedged
# text with no cancer signal at all, and a masked non-X compound.
SYNTHETIC_DIAGNOSES = [
    "MAST CELL TUMOR",
    "Hemangiosarcoma of the spleen",
    "No evidence of neoplasia in the biopsy",
    "Suspicious growth of uncertain type",
    "Osteosarcoma of the distal radius, margins incomplete",
    "GIST of the small intestine",
    "MCT grade II, mitotic index low",
    "Benign lipoma of the subcutis",
    "Metastatic carcinoma to lymph node",
    "rule out lymphoma, cannot confirm on cytology",
    "non-B cell lymphoma suspected",
    "chronic otitis externa, no neoplastic cells seen",
    "possible mammary gland adenocarcinoma",
    "melanoma versus melanocytoma, uncertain",
    "malignant tumor with small cell morphology, type indeterminate",
]


@pytest.fixture(scope="module")
def taxonomies():
    old_labels = old_load_labels_taxonomy(OLD_LABELS_CSV)
    new_labels = load_labels_taxonomy(NEW_LABELS_CSV)
    assert [(l.code, l.group, l.term) for l in old_labels] == [(l.code, l.group, l.term) for l in new_labels]
    return old_labels, new_labels, old_build_keyword_index(old_labels), build_keyword_index(new_labels)


@pytest.mark.parametrize("text", SYNTHETIC_DIAGNOSES)
def test_keyword_tiers_match_old_pipeline(text, taxonomies):
    old_labels, new_labels, old_kw_index, new_kw_index = taxonomies

    old_masked = old_mask_negation(old_normalize_llm(text))
    new_masked = mask_negation(normalize_llm(text))
    assert new_masked == old_masked

    old_result = old_tier1_exact(old_masked, old_kw_index, old_labels) or old_tier2_fuzzy(old_masked, old_labels)
    new_result = tier1_exact(new_masked, new_kw_index, new_labels) or tier2_fuzzy(new_masked, new_labels)

    if old_result is None:
        assert new_result is None
        assert has_signal(new_masked) == old_has_signal(old_masked)
    else:
        assert new_result is not None
        assert (new_result.code, new_result.method, new_result.stage) == \
            (old_result.code, old_result.method, old_result.stage)


# ---------------------------------------------------------------------------
# Tier 3 prompt: must be byte-identical to the old pipeline's, including the
# literal em dashes in the negation/hedging rule lines.
# ---------------------------------------------------------------------------

# (diagnosis text, number of leading real-taxonomy labels used as candidates)
# - a site keyword ("subcutaneous"/"oral mucosa") to exercise the site-hint
#   branch, and one plain diagnosis with no site keyword at all.
PROMPT_TEST_CASES = [
    ("Mass in the subcutaneous tissue of the flank", 3),
    ("A diagnosis of uncertain histologic origin", 5),
    ("Nodule in the oral mucosa, uncertain behavior", 1),
]


@pytest.mark.parametrize("text,n_candidates", PROMPT_TEST_CASES)
def test_build_llm_prompt_byte_identical_to_old(text, n_candidates, taxonomies):
    _old_labels, new_labels, _old_kw_index, _new_kw_index = taxonomies
    candidates = new_labels[:n_candidates]
    assert build_llm_prompt(text, candidates) == old_build_llm_prompt(text, candidates)
