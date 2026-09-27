"""Tests for coding.rule: the vagueness table."""

from __future__ import annotations

import pandas as pd
import pytest

from coding.rule import DECISIVE, VAGUE, UnknownDecisionError, case_is_vague, row_is_decisive, row_outcome

# The real (decision_stage, method) pairs the cascade produces
# (icd-mapping-strategy.md "Coding a case"; ml-rewrite-plan.md Decisions).
REAL_PAIRS = [
    ("no_signal", "No Match", DECISIVE),
    ("tier1_exact", "Exact", DECISIVE),
    ("tier2_fuzzy", "Fuzzy", DECISIVE),
    ("tier3_llm", "LLM", DECISIVE),
    ("tier3_llm", "No Match", DECISIVE),
    ("tier3_llm", "Uncertain", VAGUE),
    ("tier3_no_candidates", "No Match", VAGUE),
]

# The four extra pairs diagnosis_mapping.cleanup's ensemble verification pass
# can produce on top of the cascade's own seven (WP9 fix 1): it rewrites a
# confirmed Exact/Fuzzy row's method to No Match/Uncertain but never touches
# decision_stage.
CLEANUP_PAIRS = [
    ("tier1_exact", "No Match", DECISIVE),
    ("tier1_exact", "Uncertain", VAGUE),
    ("tier2_fuzzy", "No Match", DECISIVE),
    ("tier2_fuzzy", "Uncertain", VAGUE),
]


@pytest.mark.parametrize("stage,method,expected", REAL_PAIRS)
def test_row_outcome_real_pairs(stage, method, expected):
    assert row_outcome(stage, method) == expected
    assert row_is_decisive(stage, method) == (expected == DECISIVE)


@pytest.mark.parametrize("stage,method,expected", CLEANUP_PAIRS)
def test_row_outcome_cleanup_pairs(stage, method, expected):
    assert row_outcome(stage, method) == expected
    assert row_is_decisive(stage, method) == (expected == DECISIVE)


@pytest.mark.parametrize("stage,method", [
    ("tier1_exact", "Fuzzy"),       # a real stage paired with the wrong method
    ("tier2_fuzzy", "Exact"),
    ("made_up_stage", "Exact"),
    ("no_signal", "Exact"),
    ("tier3_no_candidates", "Uncertain"),  # cleanup never touches tier3_no_candidates rows
])
def test_unknown_pair_raises(stage, method):
    with pytest.raises(UnknownDecisionError):
        row_outcome(stage, method)


def test_case_is_vague_false_when_every_row_decisive():
    rows = pd.DataFrame({
        "decision_stage": ["no_signal", "tier1_exact", "tier3_llm"],
        "method": ["No Match", "Exact", "No Match"],
    })
    assert case_is_vague(rows) is False


def test_case_is_vague_true_when_any_row_vague():
    rows = pd.DataFrame({
        "decision_stage": ["tier1_exact", "tier3_llm"],
        "method": ["Exact", "Uncertain"],
    })
    assert case_is_vague(rows) is True


def test_case_is_vague_raises_on_unknown_row():
    # tier1_exact/Uncertain is now a real (cleanup) pair — use one still outside the table.
    rows = pd.DataFrame({"decision_stage": ["tier1_exact"], "method": ["Fuzzy"]})
    with pytest.raises(UnknownDecisionError):
        case_is_vague(rows)
