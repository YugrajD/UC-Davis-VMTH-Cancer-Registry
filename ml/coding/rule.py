"""The vagueness table: which diagnosis-mapping rows are decisive vs vague.

Read from ``decision_stage`` (which tier of the cascade produced the row) and
``method`` (what that tier — or a later cleanup pass — actually did), per
icd-mapping-strategy.md's "Coding a case" table and ml-rewrite-plan.md's
Decisions:

- ``no_signal`` / "No Match" — decisive: no cancer vocabulary at all.
- ``tier1_exact`` / "Exact" — decisive: keyword index matched.
- ``tier2_fuzzy`` / "Fuzzy" — decisive, *provisionally* (settled by the 1.1
  Tier-3 audit; no switch has been thrown yet, so it stays decisive here).
- ``tier3_llm`` / "LLM" — decisive: the LLM answered a code.
- ``tier3_llm`` / "No Match" — decisive non-cancer: a declined LLM answer
  ("Declined LLM answers (tier3_llm 'No Match') are decisive non-cancer. No
  switch." — ml-rewrite-plan.md Decisions).
- ``tier3_llm`` / "Uncertain" — vague: the LLM hedged.
- ``tier3_no_candidates`` / "No Match" — vague: cancer vocabulary was present,
  but the LLM was never asked.

``diagnosis_mapping/cleanup.py``'s ensemble verification pass
(``silver.run(cleanup_enabled=True)``) rewrites a confirmed Exact/Fuzzy/LLM
row's ``method`` to "No Match" or "Uncertain" when the verification models
overturn or can't confirm it — but it never touches ``decision_stage`` (see
``cleanup._apply_resolution`` / ``_apply_uncertain``), so it can produce four
more pairs the cascade alone never would:

- ``tier1_exact`` / "No Match" and ``tier2_fuzzy`` / "No Match" — cleanup
  decided the row is non-cancer: decisive non-cancer, same as any other
  "No Match" (cleanup's resolution rules give one definite answer or
  ``Uncertain``, never a separate lower-confidence non-cancer outcome).
- ``tier1_exact`` / "Uncertain" and ``tier2_fuzzy`` / "Uncertain" — cleanup
  couldn't confirm the row (the verification models disagreed, or a
  ``WRONG_should_be`` term didn't resolve to a taxonomy label): vague, same
  as any other "Uncertain".

A cleanup pass never touches a ``tier3_llm``/``tier3_no_candidates`` row
(their "No Match" rows are outside cleanup's ``_CONFIRMED_METHODS``) and never
produces a pair beyond the four above, so together with the cascade's own
seven pairs this is the complete set below.

These are the only ``(decision_stage, method)`` pairs the cascade
(``diagnosis_mapping.keyword_tiers`` / ``llm_tier``) and its optional cleanup
pass (``diagnosis_mapping.cleanup``) can produce. Any other pair is a bug or a
cascade/cleanup change this table hasn't caught up with, so it raises rather
than silently defaulting either way.
"""

from __future__ import annotations

import pandas as pd

DECISIVE = "decisive"
VAGUE = "vague"

_OUTCOME_OF_PAIR: dict[tuple[str, str], str] = {
    ("no_signal", "No Match"): DECISIVE,
    ("tier1_exact", "Exact"): DECISIVE,
    ("tier1_exact", "No Match"): DECISIVE,   # cleanup overturned an Exact match to non-cancer
    ("tier1_exact", "Uncertain"): VAGUE,     # cleanup couldn't confirm an Exact match
    ("tier2_fuzzy", "Fuzzy"): DECISIVE,
    ("tier2_fuzzy", "No Match"): DECISIVE,   # cleanup overturned a Fuzzy match to non-cancer
    ("tier2_fuzzy", "Uncertain"): VAGUE,     # cleanup couldn't confirm a Fuzzy match
    ("tier3_llm", "LLM"): DECISIVE,
    ("tier3_llm", "No Match"): DECISIVE,
    ("tier3_llm", "Uncertain"): VAGUE,
    ("tier3_no_candidates", "No Match"): VAGUE,
}


class UnknownDecisionError(ValueError):
    """A ``(decision_stage, method)`` pair outside the vagueness table above."""


def row_outcome(decision_stage: str, method: str) -> str:
    """``DECISIVE`` or ``VAGUE`` for one diagnosis row.

    Raises ``UnknownDecisionError`` on any pair not in the table above — never
    defaults to either outcome, per the WP9 brief ("An unknown pair must
    raise, not default.").
    """
    try:
        return _OUTCOME_OF_PAIR[(decision_stage, method)]
    except KeyError:
        raise UnknownDecisionError(
            f"unrecognized (decision_stage, method) pair: ({decision_stage!r}, {method!r})"
        ) from None


def row_is_decisive(decision_stage: str, method: str) -> bool:
    return row_outcome(decision_stage, method) == DECISIVE


def case_is_vague(rows: pd.DataFrame) -> bool:
    """True if ANY of a case's diagnosis rows is vague.

    icd-mapping-strategy.md: "If any row of a case is vague, the whole case is
    queued." Expects a ``decision_stage``/``method`` column pair (e.g. one
    case's slice of a silver annotation table); raises on any row whose pair
    is not in the vagueness table.
    """
    return any(
        row_outcome(stage, method) == VAGUE
        for stage, method in zip(rows["decision_stage"], rows["method"])
    )
