"""Score cancer label predictions against expected labels, one verdict per row.

Exact port of the legacy per-code scorer (``ml/evaluation/evaluate.py``).
Each predicted label receives one of six verdicts:

  good           — predicted term exactly matches a verified label for this case
  slightly_off   — correct cancer group, wrong specific term
  completely_off — neither term nor group matches any verified label for this case
  false_positive — model made a positive prediction for a case with no verified labels
  false_negative — case has verified labels but model predicted "Non-Cancer" (or
                   "Uncategorized" for legacy CSVs), or case has no prediction row at all
  true_negative  — model correctly predicted "Non-Cancer" / "Uncategorized" for a
                   non-cancer case (excluded from the verdict table and from metrics)

``score()`` returns the rows the legacy ``evaluate()`` wrote to evaluation.csv, in
the same order, with the same ``verdict`` / ``expected_term`` / ``expected_group``
values. It keeps every prediction column (legacy dropped ``method`` and
``predicted_code`` from the CSV), so compare on the legacy columns.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import pandas as pd

# Terms written to predictions.csv for gate-rejected / non-cancer cases.
# "Non-Cancer" is the current value; "Uncategorized" is the legacy name kept
# here so evaluation scripts handle both old and new CSVs correctly.
NON_CANCER_PRED_TERMS = frozenset({"Non-Cancer", "Uncategorized"})

VERDICTS = ("good", "slightly_off", "completely_off", "false_positive", "false_negative")
GOOD = ("good",)
GOOD_PLUS_SLIGHT = ("good", "slightly_off")


def score_prediction(predicted_term: str, predicted_group: str,
                     matched_terms: set[str], matched_groups: set[str],
                     uncommon_groups: frozenset[str] = frozenset()) -> str:
    """Score one petbert prediction against the case's keyword label sets."""
    if not matched_terms:
        # Model correctly abstained on a non-cancer case → true negative, not FP.
        if predicted_term in NON_CANCER_PRED_TERMS:
            return "true_negative"
        return "false_positive"
    # Model abstained on a confirmed cancer case → false negative, not completely_off.
    if predicted_term in NON_CANCER_PRED_TERMS:
        return "false_negative"
    if predicted_term in matched_terms:
        return "good"
    if predicted_group in matched_groups:
        return "slightly_off"
    # Predicting any uncommon group for a case whose true group is genuinely uncommon counts
    # as slightly_off — the model got the right tier, just not the specific term.
    # This covers both: predicted_group == "Uncommon" (training bucket) and any specific
    # uncommon group name resolved by keyword matching at inference time.
    if uncommon_groups and matched_groups & uncommon_groups and \
            (predicted_group == "Uncommon" or predicted_group in uncommon_groups):
        return "slightly_off"
    return "completely_off"


def _covers_group(predicted_group: str, expected_group: str | None,
                  uncommon_groups: frozenset[str]) -> bool:
    """Mirror the slightly_off rule from score_prediction() for one expected group."""
    if predicted_group == expected_group:
        return True
    if uncommon_groups and expected_group in uncommon_groups and \
            (predicted_group == "Uncommon" or predicted_group in uncommon_groups):
        return True
    return False


def _records(df: pd.DataFrame, key_columns: list[str]) -> list[dict]:
    # Legacy read everything with csv.DictReader (all str, empty = ""); match that
    # for the columns the scorer looks at, leave the rest untouched.
    df = df.copy()
    for column in key_columns:
        df[column] = df[column].fillna("").astype(str)
    return df.to_dict("records")


def _case_sort_key(row: dict):
    suffix = row["case_id"].split("-")[-1]
    return int(suffix) if suffix.isdigit() else row["case_id"]


def score(expectations: pd.DataFrame, predictions: pd.DataFrame,
          uncommon_groups: Iterable[str]) -> pd.DataFrame:
    """Per-row verdict table, identical to legacy evaluate()'s evaluation.csv rows.

    expectations: ``case_id, matched_term, matched_group`` (any extra columns
        ignored); an empty/blank term means no cancer. Pass string case_ids.
    predictions: legacy prediction columns (``case_id, predicted_term,
        predicted_group`` are read; the rest are carried through).
    uncommon_groups: groups merged into "Uncommon" (empty disables the rule).

    Filter both tables to the evaluated cases before calling (legacy --test-cases).
    Returns the prediction columns + ``expected_term, expected_group, verdict``;
    true negatives are excluded.
    """
    uncommon_groups = frozenset(uncommon_groups)
    kw_rows = _records(expectations, ["case_id", "matched_term", "matched_group"])
    pb_rows = _records(predictions, ["case_id", "predicted_term", "predicted_group"])

    # Build per-case label sets from keyword predictions
    case_terms: dict[str, set[str]] = defaultdict(set)
    case_groups: dict[str, set[str]] = defaultdict(set)
    # Global, last-seen-wins term → group map (legacy evaluate.py:82-87).
    term_to_group: dict[str, str] = {}
    for row in kw_rows:
        if row["matched_term"].strip():
            case_terms[row["case_id"]].add(row["matched_term"])
            if row["matched_group"].strip():
                term_to_group[row["matched_term"]] = row["matched_group"]
        if row["matched_group"].strip():
            case_groups[row["case_id"]].add(row["matched_group"])

    # Score every petbert prediction row
    out_rows: list[dict] = []
    uncategorized_cancer_case_ids: set[str] = set()
    for row in pb_rows:
        cid = row["case_id"]
        verdict = score_prediction(
            row["predicted_term"], row["predicted_group"],
            case_terms.get(cid, set()), case_groups.get(cid, set()),
            uncommon_groups,
        )
        # true_negatives are excluded (correct abstentions add no signal)
        if verdict == "true_negative":
            continue
        out_rows.append({
            **row,
            "expected_term": " | ".join(sorted(case_terms.get(cid, set()))),
            "expected_group": " | ".join(sorted(case_groups.get(cid, set()))),
            "verdict": verdict,
        })
        if row["predicted_term"] in NON_CANCER_PRED_TERMS and verdict == "false_negative":
            uncategorized_cancer_case_ids.add(cid)

    # Determine which expected terms are covered by good or slightly_off predictions.
    # A good prediction covers its exact term; a slightly_off prediction covers all
    # expected terms whose group matches the predicted group.
    covered_terms: dict[str, set[str]] = defaultdict(set)
    for row in out_rows:
        cid = row["case_id"]
        if row["verdict"] == "good":
            covered_terms[cid].add(row["predicted_term"])
        elif row["verdict"] == "slightly_off":
            for term in case_terms.get(cid, set()):
                if _covers_group(row["predicted_group"], term_to_group.get(term), uncommon_groups):
                    covered_terms[cid].add(term)

    # One FN row per uncovered expected term. Cases that predicted "Uncategorized"
    # already have a FN prediction row and are excluded to avoid double-counting.
    # Legacy iterated the term set in hash order; sorted here so the table is
    # deterministic (same rows, stable order).
    blank_pred = {column: "" for column in predictions.columns}
    fn_rows = [
        {
            **blank_pred,
            "case_id": cid,
            "expected_term": term,
            "expected_group": term_to_group.get(term, ""),
            "verdict": "false_negative",
        }
        for cid, terms in case_terms.items()
        if cid not in uncategorized_cancer_case_ids
        for term in sorted(terms)
        if term not in covered_terms.get(cid, set())
    ]
    all_rows = sorted(out_rows + fn_rows, key=_case_sort_key)
    columns = list(predictions.columns) + ["expected_term", "expected_group", "verdict"]
    return pd.DataFrame(all_rows, columns=columns)


def share(table: pd.DataFrame, verdicts: Iterable[str], weight_col: str | None = None) -> float:
    """Share of rows whose verdict is in ``verdicts``, optionally weighted.

    The denominator is every row of the verdict table — true negatives are
    already excluded by score(). NaN for an empty table.
    """
    hit = table["verdict"].isin(tuple(verdicts)).to_numpy()
    weights = table[weight_col].to_numpy(dtype=float) if weight_col else None
    if weights is None:
        return float(hit.mean()) if len(hit) else float("nan")
    total = weights.sum()
    return float((weights * hit).sum() / total) if total > 0 else float("nan")


def summarize(table: pd.DataFrame) -> dict:
    """Legacy OVERALL counts and shares, plus G+S.

    total = every verdict-table row (prediction rows + per-term FN rows); true
    negatives are not in the table so never in the denominator. Shares are
    unrounded fractions (legacy wrote round(x * 100, 1)).
    """
    counts = table["verdict"].value_counts()
    summary: dict = {"total": len(table)}
    for verdict in VERDICTS:
        summary[verdict] = int(counts.get(verdict, 0))
        summary[f"{verdict}_share"] = share(table, (verdict,))
    summary["good_plus_slight"] = summary["good"] + summary["slightly_off"]
    summary["good_plus_slight_share"] = share(table, GOOD_PLUS_SLIGHT)
    return summary
