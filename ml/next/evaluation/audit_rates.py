"""Per-stratum verdict rates for the row-level Tier-3 audit, with Wilson/Kish CIs.

Carries over ``ml/annotation/gold/rates.py``, now reading the audit store
(``config.AUDIT_STORE_CSV``, written by ``manual_audit.tier3_audit.ingest``).

The audit deliberately over-samples the small strata, so a raw rate off the store
is not a Tier-3 population rate. Every row carries ``sample_weight`` (N_h/n_h),
and this module reports both:

  - the **sample** rate  (what the reviewer actually saw)
  - the **weighted** rate (what it implies for the stratum's whole population),
    with a Wilson interval on the Kish effective n (``intervals.weighted_proportion``).

Deliberately *not* wired into the verdict scorer: this is a row-level sample, and
``verdicts.score`` scores per case against a case's whole term set, so the
un-sampled rows of a partially-covered case would read as false positives.

The population reported is the **test-split** Tier-3 population, because that is
what ``sample`` drew from. Extrapolating further, to the whole corpus, is a separate
step and this tool does not do it for you.

``sample_weight`` is fixed at sample time as N_h/n_h for the *whole* batch, so Σweight
only reconstructs the stratum population once that batch is fully reviewed. Run this
on a partial batch — a ``pilot`` slice, or a batch half filled in — and every
"represents" figure is proportionally short. That is why a pilot is a comprehension
check and not a measurement; this tool warns when it sees a partial batch, but cannot
correct for it. (The weighted *rates* are ratios, so they are unaffected.)

It also answers the strategy's two open vagueness questions (roadmap 1.1):
- **declined LLM answers**: the weighted share of ``tier3_llm_no_match`` rows the
  reviewer says carry a real cancer (verdict ``wrong``) — missed cancers;
- **tier2_fuzzy**: the weighted share of ``tier2_fuzzy`` rows judged ``wrong``.
Both use every reviewed row of the stratum as the denominator; the ``uncertain``
share is reported beside each.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import config
import io_utils
from evaluation import intervals

VERDICT_ORDER = ["correct", "wrong", "no_cancer", "uncertain"]

# What each stratum was drawn to answer, and which verdict is the finding.
STRATUM_QUESTION = {
    "tier3_llm_no_match": (
        "the model refused to label it — is there a cancer it missed?", "wrong"),
    "tier3_no_candidates": (
        "no shortlist was built, so the model was never asked — recall hole?", "wrong"),
    "tier3_llm_answered": (
        "the model picked a term — is it right?", "correct"),
    "tier3_llm_uncertain": (
        "the model called it hedged — is it genuinely unclassifiable?", "uncertain"),
    "tier2_fuzzy": (
        "partial-overlap match — same clinical entity?", "correct"),
}

# The two strata that together form the false-negative reservoir: rows the
# pipeline currently drops as non-cancer without anything checking them.
DROPPED_STRATA = ("tier3_llm_no_match", "tier3_no_candidates")

# A full batch draws 25-50 rows per stratum; far fewer means a partial batch.
PARTIAL_BATCH_ROWS = 20


class AuditRatesError(Exception):
    """No audit rows to rate."""


def load_audit_store(audit_store_csv: str | Path | None = None) -> pd.DataFrame:
    path = Path(audit_store_csv) if audit_store_csv is not None else config.AUDIT_STORE_CSV
    rows = (io_utils.read_csv(path, encoding="utf-8", dtype=str, keep_default_na=False)
            if path.is_file() else pd.DataFrame())
    if rows.empty:
        raise AuditRatesError(f"no audit rows in {path}; ingest a Tier-3 audit batch first")
    rows["weight"] = rows["sample_weight"].astype(float)
    return rows


def rate(rows: pd.DataFrame, verdict: str) -> dict:
    """Sample and weighted share of ``verdict`` in ``rows``, with a Kish/Wilson 95% CI (strata = sample_stratum)."""
    hit = (rows["verdict"] == verdict).to_numpy()
    if not len(rows):
        return {"n": 0, "hits": 0, "sample_share": float("nan"), "represents": 0.0,
                "weighted_share": float("nan"), "lo": float("nan"), "hi": float("nan")}
    est, low, high = intervals.weighted_proportion(hit, rows["weight"], rows["sample_stratum"])
    return {"n": len(rows), "hits": int(hit.sum()), "sample_share": float(hit.mean()),
            "represents": float(rows["weight"].sum()), "weighted_share": est, "lo": low, "hi": high}


def stratum_rates(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per (stratum, verdict seen in it), strata ordered by the population they represent."""
    records = []
    order = rows.groupby("sample_stratum")["weight"].sum().sort_values(ascending=False).index
    for stratum in order:
        in_stratum = rows[rows["sample_stratum"] == stratum]
        headline = STRATUM_QUESTION.get(stratum, ("", None))[1]
        seen = VERDICT_ORDER + sorted(set(in_stratum["verdict"]) - set(VERDICT_ORDER))
        for verdict in seen:
            if (in_stratum["verdict"] == verdict).any():
                records.append({"stratum": stratum, "verdict": verdict, "headline": verdict == headline,
                                **rate(in_stratum, verdict)})
    return pd.DataFrame(records)


def audit_rates(audit_store_csv: str | Path | None = None) -> dict:
    """Stratum rates, the false-negative reservoir, the two open questions, and warnings."""
    rows = load_audit_store(audit_store_csv)
    declined = rows[rows["sample_stratum"] == "tier3_llm_no_match"]
    fuzzy = rows[rows["sample_stratum"] == "tier2_fuzzy"]
    counts = rows["sample_stratum"].value_counts()
    thin = {s: int(n) for s, n in counts.items() if n < PARTIAL_BATCH_ROWS}
    warnings = []
    if thin:
        warnings.append("this looks like a PARTIAL batch (" + ", ".join(f"{s} has only {n} row(s)" for s, n in
                        sorted(thin.items())) + "); every 'represents' figure is proportionally short — "
                        "read the sample rates as a comprehension check, not a measurement")
    if rows["batch"].nunique() > 1:
        warnings.append(f"{rows['batch'].nunique()} batches pooled; 'represents' sums their populations")
    return {
        "rows": len(rows),
        "cases": rows["case_id"].nunique(),
        "by_stratum": stratum_rates(rows),
        "reservoir": rate(rows[rows["sample_stratum"].isin(DROPPED_STRATA)], "wrong"),
        "declined_missed_cancer": rate(declined, "wrong"),
        "declined_uncertain": rate(declined, "uncertain"),
        "tier2_fuzzy_error": rate(fuzzy, "wrong"),
        "tier2_fuzzy_uncertain": rate(fuzzy, "uncertain"),
        "warnings": warnings,
    }
