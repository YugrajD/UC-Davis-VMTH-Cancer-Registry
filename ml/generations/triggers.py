"""The three retraining triggers (icd-mapping-strategy.md, "The full cycle").

Each returns a ``Trigger``: met / not met plus the numbers behind it. The
thresholds are the strategy's placeholders, to settle on real data.

1. ``new_silver_lineage``: the silver generation a challenger trained on
   differs from the incumbent's (manifest ``parents.silver_id``). Not met when
   either side is unknown.
2. ``random_slice_drop``: on the latest upload period's random slice, the
   incumbent's weighted per-code good share is below its good share on the rest
   of gold-eval, and the two 95% stratified case-cluster bootstrap intervals do
   not overlap (slice high < rest low). The latest period is excluded from the
   reference so the two samples are disjoint: the reference stands for "the
   incumbent's gold-eval accuracy" before this period arrived. Periods compare
   as strings (``YYYY-MM`` sorts correctly).
3. ``gold_train_growth``: gold-train codes now (``manual_audit.gold.gold_train``,
   ``NO_CANCER`` rows excluded) minus the count the incumbent trained on
   (manifest ``parents.gold_train_codes``; 0 when absent, e.g. gen-0) is at
   least ``GOLD_TRAIN_GROWTH``. A count difference, so a re-reviewed case whose
   code set is replaced does not count as new.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

import pandas as pd

from evaluation import intervals, verdicts
from manual_audit import gold

GOLD_TRAIN_GROWTH = 200


@dataclass(frozen=True)
class Trigger:
    name: str
    met: bool
    numbers: dict = field(default_factory=dict)


def new_silver_lineage(incumbent_manifest: dict, silver_id: str | None) -> Trigger:
    incumbent = (incumbent_manifest.get("parents") or {}).get("silver_id")
    met = incumbent is not None and silver_id is not None and silver_id != incumbent
    return Trigger("new_silver_lineage", met, {"incumbent_silver_id": incumbent, "silver_id": silver_id})


def gold_train_growth(incumbent_manifest: dict, split_id: str, gold_csv=None,
                      threshold: int = GOLD_TRAIN_GROWTH) -> Trigger:
    trained_on = int((incumbent_manifest.get("parents") or {}).get("gold_train_codes") or 0)
    rows = gold.gold_train(split_id, gold_csv)
    now = int((rows["code"] != gold.NO_CANCER).sum())
    return Trigger("gold_train_growth", now - trained_on >= threshold,
                   {"trained_on_codes": trained_on, "gold_train_codes": now, "new_codes": now - trained_on,
                    "threshold": threshold})


def random_slice_drop(bronze_table: pd.DataFrame, cases: pd.DataFrame, gold_rows: pd.DataFrame,
                      n_boot: int = 1000, seed: int = 0) -> Trigger:
    """``bronze_table``: the incumbent's gold-eval verdict table with ``weight``
    (``evaluation.gold_eval.score_bronze``); ``cases``: ``gold_eval.case_weights``."""
    slice_rows = gold_rows[gold_rows["origin"] == "random_slice"]
    if slice_rows.empty:
        return Trigger("random_slice_drop", False, {"slice_cases": 0})
    period = slice_rows["upload_period"].max()
    in_slice = cases.index.isin(slice_rows.loc[slice_rows["upload_period"] == period, "case_id"])
    metric = partial(verdicts.share, verdicts=verdicts.GOOD, weight_col="weight")

    def good(subset: pd.DataFrame) -> tuple[float, float, float]:
        rows = bronze_table[bronze_table["case_id"].isin(subset.index)]
        if rows.empty:
            return float("nan"), float("nan"), float("nan")
        return intervals.bootstrap(rows, metric, subset["stratum"], n_boot, seed)

    slice_good, slice_lo, slice_hi = good(cases[in_slice])
    rest_good, rest_lo, rest_hi = good(cases[~in_slice])
    return Trigger("random_slice_drop", bool(slice_hi < rest_lo), {
        "period": period, "slice_cases": int(in_slice.sum()), "reference_cases": int((~in_slice).sum()),
        "slice_good": slice_good, "slice_lo": slice_lo, "slice_hi": slice_hi,
        "reference_good": rest_good, "reference_lo": rest_lo, "reference_hi": rest_hi,
    })
