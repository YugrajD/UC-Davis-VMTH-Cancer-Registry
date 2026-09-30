"""Confidence intervals for verdict shares.

- ``wilson``: Wilson score interval for a binomial proportion.
- ``kish_n_eff``: Kish effective sample size of a set of weights.
- ``weighted_proportion``: stratified weighted proportion with a Wilson interval
  on the design-effect (Kish) effective n. Treats rows as independent — for
  per-code metrics, where a case's codes are correlated, use the bootstrap.
- ``bootstrap`` / ``paired_bootstrap``: stratified case-cluster bootstrap. Cases
  (not rows) are resampled with replacement within each stratum, so every row
  of a case moves as a unit. The metric is any function of a verdict table
  (e.g. ``functools.partial(verdicts.share, verdicts=GOOD, weight_col="weight")``).
  Intervals are 95% percentile intervals. A replicate that draws only zero-row
  cases (e.g. true negatives) has an undefined metric (NaN) and is skipped.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pandas as pd

Metric = Callable[[pd.DataFrame], float]


def wilson(successes: float, n: float, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval (low, high). ``successes`` and ``n`` may be fractional
    (effective counts). An empty sample gives the uninformative (0, 1)."""
    if n <= 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def kish_n_eff(weights) -> float:
    """Kish effective sample size: (Σw)² / Σw²."""
    w = np.asarray(weights, dtype=float)
    return float(w.sum() ** 2 / (w * w).sum())


def weighted_proportion(successes, weights, strata=None, z: float = 1.96) -> tuple[float, float, float]:
    """(estimate, low, high) for a weighted proportion over strata.

    Per stratum h: p_h = weighted share, n_eff_h = Kish n_eff of its weights,
    W_h = its share of the total weight. Combined: p = Σ W_h p_h and
    Var = Σ W_h² p_h(1−p_h) / n_eff_h; the Wilson interval is taken at
    n_eff = p(1−p) / Var (the design-effect-adjusted n). When Var is 0 (every
    stratum pure) n_eff falls back to the Kish n_eff of all weights.
    """
    y = np.asarray(successes, dtype=float)
    w = np.asarray(weights, dtype=float)
    strata = np.zeros(len(y)) if strata is None else np.asarray(strata)
    total_weight = w.sum()
    p = 0.0
    var = 0.0
    for stratum in pd.unique(strata):
        in_stratum = strata == stratum
        w_h = w[in_stratum]
        share_h = w_h.sum() / total_weight
        p_h = (w_h * y[in_stratum]).sum() / w_h.sum()
        p += share_h * p_h
        var += share_h ** 2 * p_h * (1 - p_h) / kish_n_eff(w_h)
    n_eff = p * (1 - p) / var if var > 0 else kish_n_eff(w)
    low, high = wilson(p * n_eff, n_eff, z)
    return float(p), low, high


def _row_index(table: pd.DataFrame, case_position: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(rows sorted by case, first sorted row of each case, row count of each case)."""
    positions = table["case_id"].map(case_position)
    if positions.isna().any():
        raise ValueError("verdict table has case_ids missing from case_strata")
    positions = positions.to_numpy(dtype=int)
    order = np.argsort(positions, kind="stable")
    counts = np.bincount(positions, minlength=len(case_position))
    starts = np.cumsum(counts) - counts
    return order, starts, counts


def _rows_of(drawn: np.ndarray, order: np.ndarray, starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Row positions of every drawn case (repeats included), each case's rows contiguous."""
    n_rows = counts[drawn]
    offsets = np.repeat(starts[drawn] - (np.cumsum(n_rows) - n_rows), n_rows)
    return order[offsets + np.arange(n_rows.sum())]


def _replicates(tables: list[pd.DataFrame], metric: Metric, case_strata: pd.Series,
                n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, len(tables)) metric values; every table sees the same resampled cases."""
    case_position = {case_id: i for i, case_id in enumerate(case_strata.index)}
    strata = case_strata.to_numpy()
    stratum_cases = [np.flatnonzero(strata == s) for s in pd.unique(strata)]
    indexes = [_row_index(table, case_position) for table in tables]
    rng = np.random.default_rng(seed)
    out = np.empty((n_boot, len(tables)))
    for b in range(n_boot):
        drawn = np.concatenate([cases[rng.integers(0, len(cases), len(cases))] for cases in stratum_cases])
        for j, (table, index) in enumerate(zip(tables, indexes)):
            out[b, j] = metric(table.iloc[_rows_of(drawn, *index)])
    return out


def bootstrap(table: pd.DataFrame, metric: Metric, case_strata: pd.Series,
              n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """(estimate, low, high): metric on the full table plus a 95% stratified
    case-cluster bootstrap interval.

    case_strata: stratum per case, indexed by case_id. It defines the resampled
    population, so include cases with no rows in ``table`` (e.g. true negatives).
    """
    values = _replicates([table], metric, case_strata, n_boot, seed)[:, 0]
    low, high = np.nanpercentile(values, [2.5, 97.5])
    return metric(table), float(low), float(high)


def paired_bootstrap(challenger: pd.DataFrame, incumbent: pd.DataFrame, metric: Metric,
                     case_strata: pd.Series, n_boot: int = 1000,
                     seed: int = 0) -> tuple[float, float, float]:
    """(difference, low, high) for metric(challenger) − metric(incumbent), both
    scored on the same resampled cases in every replicate."""
    values = _replicates([challenger, incumbent], metric, case_strata, n_boot, seed)
    low, high = np.nanpercentile(values[:, 0] - values[:, 1], [2.5, 97.5])
    return metric(challenger) - metric(incumbent), float(low), float(high)
