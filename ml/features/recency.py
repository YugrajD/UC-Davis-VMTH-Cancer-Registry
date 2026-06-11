"""Compute normalized exponential recency weights keyed on report date (DtOfRq).

Newer reports receive higher weight; the decay rate is controlled by
half_life_years.  Weights are normalized to mean 1.0 so the effective loss
scale and learning rate are unchanged relative to the unweighted baseline.
"""

from __future__ import annotations

import datetime
from typing import Sequence

import numpy as np
import pandas as pd

from features.demographics import _parse_iso


def recency_weights(
    case_ids: Sequence[str],
    half_life_years: float,
    *,
    demographics_csv: str,
    reference_date: datetime.date | None = None,
) -> np.ndarray:
    """Return float32 recency weights aligned to case_ids, normalized to mean 1.0.

    Args:
        case_ids: Ordered sequence of case IDs matching the training rows.
        half_life_years: Exponential half-life. A report this many years before
            reference_date receives weight 0.5.
        demographics_csv: Path to demographics.csv (case_id, DtOfRq, ...).
        reference_date: Date treated as age 0 (newest). Defaults to the max
            parsed DtOfRq across the full demographics table.

    Returns:
        (N,) float32 array, mean == 1.0.
    """
    df = pd.read_csv(demographics_csv, encoding="utf-8-sig", dtype=str).fillna("")
    df["case_id"] = df["case_id"].str.strip()
    df = df.set_index("case_id")

    # Parse all report dates; keep only parseable ones for reference + median.
    all_dates: dict[str, datetime.date] = {}
    for cid, row in df.iterrows():
        dt = _parse_iso(row.get("DtOfRq", ""))
        if dt is not None:
            all_dates[str(cid)] = dt

    known_dates = sorted(all_dates.values())
    if not known_dates:
        # No dates at all — return uniform weights.
        return np.ones(len(case_ids), dtype=np.float32)

    if reference_date is None:
        reference_date = known_dates[-1]  # newest report = age 0

    # Median date for cases missing from demographics or with unparseable dates.
    # Median is neutral recency rather than best or worst, avoiding distortion.
    median_date = known_dates[len(known_dates) // 2]

    ages = np.empty(len(case_ids), dtype=np.float64)
    for i, cid in enumerate(case_ids):
        dt = all_dates.get(str(cid), median_date)
        ages[i] = (reference_date - dt).days / 365.25

    weights = np.power(0.5, ages / half_life_years).astype(np.float32)
    weights /= weights.mean()  # normalize: preserves loss scale / effective LR

    ess = float(weights.sum()) ** 2 / float((weights ** 2).sum())
    print(
        f"[recency] half_life={half_life_years}y  N={len(case_ids)}  "
        f"ESS={ess:.0f} ({100*ess/len(case_ids):.1f}%)  "
        f"w min={weights.min():.3f} mean={weights.mean():.3f} max={weights.max():.3f}"
    )

    return weights
