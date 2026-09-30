"""Tests for evaluation/intervals.py: Wilson, Kish, weighted proportion, case-cluster bootstrap."""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd
import pytest

from evaluation import intervals as iv
from evaluation.verdicts import GOOD, share

GOOD_SHARE = functools.partial(share, verdicts=GOOD)
WEIGHTED_GOOD_SHARE = functools.partial(share, verdicts=GOOD, weight_col="weight")


def test_wilson_matches_hand_computed_values() -> None:
    # 8/10, z=1.96: centre = (0.8 + 3.8416/20) / 1.38416 = 0.716738,
    # half = 1.96 * sqrt(0.016 + 0.009604) / 1.38416 = 0.226581.
    assert iv.wilson(8, 10) == pytest.approx((0.490157, 0.943319), abs=1e-6)
    # 0/10: centre = half = 0.19208 / 1.38416 = 0.138770 → (0, 0.277540).
    assert iv.wilson(0, 10) == pytest.approx((0.0, 0.277540), abs=1e-6)
    assert iv.wilson(0, 0) == (0.0, 1.0)


def test_kish_n_eff_matches_hand_computed_values() -> None:
    assert iv.kish_n_eff([1, 1, 1, 1]) == pytest.approx(4.0)
    assert iv.kish_n_eff([1, 3]) == pytest.approx(16 / 10)
    assert iv.kish_n_eff([2, 2, 90]) == pytest.approx(94 ** 2 / 8108)


def test_weighted_proportion_single_stratum_equal_weights_is_plain_wilson() -> None:
    y = [1] * 8 + [0] * 2
    p, low, high = iv.weighted_proportion(y, [5.0] * 10)
    assert p == pytest.approx(0.8)
    assert (low, high) == pytest.approx(iv.wilson(8, 10))


def test_weighted_proportion_combines_strata_by_hand() -> None:
    # A: 4 rows weight 1, 3 successes; B: 2 rows weight 3, 1 success.
    y = [1, 1, 1, 0, 1, 0]
    w = [1, 1, 1, 1, 3, 3]
    strata = ["A"] * 4 + ["B"] * 2
    # W_A = 4/10, W_B = 6/10; p = 0.4*0.75 + 0.6*0.5 = 0.6
    # Var = 0.16*0.1875/4 + 0.36*0.25/2 = 0.0075 + 0.045 = 0.0525; n_eff = 0.24/0.0525
    p, low, high = iv.weighted_proportion(y, w, strata)
    assert p == pytest.approx(0.6)
    n_eff = 0.24 / 0.0525
    assert (low, high) == pytest.approx(iv.wilson(0.6 * n_eff, n_eff))


def _oversampled_sample() -> tuple[pd.DataFrame, pd.Series]:
    """Population: A = 1,000 cases at 80% good, B = 9,000 cases at 20% good
    (truth 0.26). Sample 500 of A (400 good) and 100 of B (20 good); weight N_h/n_h."""
    rows = []
    for stratum, n, n_good, weight in [("A", 500, 400, 2.0), ("B", 100, 20, 90.0)]:
        for i in range(n):
            rows.append((f"{stratum}-{i}", "good" if i < n_good else "completely_off", weight, stratum))
    table = pd.DataFrame(rows, columns=["case_id", "verdict", "weight", "stratum"])
    return table, table.set_index("case_id")["stratum"]


def test_oversampling_a_stratum_does_not_bias_the_weighted_estimate() -> None:
    table, case_strata = _oversampled_sample()
    hits = (table["verdict"] == "good").astype(int)

    assert GOOD_SHARE(table) == pytest.approx(0.7)  # unweighted: biased toward A
    p, low, high = iv.weighted_proportion(hits, table["weight"], table["stratum"])
    assert p == pytest.approx(0.26)
    assert low < 0.26 < high
    estimate, low, high = iv.bootstrap(table, WEIGHTED_GOOD_SHARE, case_strata, n_boot=200, seed=0)
    assert estimate == pytest.approx(0.26)
    assert low < 0.26 < high


def test_bootstrap_resamples_whole_cases_within_strata() -> None:
    # One case with 50 codes, 40 single-code cases; the big case's rows must
    # appear in multiples of 50, and stratum sizes must be preserved.
    rows = [("BIG", "good", "S1")] * 50 + [(f"C-{i}", "completely_off", "S1" if i < 10 else "S2") for i in range(40)]
    table = pd.DataFrame(rows, columns=["case_id", "verdict", "stratum"])
    case_strata = table.drop_duplicates("case_id").set_index("case_id")["stratum"]
    seen: list[pd.DataFrame] = []

    def recording_metric(t: pd.DataFrame) -> float:
        seen.append(t)
        return GOOD_SHARE(t)

    iv.bootstrap(table, recording_metric, case_strata, n_boot=100, seed=3)
    replicates = seen[:-1]  # the last call is the full-table point estimate
    assert len(replicates) == 100
    big_draws = [int((t["case_id"] == "BIG").sum()) for t in replicates]
    assert all(n % 50 == 0 for n in big_draws)
    assert len(set(big_draws)) > 1  # the big case really is being resampled
    # Stratum sizes are fixed: every replicate draws 11 S1 cases and 30 S2 cases.
    for t, n_big_rows in zip(replicates, big_draws):
        n_s1_small = int(((t["stratum"] == "S1") & (t["case_id"] != "BIG")).sum())
        assert n_s1_small + n_big_rows // 50 == 11
        assert int((t["stratum"] == "S2").sum()) == 30


def test_bootstrap_rejects_rows_outside_case_strata() -> None:
    table = pd.DataFrame({"case_id": ["A", "B"], "verdict": ["good", "good"]})
    with pytest.raises(ValueError):
        iv.bootstrap(table, GOOD_SHARE, pd.Series({"A": "s"}), n_boot=10)


def test_bootstrap_bounds_stay_finite_when_a_replicate_draws_only_zero_row_cases() -> None:
    # C has no rows (a true negative), so some replicates draw C, C, C → NaN metric.
    incumbent = pd.DataFrame({"case_id": ["A", "B"], "verdict": ["good", "completely_off"],
                              "weight": [1.0, 1.0]})
    challenger = incumbent.assign(verdict=["good", "good"])
    case_strata = pd.Series({"A": "s1", "B": "s1", "C": "s1"})

    estimate, low, high = iv.bootstrap(incumbent, WEIGHTED_GOOD_SHARE, case_strata, n_boot=1000, seed=3)
    assert estimate == 0.5
    assert np.isfinite([low, high]).all() and low <= estimate <= high
    diff, low, high = iv.paired_bootstrap(challenger, incumbent, WEIGHTED_GOOD_SHARE, case_strata,
                                          n_boot=1000, seed=3)
    assert diff == 0.5
    assert np.isfinite([low, high]).all() and low <= diff <= high


def test_paired_bootstrap_on_identical_tables_is_exactly_zero() -> None:
    table, case_strata = _oversampled_sample()
    assert iv.paired_bootstrap(table, table.copy(), WEIGHTED_GOOD_SHARE, case_strata,
                               n_boot=200, seed=1) == (0.0, 0.0, 0.0)


def test_paired_bootstrap_detects_a_clear_improvement() -> None:
    incumbent, case_strata = _oversampled_sample()
    challenger = incumbent.copy()
    challenger.loc[challenger["case_id"].str.startswith("B-"), "verdict"] = "good"
    diff, low, high = iv.paired_bootstrap(challenger, incumbent, WEIGHTED_GOOD_SHARE, case_strata,
                                          n_boot=200, seed=1)
    assert diff == pytest.approx((0.1 * 0.8 + 0.9 * 1.0) - 0.26)
    assert 0 < low <= diff <= high


def _clustered_dataset(rng: np.random.Generator) -> tuple[pd.DataFrame, pd.Series]:
    """Two strata, A oversampled (30% of the population, 60% of the sample).
    Each case draws its own good-rate from Beta(mean p_h) and has 1 + Poisson(1)
    codes, so a case's codes are correlated. Truth = 0.3*0.8 + 0.7*0.4 = 0.52."""
    rows, strata = [], {}
    for stratum, n, p, population_share in [("A", 60, 0.8, 0.3), ("B", 40, 0.4, 0.7)]:
        for i in range(n):
            case_id = f"{stratum}-{i}"
            strata[case_id] = stratum
            q = rng.beta(2 * p, 2 * (1 - p))
            for good in rng.random(1 + rng.poisson(1)) < q:
                rows.append((case_id, "good" if good else "completely_off", population_share / n))
    return pd.DataFrame(rows, columns=["case_id", "verdict", "weight"]), pd.Series(strata)


def test_bootstrap_interval_covers_the_truth_about_95_percent_of_the_time() -> None:
    rng = np.random.default_rng(20260925)
    n_datasets = 200
    covered = 0
    for k in range(n_datasets):
        table, case_strata = _clustered_dataset(rng)
        _, low, high = iv.bootstrap(table, WEIGHTED_GOOD_SHARE, case_strata, n_boot=200, seed=k)
        covered += low <= 0.52 <= high
    # Binomial sd at 200 datasets is ~1.5 pp.
    assert 0.90 <= covered / n_datasets <= 0.99


def test_weighted_proportion_covers_the_truth_about_95_percent_of_the_time() -> None:
    # Independent rows, A oversampled; truth = 0.3*0.8 + 0.7*0.4 = 0.52.
    rng = np.random.default_rng(7)
    n_datasets = 1000
    strata = np.array(["A"] * 60 + ["B"] * 40)
    weights = np.where(strata == "A", 0.3 / 60, 0.7 / 40)
    rates = np.where(strata == "A", 0.8, 0.4)
    covered = 0
    for _ in range(n_datasets):
        _, low, high = iv.weighted_proportion(rng.random(100) < rates, weights, strata)
        covered += low <= 0.52 <= high
    assert 0.93 <= covered / n_datasets <= 0.97
