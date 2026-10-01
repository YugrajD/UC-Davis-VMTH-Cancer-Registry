"""Fit {gate_t, tau, rho} by grid search on a set of calibration cases, with WP14's objective.

Objective (report_mapping/training/calibrate.py): G+S share of the evaluation.verdicts table of the predictions
against the silver labels on the partition; ties go to the higher Good share, then to the first grid point.

Speed: a case's verdicts depend only on which of three row patterns it gets - rejected (one Non-Cancer row),
top-1 only, or top-1 + top-2 - and not on any other case. The three patterns are scored ONCE through
``verdicts.score`` on the rows ``decide`` writes; a threshold point then just picks one pattern per case and sums
per-case verdict counts. The numbers equal ``verdicts.summarize`` of the real predictions (see the tests).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bronze_rr import decide
from bronze_rr.decide import NEVER, Thresholds
from bronze_rr.scoring import RetrieverScores
from evaluation import verdicts

GATE_GRID = tuple(sorted({round(0.05 * k, 2) for k in range(1, 20)} | {0.80}))
TAU_GRID = tuple(round(0.1 * k, 1) for k in range(1, 10)) + (NEVER,)
RHO_GRID = (0.0, 0.25, 0.5, 0.75)
OBJECTIVE = ("G+S share of the evaluation.verdicts table vs the silver labels on the calibration cases; "
             "ties -> higher Good share, then first grid point (gate_t, tau, rho order)")

REJECT, TOP1, TOP2 = 0, 1, 2
_PATTERN_THRESHOLDS = {REJECT: Thresholds(2.0, NEVER, 0.0), TOP1: Thresholds(0.0, NEVER, 0.0), TOP2: Thresholds(0.0, 0.0, 0.0)}


@dataclass(frozen=True)
class PatternCounts:
    gate_probs: np.ndarray   # (N,) float
    p1: np.ndarray           # (N,) rank-1 probability
    p2: np.ndarray           # (N,) rank-2 probability
    counts: np.ndarray       # (3 patterns, N, 5 verdicts) int
    fixed: np.ndarray        # (5,) verdict counts of labelled cases that were not scored (same in every pattern)

    def pattern(self, th: Thresholds) -> np.ndarray:
        passed = self.gate_probs >= th.gate_t
        ratio = self.p2 / np.where(self.p1 > 0, self.p1, 1.0)  # same arithmetic as decide.wants_second
        second = passed & (self.p1 > 0) & (self.p2 >= th.tau) & (ratio >= th.rho)
        return np.where(~passed, REJECT, np.where(second, TOP2, TOP1))

    def totals(self, th: Thresholds) -> np.ndarray:
        chosen = self.counts[self.pattern(th), np.arange(len(self.gate_probs))]
        return chosen.sum(axis=0) + self.fixed

    def objective(self, th: Thresholds) -> tuple[float, float]:
        """(G+S share, Good share) of the verdict rows at ``th``."""
        t = self.totals(th)
        n = int(t.sum())
        return ((int(t[0] + t[1]) / n, int(t[0]) / n) if n else (0.0, 0.0))


def build_pattern_counts(scores: RetrieverScores, gate_probs: np.ndarray, has_text, labels: pd.DataFrame,
                         universe_ids, uncommon_groups, term_info) -> PatternCounts:
    """``labels``: silver table (case_id, matched_term, matched_group). ``universe_ids``: every case the
    objective counts (the partition); labelled ones that were not scored contribute FN rows in every pattern."""
    expectations = labels[labels["case_id"].isin(universe_ids)]
    index = {c: i for i, c in enumerate(scores.case_ids)}
    gate_by_case = dict(zip(scores.case_ids, gate_probs))
    counts = np.zeros((3, len(index), len(verdicts.VERDICTS)), dtype=np.int64)
    fixed = np.zeros(len(verdicts.VERDICTS), dtype=np.int64)
    for pattern, th in _PATTERN_THRESHOLDS.items():
        predictions = decide.decide_all(scores, gate_by_case, has_text, th, term_info, "calibration")
        table = verdicts.score(expectations, predictions, uncommon_groups)
        per_case = pd.crosstab(table["case_id"], table["verdict"]).reindex(columns=list(verdicts.VERDICTS), fill_value=0)
        scored = np.fromiter((c in index for c in per_case.index), dtype=bool, count=len(per_case))
        rows = [index[c] for c in per_case.index[scored]]
        counts[pattern, rows] = per_case.to_numpy()[scored]
        if pattern == REJECT:
            fixed = per_case.to_numpy()[~scored].sum(axis=0).astype(np.int64)
    return PatternCounts(np.asarray(gate_probs, dtype=float), scores.probs[:, 0], scores.probs[:, 1], counts, fixed)


def fit_thresholds(pc: PatternCounts) -> dict:
    """Grid search. Returns {"thresholds": Thresholds, "gs_share", "good_share", "curve"}: ``curve`` has, per
    gate_t, the best (tau, rho) there and its objective ("best_*") and the objective at the chosen (tau, rho)
    ("at_chosen_*"), for the "no cliff along gate_t" check."""
    best, best_key = None, None
    per_gate: dict[float, tuple] = {}
    for gate_t in GATE_GRID:
        for tau in TAU_GRID:
            for rho in RHO_GRID:
                th = Thresholds(gate_t, tau, rho)
                key = pc.objective(th)
                if best_key is None or key > best_key:
                    best, best_key = th, key
                if gate_t not in per_gate or key > per_gate[gate_t][0]:
                    per_gate[gate_t] = (key, th)
    curve = []
    for gate_t in GATE_GRID:
        (gs, good), th = per_gate[gate_t]
        at_gs, at_good = pc.objective(Thresholds(gate_t, best.tau, best.rho))
        curve.append({"gate_t": gate_t, "best_tau": th.tau, "best_rho": th.rho, "best_gs": gs, "best_good": good,
                      "at_chosen_gs": at_gs, "at_chosen_good": at_good})
    return {"thresholds": best, "gs_share": best_key[0], "good_share": best_key[1], "curve": curve}
