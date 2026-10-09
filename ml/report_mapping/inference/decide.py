"""Gate probability + ranked term probabilities + thresholds -> predictions-CSV rows.

A rejected case (gate probability below ``gate_t``) gets one Non-Cancer row, method ``rejected_by_case_presence``
(the string coding/combine.py keys on). Otherwise the rank-1 term, plus the rank-2 term iff its probability is
>= ``tau`` AND its ratio to the rank-1 probability is >= ``rho``. A case with no report text is never scored,
so it gets no row. ``unidentified_cancer`` is never emitted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:  # annotation only: keeps this module free of torch for coding/ and handoff/
    from report_mapping.inference.scoring import RankedScores

COLUMNS = ["case_id", "diagnosis_index", "predicted_term", "predicted_group", "predicted_code",
           "case_presence_prob", "confidence", "group_prob", "method", "generation_id"]
NON_CANCER = "Non-Cancer"
REJECTED_METHOD = "rejected_by_case_presence"
TERM_METHOD = "term_head"
NEVER = 1.01  # a tau above every probability: never emit a second term


@dataclass(frozen=True)
class Thresholds:
    gate_t: float
    tau: float
    rho: float

    def as_dict(self) -> dict:
        return asdict(self)


def wants_second(p1: float, p2: float, th: Thresholds) -> bool:
    return p2 >= th.tau and p1 > 0 and p2 / p1 >= th.rho


def case_rows(case_id: str, gate_prob: float, terms: list[str], probs, th: Thresholds,
              term_info: dict[str, tuple[str, str]], generation_id: str) -> list[dict]:
    """Rows of one case. ``terms``/``probs`` are the ranked candidates, best first (at least two)."""
    base = {"case_id": case_id, "case_presence_prob": f"{gate_prob:.4f}", "group_prob": "", "generation_id": generation_id}
    if gate_prob < th.gate_t:
        return [{**base, "diagnosis_index": 1, "predicted_term": NON_CANCER, "predicted_group": NON_CANCER,
                 "predicted_code": "", "confidence": "0.00", "method": REJECTED_METHOD}]
    ranks = [0, 1] if wants_second(float(probs[0]), float(probs[1]), th) else [0]
    rows = []
    for rank in ranks:
        group, code = term_info[terms[rank]]
        rows.append({**base, "diagnosis_index": rank + 1, "predicted_term": terms[rank], "predicted_group": group,
                     "predicted_code": code, "confidence": f"{float(probs[rank]):.2f}", "method": TERM_METHOD})
    return rows


def decide_all(scores: RankedScores, th: Thresholds, term_info: dict[str, tuple[str, str]],
               generation_id: str) -> pd.DataFrame:
    """Predictions for every scored case, in ``scores`` order."""
    rows = []
    for i, case_id in enumerate(scores.case_ids):
        rows += case_rows(case_id, float(scores.gate_probs[i]), scores.ranked_terms(i, 2), scores.probs[i], th,
                          term_info, generation_id)
    return pd.DataFrame(rows, columns=COLUMNS).astype(str)
