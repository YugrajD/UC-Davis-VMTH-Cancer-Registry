"""Helpers shared by the per-stage evaluation modules.

These were previously copy-pasted across evaluate_case_presence,
evaluate_groups, and evaluate_label_presence.
"""

from __future__ import annotations

from pathlib import Path


def safe_div(num: float, den: float) -> float:
    return num / den if den > 0 else 0.0


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = safe_div(tp, tp + fp)
    r = safe_div(tp, tp + fn)
    f = safe_div(2 * p * r, p + r)
    return p, r, f


def load_filter_ids(cases_txt: str) -> set[str] | None:
    if not cases_txt:
        return None
    p = Path(cases_txt)
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def load_uncommon_groups(uncommon_groups_path: str) -> frozenset[str]:
    p = Path(uncommon_groups_path)
    if not p.exists():
        return frozenset()
    with open(p, encoding="utf-8") as f:
        return frozenset(line.strip() for line in f if line.strip())
