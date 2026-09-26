"""Golden tests: evaluation/verdicts.score() against the legacy ml/evaluation/evaluate.py.

LEGACY COMPARISON — deleted at cutover (WP13), when the old ml/ tree (including
ml/evaluation/) is removed. The legacy scorer exists here only to prove the new
score() produces the same rows and verdicts.

Synthetic expectation/prediction tables only (taxonomy-style terms, no report
or diagnosis text): every verdict, multi-code cases, no-cancer cases (with and
without an expectation row), false positives on no-cancer cases, the uncommon
group slightly_off rule and its FN coverage, Non-Cancer on a cancer case, the
legacy "Uncategorized" term, duplicate predictions and duplicate expectations.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluation.verdicts import score, summarize

from . import fixtures as fx

_ML_DIR = Path(__file__).resolve().parents[2]


def _load_legacy_evaluate():
    """Load ml/evaluation/evaluate.py by file path.

    Appending ml/ to sys.path (as fixtures.py does) is not enough here: the new
    ml/next/evaluation package shadows the old ml/evaluation package, so the
    legacy ``from evaluation.common import ...`` is pointed at the old
    common.py for the duration of the load. The legacy module's own
    ``sys.path.insert(0, ml/)`` is undone afterwards.
    """
    def load(name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    saved_path = list(sys.path)
    common = load("legacy_evaluation_common", _ML_DIR / "evaluation" / "common.py")
    sys.modules["evaluation.common"] = common
    try:
        return load("legacy_evaluation_evaluate", _ML_DIR / "evaluation" / "evaluate.py")
    finally:
        del sys.modules["evaluation.common"]
        sys.path[:] = saved_path


legacy_evaluate = _load_legacy_evaluate().evaluate

PREDICTION_COLUMNS = [
    "case_id", "diagnosis_index", "predicted_term", "predicted_group", "predicted_code",
    "case_presence_prob", "confidence", "group_prob", "method",
]
# evaluation.csv columns: legacy drops method and predicted_code.
LEGACY_COLUMNS = [
    "case_id", "diagnosis_index", "predicted_term", "expected_term", "predicted_group",
    "expected_group", "case_presence_prob", "confidence", "group_prob", "verdict",
]

MCT = "Mast Cell Tumors"
SARC = "Rare Sarcomas"
CARC = "Rare Carcinomas"
NOS = "Neoplasms, NOS"
MCT_MAL, MCT_BEN, MCS = "Mast cell tumor, malignant", "Mast cell tumor, benign", "Mast cell sarcoma"
FIB, SCC, NEO = "Fibrosarcoma, NOS", "Squamous cell carcinoma, NOS", "Neoplasm, benign"
UNCOMMON = frozenset({SARC, CARC, NOS})

# (case_id, matched_term, matched_group)
_EXPECTATIONS = [
    ("CASE-0001", MCT_MAL, MCT),   # good
    ("CASE-0002", MCT_MAL, MCT),   # slightly_off, same group covers the term
    ("CASE-0003", FIB, SARC),      # slightly_off via predicted "Uncommon"
    ("CASE-0004", FIB, SARC),      # slightly_off via another uncommon group
    ("CASE-0005", MCT_MAL, MCT),   # completely_off (uncommon pred, common truth) + FN row
    ("CASE-0006", SCC, CARC),      # completely_off (common pred, uncommon truth) + FN row
    ("CASE-0007", MCT_MAL, MCT),   # multi-code: good + slightly_off, two uncovered FN rows
    ("CASE-0007", FIB, SARC),
    ("CASE-0007", NEO, NOS),
    ("CASE-0008", MCT_BEN, MCT),   # Non-Cancer on a multi-code cancer case: one FN row only
    ("CASE-0008", SCC, CARC),
    ("CASE-0009", MCT_MAL, MCT),   # Non-Cancer beside a real prediction: no per-term FN rows
    ("CASE-0009", FIB, SARC),
    ("CASE-0010", SCC, CARC),      # legacy "Uncategorized" term → FN
    ("CASE-0011", MCT_MAL, MCT),   # no prediction rows at all → one FN row per term
    ("CASE-0011", SCC, CARC),
    ("CASE-0012", "", ""),         # no cancer, Non-Cancer pred → true negative (dropped)
    ("CASE-0013", "", ""),         # no cancer, cancer pred → false positive
    ("CASE-0016", MCT_MAL, MCT),   # duplicate predictions and a duplicate expectation
    ("CASE-0016", MCT_MAL, MCT),
    ("CASE-0017", " ", MCT),       # blank term, group only → still a no-cancer case
    ("CASE-0018", FIB, SARC),      # Uncommon slightly_off covers FIB but not MCT
    ("CASE-0018", MCT_MAL, MCT),
    ("CASE-0019", NEO, NOS),       # slightly_off inside Neoplasms, NOS
    ("CASE-0020", MCS, SARC),      # inconsistent term→group (global last-seen map)
    ("CASE-0021", MCS, MCT),       # ...its FN row takes the last-seen group
]

# (case_id, predicted_term, predicted_group)
_PREDICTIONS = [
    ("CASE-0001", MCT_MAL, MCT),
    ("CASE-0002", MCS, MCT),
    ("CASE-0003", SCC, "Uncommon"),
    ("CASE-0004", SCC, CARC),
    ("CASE-0005", FIB, SARC),
    ("CASE-0006", MCT_MAL, MCT),
    ("CASE-0007", MCT_MAL, MCT),
    ("CASE-0007", MCS, MCT),
    ("CASE-0008", "Non-Cancer", "Non-Cancer"),
    ("CASE-0009", "Non-Cancer", "Non-Cancer"),
    ("CASE-0009", MCT_MAL, MCT),
    ("CASE-0010", "Uncategorized", "Uncategorized"),
    ("CASE-0012", "Non-Cancer", "Non-Cancer"),
    ("CASE-0013", FIB, SARC),
    ("CASE-0014", MCT_MAL, MCT),                      # no expectation rows → FP
    ("CASE-0015", "Uncategorized", "Uncategorized"),  # no expectation rows → TN
    ("CASE-0016", MCT_MAL, MCT),
    ("CASE-0016", MCT_MAL, MCT),
    ("CASE-0017", MCT_MAL, MCT),
    ("CASE-0018", SCC, "Uncommon"),
    ("CASE-0019", "Neoplasm, malignant", NOS),
    ("CASE-0020", SCC, CARC),
    ("CASE-0021", FIB, SARC),
]


def _expectations(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["case_id", "matched_term", "matched_group"])


def _predictions(rows) -> pd.DataFrame:
    full = [
        (cid, str(i + 1), term, group, f"{i:04d}/3", "0.9", "0.8", "0.7", "petbert")
        for i, (cid, term, group) in enumerate(rows)
    ]
    return pd.DataFrame(full, columns=PREDICTION_COLUMNS)


def _run_legacy(tmp_path: Path, expectations: pd.DataFrame, predictions: pd.DataFrame,
                uncommon: frozenset[str]) -> tuple[pd.DataFrame, dict]:
    pred_csv, exp_csv = tmp_path / "pred.csv", tmp_path / "exp.csv"
    predictions.to_csv(pred_csv, index=False)
    expectations.to_csv(exp_csv, index=False)
    uncommon_file = ""
    if uncommon:
        uncommon_file = str(tmp_path / "uncommon_groups.txt")
        Path(uncommon_file).write_text("\n".join(sorted(uncommon)) + "\n", encoding="utf-8")
    out_dir = tmp_path / "legacy_out"
    legacy_evaluate(pred_csv, exp_csv, out_dir, cases_txt="", uncommon_groups_file=uncommon_file)
    table = pd.read_csv(out_dir / "evaluation.csv", dtype=str, keep_default_na=False)
    summary = pd.read_csv(out_dir / "evaluation_summary.csv")
    return table, summary[summary["scope"] == "OVERALL"].iloc[0].to_dict()


def _assert_same_rows(new: pd.DataFrame, old: pd.DataFrame) -> None:
    new = new[LEGACY_COLUMNS]
    assert list(old.columns) == LEGACY_COLUMNS
    # Same case order; within a case legacy emitted FN rows in set-hash order,
    # so compare the rows themselves as a multiset.
    assert list(new["case_id"]) == list(old["case_id"])
    assert sorted(map(tuple, new.values.tolist())) == sorted(map(tuple, old.values.tolist()))


@pytest.mark.parametrize("uncommon", [UNCOMMON, frozenset()], ids=["uncommon", "no-uncommon"])
def test_score_matches_legacy_on_every_edge_case(tmp_path: Path, uncommon: frozenset[str]) -> None:
    expectations, predictions = _expectations(_EXPECTATIONS), _predictions(_PREDICTIONS)
    old, old_overall = _run_legacy(tmp_path, expectations, predictions, uncommon)
    new = score(expectations, predictions, uncommon)
    _assert_same_rows(new, old)

    summary = summarize(new)
    assert summary["total"] == old_overall["total"]
    for verdict in ["good", "slightly_off", "completely_off", "false_positive", "false_negative"]:
        assert summary[verdict] == old_overall[verdict]
        assert round(summary[f"{verdict}_share"] * 100, 1) == old_overall[f"{verdict}_pct"]


def test_edge_case_verdicts_are_the_expected_ones() -> None:
    """Pins the hand-derived verdicts so a legacy-and-new shared mistake still shows."""
    table = score(_expectations(_EXPECTATIONS), _predictions(_PREDICTIONS), UNCOMMON)
    by_case = table.groupby("case_id")["verdict"].apply(sorted).to_dict()
    assert by_case["CASE-0001"] == ["good"]
    assert by_case["CASE-0003"] == ["slightly_off"]
    assert by_case["CASE-0004"] == ["slightly_off"]
    assert by_case["CASE-0005"] == ["completely_off", "false_negative"]
    assert by_case["CASE-0007"] == ["false_negative", "false_negative", "good", "slightly_off"]
    assert by_case["CASE-0008"] == ["false_negative"]
    assert by_case["CASE-0009"] == ["false_negative", "good"]
    assert by_case["CASE-0011"] == ["false_negative", "false_negative"]
    assert by_case["CASE-0013"] == ["false_positive"]
    assert by_case["CASE-0014"] == ["false_positive"]
    assert by_case["CASE-0016"] == ["good", "good"]
    assert by_case["CASE-0017"] == ["false_positive"]
    assert by_case["CASE-0018"] == ["false_negative", "slightly_off"]
    assert "CASE-0012" not in by_case and "CASE-0015" not in by_case  # true negatives dropped
    fn_0021 = table[(table["case_id"] == "CASE-0021") & (table["verdict"] == "false_negative")]
    assert fn_0021["expected_group"].tolist() == [MCT]

    summary = summarize(table)
    assert summary["good_plus_slight_share"] == pytest.approx(
        (summary["good"] + summary["slightly_off"]) / len(table))


def test_score_matches_legacy_on_random_tables(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    labels = [(term, group) for group, entries in fx.TAXONOMY_GROUPS.items() for _, term in entries]
    pred_labels = labels + [("Other term", "Uncommon"), ("Non-Cancer", "Non-Cancer"),
                            ("Uncategorized", "Uncategorized")]
    expectations, predictions = [], []
    for i in range(1, 301):
        cid = f"CASE-{i:04d}"
        for _ in range(rng.integers(0, 4)):  # 0 codes = case absent from expectations
            term, group = labels[rng.integers(len(labels))] if rng.random() < 0.8 else ("", "")
            expectations.append((cid, term, group))
        for _ in range(rng.integers(0, 3)):
            predictions.append((cid, *pred_labels[rng.integers(len(pred_labels))]))
    expectations, predictions = _expectations(expectations), _predictions(predictions)

    old, _ = _run_legacy(tmp_path, expectations, predictions, UNCOMMON)
    new = score(expectations, predictions, UNCOMMON)
    assert len(new) > 300
    assert set(new["verdict"]) == {"good", "slightly_off", "completely_off", "false_positive", "false_negative"}
    _assert_same_rows(new, old)
