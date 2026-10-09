"""The combination rule: gold > silver > bronze, one combined prediction per case.

Per icd-mapping-strategy.md ("Coding a case") and ml-rewrite-plan.md's
Artefacts:

- A case with any gold row (any origin) takes gold — ``code_source=manual``,
  ``review_status=confirmed`` (gold is already a specialist's review).
- Else, a case with diagnosis (silver) rows where every row is decisive
  takes silver — ``code_source=diagnosis``, ``review_status=auto_accepted``.
- Else, a case with diagnosis rows where any row is vague gets **no** combined
  code here — it is only in the review queue (``coding.queue``) until gold
  resolves it.
- Else (no diagnosis rows at all), the case takes bronze —
  ``code_source=report``, ``review_status`` = ``queued`` if bronze
  needs review, else ``auto_accepted``. **Bronze never overrides silver**:
  bronze is only ever used for a case with zero diagnosis rows.

A non-cancer result (silver's ``no_signal``/declined-LLM rows, or a bronze
row rejected by the case-presence gate) is written as one ``NO_CANCER`` row —
the same sentinel ``manual_audit.gold.NO_CANCER`` uses, so there is exactly
one sentinel across the codebase.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import config
import io_utils
from coding.rule import case_is_vague
from diagnosis_mapping.silver import load_silver
from generations.manifest import read_manifest
from manual_audit.gold import NO_CANCER, load_gold
from report_mapping.generation import read_thresholds, resolve_generation_dir
from report_mapping.inference.decide import REJECTED_METHOD

COMBINED_PREDICTIONS_COLUMNS = [
    "case_id", "code", "term", "group", "code_source",
    "source_version", "source_confidence", "review_status", "n_codes",
]

# ---------------------------------------------------------------------------
# The bronze review rule (restructure-plan.md decision 7) — the only one: the
# backend's ingest-time gate was retired in 83262de. A bronze-only case is
# queued when ANY of its rows was rejected by the case-presence gate, OR its
# rank-1 row's confidence is below the generation's ``review_cut``
# (``thresholds.json``, fitted at calibration time), OR it has no rank-1 row.
# ---------------------------------------------------------------------------


def load_bronze_predictions(predictions_csv: str | Path) -> pd.DataFrame:
    return io_utils.read_csv(predictions_csv, encoding="utf-8", dtype=str, keep_default_na=False)


def bronze_review_cut(bronze: pd.DataFrame, generation: str) -> tuple[str, float]:
    """(generation_id, review_cut) of ``generation`` ("current", "candidate" or a directory), if ``bronze``
    carries exactly that generation's id; refuses otherwise. A cut fitted for other weights would queue the
    wrong share of cases."""
    directory = resolve_generation_dir(generation)
    generation_id = read_manifest(directory)["generation_id"]
    found = sorted(set(bronze["generation_id"]))
    if found != [generation_id]:
        raise ValueError(f"predictions carry generation_id {found}, but generation {generation!r} is "
                         f"{generation_id!r}; pass the generation that made them")
    return generation_id, float(read_thresholds(directory)["review_cut"])


def bronze_case_is_low_confidence(case_rows: pd.DataFrame, review_cut: float) -> bool:
    """The bronze review rule for one case's prediction rows (``diagnosis_index`` 1, 2, ...): a gate-rejected
    row anywhere, no rank-1 row, or a rank-1 confidence below ``review_cut``."""
    rank1 = case_rows[case_rows["diagnosis_index"] == "1"]
    if rank1.empty or (case_rows["method"] == REJECTED_METHOD).any():
        return True
    confidence = rank1.iloc[0]["confidence"]
    return (float(confidence) if confidence else 0.0) < review_cut


def _dedup_code_rows(rows: list[dict]) -> list[dict]:
    """Distinct (code) rows, first occurrence wins; order preserved."""
    seen: set[str] = set()
    out = []
    for row in rows:
        if row["code"] in seen:
            continue
        seen.add(row["code"])
        out.append(row)
    return out


def _gold_source_version(origin: str, batch_or_export_id: str) -> str:
    """``origin`` (plus ``batch_or_export_id`` when there is one) identifies exactly
    which gold ingest produced a row. ``gold_snapshot_hash`` (a hash of gold-train
    only) can't serve this: an eval_batch/random_slice row is never in gold-train,
    so stamping it with that hash would silently misattribute its provenance."""
    return f"{origin}:{batch_or_export_id}" if batch_or_export_id else origin


def _gold_case_codes(case_id: str, rows: pd.DataFrame) -> list[dict]:
    return [{
        "case_id": case_id, "code": r["code"], "term": r["term"], "group": r["group"],
        "code_source": "manual",
        "source_version": _gold_source_version(r["origin"], r["batch_or_export_id"]),
        "source_confidence": "", "review_status": "confirmed",
    } for r in rows.to_dict("records")]


def _silver_case_codes(case_id: str, rows: pd.DataFrame, silver_id: str) -> list[dict]:
    records = rows.to_dict("records")
    cancer_rows = [
        {"case_id": case_id, "code": r["matched_code"], "term": r["matched_term"], "group": r["matched_group"],
         "code_source": "diagnosis", "source_version": silver_id, "source_confidence": r["decision_stage"],
         "review_status": "auto_accepted"}
        for r in records if r["matched_code"]
    ]
    if cancer_rows:
        return _dedup_code_rows(cancer_rows)
    first = records[0]
    return [{
        "case_id": case_id, "code": NO_CANCER, "term": "", "group": "",
        "code_source": "diagnosis", "source_version": silver_id, "source_confidence": first["decision_stage"],
        "review_status": "auto_accepted",
    }]


def _bronze_case_codes(case_id: str, rows: pd.DataFrame, generation_id: str, review_cut: float) -> list[dict]:
    """Adopted rows for one bronze-only case."""
    # diagnosis_index is read as str (dtype=str throughout); sorting it as a
    # string would put rank 10 before rank 2, so sort by its int value.
    records = rows.sort_values("diagnosis_index", key=lambda s: s.astype(int)).to_dict("records")
    cancer_rows = [
        {"case_id": case_id, "code": r["predicted_code"], "term": r["predicted_term"], "group": r["predicted_group"],
         "code_source": "report", "source_version": generation_id, "source_confidence": r["confidence"],
         "review_status": ""}
        for r in records if r["predicted_code"]
    ]
    if cancer_rows:
        combined = _dedup_code_rows(cancer_rows)
    else:
        combined = [{
            "case_id": case_id, "code": NO_CANCER, "term": "", "group": "",
            "code_source": "report", "source_version": generation_id, "source_confidence": records[0]["confidence"],
            "review_status": "",
        }]
    status = "queued" if bronze_case_is_low_confidence(rows, review_cut) else "auto_accepted"
    for row in combined:
        row["review_status"] = status
    return combined


def combine_predictions(
    silver_id: str,
    split_id: str,
    predictions_csv: str | Path,
    *,
    generation: str = "current",
) -> pd.DataFrame:
    """Build the combined-predictions table for every case in silver ∪ bronze ∪ gold.

    Gold beats silver beats bronze; a vague silver case contributes no row
    here at all (see the module docstring). ``split_id`` is accepted (and
    required) for CLI/API symmetry with ``coding.corrected``/``coding.queue``,
    which both need a split to resolve partitions; ``combine_predictions`` itself
    doesn't currently need one (gold rows are now provenance-stamped from
    their own origin, not a split-scoped gold snapshot — see
    ``_gold_source_version``).
    """
    silver = load_silver(silver_id)
    gold = load_gold()
    bronze = load_bronze_predictions(predictions_csv)

    gold_by_case = {cid: g for cid, g in gold.groupby("case_id")}
    silver_by_case = {cid: g for cid, g in silver.groupby("case_id")}
    bronze_by_case = {cid: g for cid, g in bronze.groupby("case_id")}

    # Resolved lazily (only if a case actually needs bronze coding), so a
    # run with no bronze-only cases never has to make sense of predictions_csv's
    # generation_id column at all.
    bronze_generation: tuple[str, float] | None = None

    all_cases = sorted(set(gold_by_case) | set(silver_by_case) | set(bronze_by_case))
    rows: list[dict] = []
    for case_id in all_cases:
        if case_id in gold_by_case:
            rows.extend(_gold_case_codes(case_id, gold_by_case[case_id]))
        elif case_id in silver_by_case:
            case_rows = silver_by_case[case_id]
            if case_is_vague(case_rows):
                continue  # queued, not coded (coding.queue)
            rows.extend(_silver_case_codes(case_id, case_rows, silver_id))
        elif case_id in bronze_by_case:
            if bronze_generation is None:
                bronze_generation = bronze_review_cut(bronze, generation)
            rows.extend(_bronze_case_codes(case_id, bronze_by_case[case_id], *bronze_generation))

    df = pd.DataFrame(rows, columns=COMBINED_PREDICTIONS_COLUMNS)
    # Rows per case (a NO_CANCER case has 1), on every row, so a multi-code case is visible from any of its rows.
    df["n_codes"] = df.groupby("case_id")["code"].transform("size")
    return df


def write_combined_predictions(
    silver_id: str,
    split_id: str,
    predictions_csv: str | Path,
    *,
    generation: str = "current",
    out_csv: str | Path | None = None,
) -> Path:
    out_path = Path(out_csv) if out_csv is not None else config.COMBINED_PREDICTIONS_CSV
    df = combine_predictions(silver_id, split_id, predictions_csv, generation=generation)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    io_utils.write_csv(df, out_path)
    return out_path
