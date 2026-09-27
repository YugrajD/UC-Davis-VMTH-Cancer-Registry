"""Diagnosis-Mapping audit (formerly the Tier-3 audit): which cases to review, as a list of case IDs.

It answers a different question than the case-level gold-eval batch in
``eval_batch.py``: when the diagnosis cascade reached Tier 2/3, was it right,
and are its declines silent false negatives? Rows are drawn per cascade
decision stage (``_ROW_QUOTAS``), from the eval side (calibration ∪ test) of a
split, excluding cases already in an earlier batch.

``sample`` writes two files per batch under ``config.DIAGNOSIS_MAPPING_AUDIT_DIR``:

- ``diagnosis_mapping_audit_batch<N>_key.csv`` — the sampled rows with the
  cascade's answer and sampling weight. Stays local; it holds diagnosis text.
- ``diagnosis_mapping_audit_batch<N>.txt`` — the batch's case IDs, one per line.

The specialist reviews each whole case on the dashboard (``manual_audit.audit_list``
puts ``pending_case_ids`` on the universal audit list), and the review comes
back as case-level gold with origin ``diagnosis_mapping_audit`` — neither
gold-eval nor gold-train, since the sample is skewed toward the cascade's
hardest rows (see ``generations.guards``).

Batch 1 was issued before this redesign as row-level review sheets; its pilot
slice was ingested into the row-level audit store (``config.AUDIT_STORE_CSV``,
read by ``evaluation.audit_rates``), and its remaining rows go out as case IDs
like any later batch.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

import config
import io_utils
from diagnosis_mapping.silver import load_silver
from generations.splits import load_split
from manual_audit import sheets
from manual_audit.gold import load_gold

# Everything the audit needs: case identity, the cascade's full answer, and the sampling bookkeeping.
KEY_COLS = [
    "row_id", "case_id", "diagnosis_number", "diagnosis",
    "cascade_matched_term", "cascade_matched_group", "cascade_matched_code",
    "cascade_method", "decision_stage", "sample_stratum", "sample_weight",
]

# Share of the row budget per stratum. The two biggest suspected error
# reservoirs — declines and the candidate-build hole — get the largest slices.
_ROW_QUOTAS = {
    "tier3_llm_no_match":  0.25,
    "tier3_no_candidates": 0.25,
    "tier3_llm_answered":  0.25,
    "tier3_llm_uncertain": 0.125,
    "tier2_fuzzy":         0.125,
}
_STRATUM_ORDER = tuple(_ROW_QUOTAS)

# The row-level audit store's schema; batch 1's pilot rows live there (evaluation.audit_rates reads it).
AUDIT_STORE_FIELDS = [
    "case_id", "diagnosis_number", "decision_stage", "sample_stratum", "sample_weight",
    "batch", "cascade_code", "cascade_term", "cascade_group", "cascade_method",
    "match_strength", "verdict", "corrected_code", "reviewer", "reviewed_at", "notes",
]

_KEY_NAME = re.compile(r"^diagnosis_mapping_audit_batch(\d+)_key\.csv$")


class DiagnosisMappingAuditError(Exception):
    """A Diagnosis-Mapping audit draw was refused."""


def key_path(batch: int, out_dir: str | Path | None = None) -> Path:
    out_dir = Path(out_dir) if out_dir is not None else config.DIAGNOSIS_MAPPING_AUDIT_DIR
    return out_dir / f"diagnosis_mapping_audit_batch{batch}_key.csv"


def case_list_path(batch: int, out_dir: str | Path | None = None) -> Path:
    out_dir = Path(out_dir) if out_dir is not None else config.DIAGNOSIS_MAPPING_AUDIT_DIR
    return out_dir / f"diagnosis_mapping_audit_batch{batch}.txt"


def _batch_keys(out_dir: str | Path | None) -> list[Path]:
    """Every batch's key CSV, in batch order."""
    out_dir = Path(out_dir) if out_dir is not None else config.DIAGNOSIS_MAPPING_AUDIT_DIR
    found = [(int(m.group(1)), p) for p in out_dir.glob("*_key.csv") if (m := _KEY_NAME.match(p.name))]
    return [p for _, p in sorted(found)]


def _row_stratum(row: dict) -> str | None:
    """Return the audit stratum for a silver annotation row, or None if not auditable."""
    stage = (row.get("decision_stage") or "").strip()
    if stage == "tier2_fuzzy":
        return "tier2_fuzzy"
    if stage == "tier3_no_candidates":
        return "tier3_no_candidates"
    if stage != "tier3_llm":
        return None
    method = (row.get("method") or "").strip()
    if method == "No Match":
        return "tier3_llm_no_match"
    if method == "Uncertain":
        return "tier3_llm_uncertain"
    return "tier3_llm_answered"


def _load_auditable_rows(silver_df: pd.DataFrame, case_ids: set[str]) -> dict[str, list[dict]]:
    if "decision_stage" not in silver_df.columns:
        raise DiagnosisMappingAuditError("silver generation has no 'decision_stage' column — cannot select rows")
    pools: dict[str, list[dict]] = defaultdict(list)
    for row in silver_df.to_dict("records"):
        if row["case_id"] not in case_ids:
            continue
        stratum = _row_stratum(row)
        if stratum:
            pools[stratum].append(row)
    return pools


def sample(
    silver_id: str,
    split_id: str,
    batch: int,
    n_rows: int = 200,
    seed: int = 42,
    out_dir: str | Path | None = None,
) -> dict:
    """Draw a stratified row sample from silver generation ``silver_id`` on the eval side of ``split_id``,
    skipping cases in earlier batches, and write the batch's key CSV and case-ID list."""
    key_csv, case_list = key_path(batch, out_dir), case_list_path(batch, out_dir)
    existing = [p for p in (key_csv, case_list) if p.is_file()]
    if existing:
        raise DiagnosisMappingAuditError(
            f"batch {batch} already exists ({', '.join(str(p) for p in existing)}); pick a new batch number")
    earlier = {row["case_id"] for path in _batch_keys(out_dir) for row in sheets.read_csv(path)}
    split = load_split(split_id)
    rng = random.Random(seed)
    pools = _load_auditable_rows(load_silver(silver_id), (split.calibration | split.test) - earlier)

    selected: list[dict] = []
    stratum_weight: dict[str, float] = {}
    for stratum, share in _ROW_QUOTAS.items():
        # sorted() before shuffle: dict/set iteration order varies with
        # PYTHONHASHSEED, which would make the seed fail to pin the sample.
        pool = sorted(pools.get(stratum, []), key=lambda r: (r["case_id"], r.get("diagnosis_number", "")))
        if not pool:
            continue
        rng.shuffle(pool)
        take = pool[:min(round(share * n_rows), len(pool))]
        stratum_weight[stratum] = len(pool) / len(take)
        for row in take:
            selected.append({"_stratum": stratum, **row})

    order = {s: i for i, s in enumerate(_STRATUM_ORDER)}
    selected.sort(key=lambda r: (order[r["_stratum"]], r["case_id"], r.get("diagnosis_number", "")))
    key_rows = [{
        "row_id": str(idx),
        "case_id": row["case_id"],
        "diagnosis_number": row.get("diagnosis_number", ""),
        "diagnosis": row.get("diagnosis", ""),
        "cascade_matched_term": row.get("matched_term", ""),
        "cascade_matched_group": row.get("matched_group", ""),
        "cascade_matched_code": row.get("matched_code", ""),
        "cascade_method": row.get("method", ""),
        "decision_stage": row.get("decision_stage", ""),
        "sample_stratum": row["_stratum"],
        "sample_weight": f"{stratum_weight[row['_stratum']]:.2f}",
    } for idx, row in enumerate(selected, start=1)]

    sheets.write_csv(key_csv, KEY_COLS, key_rows)
    case_ids = sorted({r["case_id"] for r in key_rows})
    case_list.write_text("".join(f"{c}\n" for c in case_ids), encoding="utf-8", newline="\n")
    return {
        "key_csv": key_csv, "case_list": case_list, "n_rows": len(key_rows), "n_cases": len(case_ids),
        "stratum_counts": {s: sum(1 for r in key_rows if r["sample_stratum"] == s) for s in _STRATUM_ORDER},
        "stratum_populations": {s: len(pools.get(s, [])) for s in _STRATUM_ORDER},
    }


def pending_case_ids(
    out_dir: str | Path | None = None,
    audit_store_csv: str | Path | None = None,
    gold_csv: str | Path | None = None,
) -> list[str]:
    """Every sampled case still awaiting review, in batch then key order: a case with a key row not yet in
    the row-level audit store, and no gold yet."""
    store_path = Path(audit_store_csv) if audit_store_csv is not None else config.AUDIT_STORE_CSV
    store = (io_utils.read_csv(store_path, encoding="utf-8", dtype=str, keep_default_na=False)
             if store_path.is_file() else pd.DataFrame(columns=AUDIT_STORE_FIELDS))
    reviewed_rows = set(zip(store["case_id"], store["diagnosis_number"]))
    gold_cases = set(load_gold(gold_csv)["case_id"])
    pending: dict[str, None] = {}
    for path in _batch_keys(out_dir):
        for row in sheets.read_csv(path):
            if (row["case_id"], row["diagnosis_number"]) not in reviewed_rows and row["case_id"] not in gold_cases:
                pending[row["case_id"]] = None
    return list(pending)
