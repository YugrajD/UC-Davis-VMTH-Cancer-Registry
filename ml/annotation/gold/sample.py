"""Row-level Tier-3 audit sample from test cases.

Draws ~n_rows individual diagnosis rows from the stages where the cascade was
actually working — the LLM tier and the two matchers around it — so every row
the reviewer touches is a row worth touching.

Why rows, not cases:  a per-case sample has to emit every diagnosis row of a
selected case, and cases picked for their one interesting row drag their
`no_signal` siblings along — measured at ~66% padding, i.e. ~5 rows reviewed per
useful row.  Sampling rows directly removes the padding entirely.

The trade this makes:  a row-level sample cannot be scored by `evaluate.py`,
which compares predictions against a case's *whole* annotated term set and would
read the un-sampled rows of a partially-covered case as false positives.  This
sample answers a different question — "when the cascade reached Tier 3, was it
right, and are its declines silent false negatives?" — and is scored per row.

Strata (`_ROW_QUOTAS`) split the LLM tier by outcome, because an accepted answer,
a decline and a hedge are three different questions:
  - the LLM answered            -> is the chosen term right?
  - the LLM declined            -> is there a cancer here it missed?
  - the LLM said "uncertain"    -> is the diagnosis genuinely unclassifiable?
  - candidates never built      -> recall hole upstream of the model
  - fuzzy token-overlap matched -> is the match the same clinical entity?

Each row carries `sample_stratum` / `sample_weight` (N_h/n_h) so per-stratum
rates extrapolate back to the Tier-3 population.

Emits a review CSV (see csv_io) plus instructions and taxonomy sidecars, and a
batch-cases ledger used by `check-split` and to exclude already-audited cases
from later batches.

The reviewer sees the **diagnosis text only** — no report sections.  The cascade
maps a single diagnosis string to a label (`pipeline._run_matching_pass` reads
one column, and the LLM prompt embeds only that string), so a reviewer given the
HIST summary or final comment would judge on evidence the cascade cannot see,
and the resulting labels would measure something the cascade cannot be fixed to
achieve.
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

from ICD_labels.taxonomy import load_labels_taxonomy

from .csv_io import write_instructions, write_review_csv, write_taxonomy_csv

# Blank human-input columns (ICD-O code is derived from group+term at ingest).
_HUMAN_COLS = ["verdict", "confirmed_term", "confirmed_group", "notes"]

# Cascade prediction columns joined from annotation.csv.
_CASCADE_COLS = [
    "cascade_matched_term",
    "cascade_matched_group",
    "cascade_matched_code",
    "cascade_method",
    "decision_stage",
]

# Sampling-provenance columns, needed to weight per-stratum rates back to the pool.
_SAMPLE_COLS = ["sample_stratum", "sample_weight"]

# Share of the row budget per stratum. The two biggest suspected error reservoirs
# — declines and the candidate-build hole — get the largest slices.
_ROW_QUOTAS = {
    "tier3_llm_no_match":  0.25,
    "tier3_no_candidates": 0.25,
    "tier3_llm_answered":  0.25,
    "tier3_llm_uncertain": 0.125,
    "tier2_fuzzy":         0.125,
}

# Presentation order: the questions most likely to change the pipeline come first.
_STRATUM_ORDER = tuple(_ROW_QUOTAS)


def _load_case_ids(txt_path: str) -> list[str]:
    with open(txt_path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _row_stratum(row: dict) -> str | None:
    """Return the audit stratum for an annotation row, or None if not auditable."""
    stage = row.get("decision_stage", "").strip()
    if stage == "tier2_fuzzy":
        return "tier2_fuzzy"
    if stage == "tier3_no_candidates":
        return "tier3_no_candidates"
    if stage != "tier3_llm":
        return None
    method = row.get("method", "").strip()
    if method == "No Match":
        return "tier3_llm_no_match"
    if method == "Uncertain":
        return "tier3_llm_uncertain"
    return "tier3_llm_answered"


def _load_auditable_rows(annotation_csv: str, case_ids: set[str]) -> dict[str, list[dict]]:
    """Return {stratum: [annotation row, ...]} for auditable rows of the given cases."""
    pools: dict[str, list[dict]] = defaultdict(list)
    with open(annotation_csv, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if "decision_stage" not in (reader.fieldnames or []):
            raise SystemExit(
                f"{annotation_csv} has no 'decision_stage' column — the Tier-3 audit "
                "cannot select rows.\nBackfill it first:\n"
                "  ml/.venv/Scripts/python.exe ml/scripts/run_annotation.py --backfill-stage"
            )
        for row in reader:
            if row["case_id"] not in case_ids:
                continue
            stratum = _row_stratum(row)
            if stratum:
                pools[stratum].append(row)
    return pools


def _taxonomy_rows(labels_csv: str) -> list[tuple[str, str, str]]:
    """Return the sorted unique (group, term, code) reference rows."""
    labels = load_labels_taxonomy(labels_csv)
    return sorted({(lbl.group, lbl.term, lbl.code) for lbl in labels})


def sample(
    annotation_csv: str,
    test_cases_txt: str,
    labels_csv: str,
    out_csv: str,
    out_instructions_md: str,
    out_taxonomy_csv: str,
    batch_cases_out: str,
    n_rows: int = 200,
    seed: int = 42,
    exclude_cases: list[str] | None = None,
) -> None:
    """Draw a stratified row-level Tier-3 audit and write the review CSV + sidecars + ledger."""
    rng = random.Random(seed)

    test_ids = set(_load_case_ids(test_cases_txt))

    # Exclusion is per case, not per row: showing a reviewer a second row of a case
    # they already judged invites them to recall the earlier verdict rather than
    # read the new diagnosis.
    excluded: set[str] = set()
    for path in (exclude_cases or []):
        excluded.update(_load_case_ids(path))

    pools = _load_auditable_rows(annotation_csv, test_ids - excluded)

    selected: list[dict] = []
    stratum_weight: dict[str, float] = {}
    for stratum, share in _ROW_QUOTAS.items():
        # sorted() before shuffle: dict/set iteration order varies with
        # PYTHONHASHSEED, which would make the seed fail to pin the sample.
        pool = sorted(pools.get(stratum, []),
                      key=lambda r: (r["case_id"], r.get("diagnosis_number", "")))
        if not pool:
            continue
        rng.shuffle(pool)
        take = pool[:min(round(share * n_rows), len(pool))]
        stratum_weight[stratum] = len(pool) / len(take)
        for row in take:
            selected.append({"_stratum": stratum, **row})

    order = {s: i for i, s in enumerate(_STRATUM_ORDER)}
    selected.sort(key=lambda r: (order[r["_stratum"]], r["case_id"],
                                 r.get("diagnosis_number", "")))

    header = (["case_id", "diagnosis_number", "diagnosis"]
              + _CASCADE_COLS + _SAMPLE_COLS + _HUMAN_COLS)

    review_rows: list[dict] = []
    for ann_row in selected:
        stratum = ann_row["_stratum"]
        row: dict = {
            "case_id": ann_row["case_id"],
            "diagnosis_number": ann_row.get("diagnosis_number", ""),
            "diagnosis": ann_row.get("diagnosis", ""),
            "cascade_matched_term": ann_row.get("matched_term", ""),
            "cascade_matched_group": ann_row.get("matched_group", ""),
            "cascade_matched_code": ann_row.get("matched_code", ""),
            "cascade_method": ann_row.get("method", ""),
            "decision_stage": ann_row.get("decision_stage", ""),
            "sample_stratum": stratum,
            "sample_weight": f"{stratum_weight[stratum]:.2f}",
        }
        for col in _HUMAN_COLS:
            row[col] = ""
        review_rows.append(row)

    write_review_csv(out_csv, header, review_rows)
    write_instructions(out_instructions_md)
    write_taxonomy_csv(out_taxonomy_csv, _taxonomy_rows(labels_csv))

    # Ledger of the cases touched, so later batches can exclude them.
    case_ids = sorted({r["case_id"] for r in review_rows})
    ledger = Path(batch_cases_out)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text("\n".join(case_ids) + "\n", encoding="utf-8")

    print(f"Sampled {len(review_rows)} rows to review, across {len(case_ids)} cases. "
          "Every row is a Tier-2/Tier-3 decision — no padding.")
    print("\nRows per stratum (drawn / population, weight):")
    for stratum in _STRATUM_ORDER:
        drawn = sum(1 for r in review_rows if r["sample_stratum"] == stratum)
        if drawn:
            print(f"  {stratum:<22}{drawn:>4} / {len(pools.get(stratum, [])):<6}"
                  f"weight={stratum_weight[stratum]:>7.1f}")
    if excluded:
        print(f"\nExcluded {len(excluded)} cases from earlier batches.")
    print(f"\nReview CSV: {out_csv}")
    print(f"Instructions: {out_instructions_md}")
    print(f"Taxonomy reference: {out_taxonomy_csv}")
    print(f"Batch cases ledger: {batch_cases_out}")
