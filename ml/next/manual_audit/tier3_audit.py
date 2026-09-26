"""Row-level Tier-3 audit: sample, pilot, ingest.

Carries over ``ml/annotation/gold/sample.py``, ``pilot.py`` and the audit half
of ``ingest.py`` (case-level gold ingest is new, in ``manual_audit/gold.py``).

This audit answers a different question than the case-level gold-eval batch in
``eval_batch.py``: when the diagnosis cascade reached Tier 2/3, was it right,
and are its declines silent false negatives? It is scored per row, not per
case, so it writes to ``config.AUDIT_STORE_CSV`` and **never** to the gold
store — a row-level judgement about one diagnosis line is not a case's
complete code set, so it can never stand in for gold (see
icd-mapping-strategy.md, "Leakage guards").

Schema rename: the legacy gold-annotation schema's ``tier`` column (always
written as the literal ``"gold"`` — every ingested audit row is a completed
human review) is renamed ``match_strength`` here. "Tier" already means the
1/2/3 LLM-cascade stage everywhere else in this codebase, and the schema
originally documented ``tier`` as an unrelated gold/silver/bronze *match
quality* enum, so keeping the name would collide with two other meanings.

Row selection, strata, quotas and the reviewer/key-CSV split are unchanged
from the legacy tool (see ``_ROW_QUOTAS`` and ``_row_stratum`` below) so the
already-issued ``tier3_audit_*`` sheets — drawn, pilot-split and partly
reviewed under the old code — ingest here byte-for-byte the same way.
"""

from __future__ import annotations

import random
from collections import defaultdict
from datetime import date
from pathlib import Path

import pandas as pd

import config
import io_utils
from diagnosis_mapping.silver import load_silver
from manual_audit import sheets
from manual_audit.terms import TermResolutionError, build_term_index, resolve_term
from taxonomy.taxonomy import load_labels_taxonomy

# ---------------------------------------------------------------------------
# Reviewer / key sheet shape (unchanged from the legacy tool)
# ---------------------------------------------------------------------------

# What the reviewer sees, and nothing else. `row_id` joins back to the key CSV;
# `Actual Diagnosis` is the only column they fill in. This is the one sheet in
# manual_audit/ that shows a prediction — see the module docstring.
REVIEW_COLS = ["row_id", "Clinical Diagnosis", "Predicted Match", "Actual Diagnosis"]

# Shown in `Predicted Match` when the cascade concluded there was no cancer. An
# empty cell there would read as missing data rather than a definite negative.
NO_PREDICTION = "(none)"

# Everything the audit needs and the reviewer does not: case identity, the
# cascade's full answer, and the sampling bookkeeping. Stays with us; joined on
# `row_id`.
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

_INSTRUCTIONS = """\
# Diagnosis review — instructions

__PREAMBLE__Open `__REVIEW_CSV__` in Excel, LibreOffice, or any spreadsheet tool.

Each row is one clinical diagnosis line that our pipeline found hard to classify.
There is no filler here — every row is one it struggled with.

| column | what it is |
| --- | --- |
| `row_id` | our reference. Please ignore it, and don't sort or delete rows. |
| `Clinical Diagnosis` | the diagnosis text — this is **all** the pipeline was given |
| `Predicted Match` | what the pipeline concluded. `(none)` = it found no cancer |
| `Actual Diagnosis` | **the only column you fill in** |

## What to do

Read the `Clinical Diagnosis`, then look at the `Predicted Match`.

- **Prediction is right** — including when it says `(none)` and you agree there is no
  reportable cancer — **leave `Actual Diagnosis` empty.**
- **Prediction is wrong** — put the correct diagnosis in `Actual Diagnosis`. This
  includes a `(none)` row that you think *does* describe a cancer; those rows are the
  whole point of this batch.
- **Can't tell from this line** — write `unclear`. That is a real finding about the
  pipeline's input, not a failure on your part.

## Filling in `Actual Diagnosis`

Copy the term **exactly** from the `Term` column of `tier3_audit_taxonomy.csv`.
Spelling matters — there are no dropdowns to catch a typo, so anything we can't find
in the list stops the whole import with a report of which rows to fix. You don't need
the group or the code: both are looked up automatically from the term.

## The one rule that matters

**Judge each row on the `Clinical Diagnosis` text alone.** That single line is all the
pipeline is given, so the question is always "is this the right label *for this
wording*" — not "is this the right label for the patient". Please don't consult the
wider report or the case history, even if you have them to hand. Negation ("no
evidence of neoplasia") and hedging ("suspected", "consistent with") count only when
they appear in that line.

Save as CSV (not .xlsx) when you are done.
"""

VALID_VERDICTS = frozenset({"correct", "wrong", "no_cancer", "uncertain"})
UNCLEAR_MARKERS = frozenset({"unclear", "uncertain", "unsure", "?"})

AUDIT_STORE_FIELDS = [
    "case_id", "diagnosis_number", "decision_stage", "sample_stratum", "sample_weight",
    "batch", "cascade_code", "cascade_term", "cascade_group", "cascade_method",
    "match_strength", "verdict", "corrected_code", "reviewer", "reviewed_at", "notes",
]


class Tier3AuditError(Exception):
    """A Tier-3 audit sample/pilot/ingest call was refused."""


# ---------------------------------------------------------------------------
# sample
# ---------------------------------------------------------------------------


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
        raise Tier3AuditError("silver generation has no 'decision_stage' column — cannot select rows")
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
    test_case_ids: set[str],
    labels_csv: str | Path,
    batch: int,
    n_rows: int = 200,
    seed: int = 42,
    exclude_case_ids: set[str] | None = None,
    out_dir: str | Path | None = None,
) -> dict:
    """Draw a stratified row-level Tier-3 audit from silver generation ``silver_id``.

    Writes the reviewer CSV, the internal key CSV, instructions and taxonomy
    sidecars, and the batch ledger under ``out_dir``. Only the reviewer CSV,
    the instructions and the taxonomy sidecar are ever sent out.
    """
    # config.* is read here, not bound as a default value, so a config path
    # swapped at runtime (as tests do) is always honoured.
    out_dir = Path(out_dir) if out_dir is not None else config.TIER3_AUDIT_DIR
    rng = random.Random(seed)
    excluded = set(exclude_case_ids or ())
    silver_df = load_silver(silver_id)
    pools = _load_auditable_rows(silver_df, test_case_ids - excluded)

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

    review_rows: list[dict] = []
    key_rows: list[dict] = []
    for idx, ann_row in enumerate(selected, start=1):
        stratum = ann_row["_stratum"]
        row_id = str(idx)
        predicted = ann_row.get("matched_term", "") or NO_PREDICTION
        review_rows.append({
            "row_id": row_id,
            "Clinical Diagnosis": ann_row.get("diagnosis", ""),
            "Predicted Match": predicted,
            "Actual Diagnosis": "",
        })
        key_rows.append({
            "row_id": row_id,
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
        })

    # Batch 1 keeps the legacy bare filenames (no "_batchN_" infix) so it lands
    # at the same names as the already-issued sheets when out_dir is the
    # default config.TIER3_AUDIT_DIR; later batches get a batch-numbered name
    # alongside it.
    infix = "" if batch == 1 else f"_batch{batch}"
    review_csv = out_dir / f"tier3_audit{infix}_review.csv"
    key_csv = out_dir / f"tier3_audit{infix}_key.csv"
    ledger = batch_ledger_path(batch, out_dir)
    if ledger.is_file():
        raise Tier3AuditError(
            f"a case ledger already exists at {ledger} for batch {batch}; sample() refuses to "
            f"overwrite it — pick a new batch number, or move the existing ledger aside first"
        )
    # WP7 fix 9: the ledger isn't the only write-once artifact here — a
    # pre-existing review or key sheet at these paths would otherwise be
    # silently overwritten (e.g. a reviewer's in-progress "Actual Diagnosis"
    # edits on review_csv), even though the ledger check above passed.
    existing_sheets = [p for p in (review_csv, key_csv) if p.is_file()]
    if existing_sheets:
        names = ", ".join(str(p) for p in existing_sheets)
        raise Tier3AuditError(
            f"sheet(s) already exist for batch {batch}: {names}; sample() refuses to overwrite "
            f"them — pick a new batch number, or move the existing sheet(s) aside first"
        )

    sheets.write_csv(review_csv, REVIEW_COLS, review_rows)
    sheets.write_csv(key_csv, KEY_COLS, key_rows)
    sheets.write_instructions(
        out_dir / f"tier3_audit{infix}_instructions.md",
        _INSTRUCTIONS.replace("__PREAMBLE__", "").replace("__REVIEW_CSV__", Path(review_csv).name),
    )
    taxonomy_rows = sorted({(lbl.group, lbl.term, lbl.code) for lbl in load_labels_taxonomy(str(labels_csv))})
    sheets.write_csv(
        out_dir / f"tier3_audit{infix}_taxonomy.csv",
        ["Group", "Term", "Code"],
        [{"Group": g, "Term": t, "Code": c} for g, t, c in taxonomy_rows],
    )
    case_ids = sorted({r["case_id"] for r in key_rows})
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text("\n".join(case_ids) + "\n", encoding="utf-8")

    return {
        "review_csv": review_csv, "key_csv": key_csv, "ledger": ledger,
        "n_rows": len(review_rows), "n_cases": len(case_ids),
        "stratum_counts": {s: sum(1 for r in key_rows if r["sample_stratum"] == s) for s in _STRATUM_ORDER},
        "stratum_populations": {s: len(pools.get(s, [])) for s in _STRATUM_ORDER},
    }


def batch_ledger_path(batch: int, out_dir: str | Path | None = None) -> Path:
    """Where the case-id ledger for ``batch`` lives, e.g. ``tier3_audit_batch1_cases.txt``."""
    out_dir = Path(out_dir) if out_dir is not None else config.TIER3_AUDIT_DIR
    return out_dir / f"tier3_audit_batch{batch}_cases.txt"


# ---------------------------------------------------------------------------
# pilot
# ---------------------------------------------------------------------------


def _allocate(counts: dict[str, int], n_rows: int, min_per_stratum: int) -> dict[str, int]:
    """Proportional allocation with largest-remainder rounding and a per-stratum floor."""
    total = sum(counts.values())
    if n_rows > total:
        raise Tier3AuditError(f"asked for {n_rows} pilot rows but the review CSV only has {total}")
    strata = sorted(counts)
    floor_needed = sum(min(min_per_stratum, counts[s]) for s in strata)
    if floor_needed > n_rows:
        raise Tier3AuditError(
            f"{n_rows} pilot rows cannot cover {len(strata)} strata at "
            f"min_per_stratum={min_per_stratum} (needs {floor_needed})"
        )

    exact = {s: n_rows * counts[s] / total for s in strata}
    alloc = {s: min(counts[s], max(int(exact[s]), min(min_per_stratum, counts[s]))) for s in strata}

    while sum(alloc.values()) < n_rows:
        cand = [s for s in strata if alloc[s] < counts[s]]
        s = max(cand, key=lambda s: (exact[s] - alloc[s], counts[s]))
        alloc[s] += 1
    while sum(alloc.values()) > n_rows:
        cand = [s for s in strata if alloc[s] > min(min_per_stratum, counts[s])]
        s = min(cand, key=lambda s: (exact[s] - alloc[s], counts[s]))
        alloc[s] -= 1
    return alloc


def pilot(
    review_csv: str | Path,
    key_csv: str | Path,
    out_pilot_csv: str | Path,
    out_remainder_csv: str | Path,
    n_rows: int = 30,
    min_per_stratum: int = 3,
    out_instructions_md: str | Path | None = None,
) -> dict:
    """Split ``review_csv`` into a stratified pilot slice plus everything left over."""
    rows = sheets.read_csv(review_csv)
    if not rows:
        raise Tier3AuditError(f"review CSV {review_csv} has no rows")
    if "row_id" not in rows[0]:
        raise Tier3AuditError(f"review CSV {review_csv} has no 'row_id' column")

    key = sheets.read_key_rows(key_csv)
    missing = [r["row_id"] for r in rows if r["row_id"] not in key]
    if missing:
        raise Tier3AuditError(f"{len(missing)} row_id(s) in {review_csv} are absent from {key_csv}")

    filled = [r for r in rows if (r.get("Actual Diagnosis") or "").strip()]
    if filled:
        raise Tier3AuditError(f"{len(filled)} row(s) in {review_csv} are already filled in")

    stratum_of = {r["row_id"]: key[r["row_id"]].get("sample_stratum", "") for r in rows}
    counts: dict[str, int] = defaultdict(int)
    for s in stratum_of.values():
        counts[s] += 1
    alloc = _allocate(dict(counts), n_rows, min_per_stratum)

    taken: dict[str, int] = defaultdict(int)
    pilot_rows, remainder_rows = [], []
    for row in rows:  # preserves within-stratum order
        s = stratum_of[row["row_id"]]
        if taken[s] < alloc.get(s, 0):
            taken[s] += 1
            pilot_rows.append(row)
        else:
            remainder_rows.append(row)

    assert len(pilot_rows) + len(remainder_rows) == len(rows)
    assert len(pilot_rows) == n_rows

    header = list(rows[0].keys())
    sheets.write_csv(out_pilot_csv, header, pilot_rows)
    sheets.write_csv(out_remainder_csv, header, remainder_rows)
    if out_instructions_md:
        preamble = (
            f"> **This is a short first pass — {n_rows} rows, not the full batch.**\n"
            ">\n"
            "> The point is to check that these instructions are clear before you spend a full\n"
            "> sitting on the rest. Fill these in, send them back, and we will go through them\n"
            "> together; the remaining rows follow once we agree the questions are landing the\n"
            "> way they read. If anything below is unclear or awkward, please just tell us when\n"
            "> you send it back — that is a useful result, not a complaint."
        )
        sheets.write_instructions(
            out_instructions_md,
            _INSTRUCTIONS.replace("__PREAMBLE__", f"{preamble}\n\n").replace(
                "__REVIEW_CSV__", Path(out_pilot_csv).name
            ),
        )
    return {"pilot_rows": len(pilot_rows), "remainder_rows": len(remainder_rows), "allocation": alloc}


# ---------------------------------------------------------------------------
# ingest (row-level audit only — never the gold store)
# ---------------------------------------------------------------------------


def _read_existing_store(path: Path) -> dict[tuple[str, str, int], dict]:
    try:
        df = sheets.read_existing_store(path, AUDIT_STORE_FIELDS)
    except sheets.StoreSchemaError as error:
        raise Tier3AuditError(str(error)) from error
    return {(r["case_id"], r["diagnosis_number"], int(r["batch"])): r for r in df.to_dict("records")}


def ingest(
    review_csv: str | Path,
    key_csv: str | Path,
    batch: int,
    reviewer: str,
    labels_csv: str | Path | None = None,
    out_csv: str | Path | None = None,
) -> dict:
    """Merge a filled Tier-3 review CSV into the row-level audit store.

    Verdict is derived (see the module docstring's table): blank -> correct or
    no_cancer depending on whether the cascade predicted anything; an "unclear"
    marker -> uncertain; any other text -> wrong, resolved to a taxonomy code.
    Any row that resolves to neither aborts the whole ingest before anything is
    written — a CSV has no dropdowns, so this is the only typo guard.

    Cumulative and keyed on (case_id, diagnosis_number, batch): re-running
    ingest for the same batch replaces its rows rather than duplicating them.
    """
    labels_csv = labels_csv if labels_csv is not None else config.LABELS_CSV
    out_csv = out_csv if out_csv is not None else config.AUDIT_STORE_CSV
    term_index = build_term_index(load_labels_taxonomy(str(labels_csv)))
    key = sheets.read_key_rows(key_csv)
    rows = sheets.read_csv(review_csv)

    errors: list[str] = []
    out_rows: list[dict] = []
    seen_row_ids: set[str] = set()
    today = date.today().isoformat()

    for idx, row in enumerate(rows, start=2):  # +1 header
        row_id = (row.get("row_id") or "").strip()
        if not row_id:
            errors.append(f"Row {idx}: row_id is blank")
            continue
        if row_id in seen_row_ids:
            errors.append(f"Row {idx}: row_id {row_id!r} appears more than once")
            continue
        seen_row_ids.add(row_id)
        if row_id not in key:
            errors.append(f"Row {idx}: row_id {row_id!r} is not in {key_csv}")
            continue

        k = key[row_id]
        actual = (row.get("Actual Diagnosis") or "").strip()
        has_prediction = bool((k.get("cascade_matched_term") or "").strip())
        corrected_code = ""

        if not actual:
            verdict = "correct" if has_prediction else "no_cancer"
        elif actual.lower().rstrip(".") in UNCLEAR_MARKERS:
            verdict = "uncertain"
        else:
            verdict = "wrong"
            try:
                corrected_code, _term, _group = resolve_term(actual, term_index)
            except TermResolutionError as error:
                errors.append(f"Row {idx}: {error} (row_id={row_id}, case_id={k.get('case_id')})")
                continue

        out_rows.append({
            "case_id": k.get("case_id", ""),
            "diagnosis_number": k.get("diagnosis_number", ""),
            "decision_stage": k.get("decision_stage", ""),
            "sample_stratum": k.get("sample_stratum", ""),
            "sample_weight": k.get("sample_weight", ""),
            "batch": batch,
            "cascade_code": k.get("cascade_matched_code", ""),
            "cascade_term": k.get("cascade_matched_term", ""),
            "cascade_group": k.get("cascade_matched_group", ""),
            "cascade_method": k.get("cascade_method", ""),
            # Every ingested row is a completed human review, so this is
            # currently always "gold" — carried over from the legacy `tier`
            # field verbatim, just renamed (see the module docstring).
            "match_strength": "gold",
            "verdict": verdict,
            "corrected_code": corrected_code,
            "reviewer": reviewer,
            "reviewed_at": today,
            "notes": actual,
        })

    if errors:
        raise Tier3AuditError("ingestion aborted due to validation failures:\n" + "\n".join(errors))

    out_path = Path(out_csv)
    store = _read_existing_store(out_path)
    added = replaced = 0
    for row in out_rows:
        rowkey = (row["case_id"], row["diagnosis_number"], row["batch"])
        if rowkey in store:
            replaced += 1
        else:
            added += 1
        store[rowkey] = row

    out_path.parent.mkdir(parents=True, exist_ok=True)
    io_utils.write_csv(pd.DataFrame(store.values(), columns=AUDIT_STORE_FIELDS), out_path)

    return {"ingested": len(out_rows), "added": added, "replaced": replaced, "total_rows": len(store)}
