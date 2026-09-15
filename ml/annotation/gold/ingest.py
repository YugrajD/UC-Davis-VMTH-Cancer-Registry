"""Ingest a filled review CSV into the gold annotation store.

Reads the filled review CSV, validates verdict values, resolves confirmed
labels, and writes GOLD_ANNOTATION_CSV.

Schema of GOLD_ANNOTATION_CSV:
  case_id, diagnosis_number, diagnosis,
  matched_term, matched_group, matched_code, matched_keyword, method, confidence,
  tier, verified_by, verified_date, provenance,
  decision_stage, sample_stratum, sample_weight, verdict, notes

The reviewer's sheet has one fill-in column, so the verdict is **derived** rather than
stated. Joined to the key CSV on `row_id`:

  Actual Diagnosis blank + a Predicted Match  → correct   (copy the cascade's answer)
  Actual Diagnosis blank + `(none)` predicted → no_cancer (empty matched_*)
  Actual Diagnosis = `unclear`                → uncertain (empty matched_*)
  Actual Diagnosis = a taxonomy term          → wrong     (resolved term/group/code)
  anything else                               → abort, with a per-row report

`unclear` is checked before taxonomy validation, or it would trip the abort.

The review CSV has no dropdowns, so a `wrong` row whose confirmed term/group is
not in the taxonomy is a validation error rather than a silently code-less gold
row.

The gold store is **cumulative**: batches 2+ merge into whatever is already there,
keyed on (case_id, diagnosis_number). Re-ingesting a row replaces it, so fixing a
reviewer's typo means editing the review CSV and re-running. Pass `--replace-store`
to start over from empty instead.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

from ICD_labels.taxonomy import load_labels_taxonomy

from .csv_io import read_key_rows, read_review_rows

VALID_VERDICTS = frozenset({"correct", "wrong", "no_cancer", "uncertain"})

# Written into `Predicted Match` by `sample` when the cascade concluded "no cancer".
NO_PREDICTION = "(none)"

# What a reviewer writes when the diagnosis line is too thin to place. Matched
# case-insensitively and before the taxonomy lookup, so it never reads as a bad term.
UNCLEAR_MARKERS = frozenset({"unclear", "uncertain", "unsure", "?"})

_GOLD_FIELDNAMES = [
    "case_id", "diagnosis_number", "diagnosis",
    "matched_term", "matched_group", "matched_code",
    "matched_keyword", "method", "confidence",
    "tier", "verified_by", "verified_date", "provenance",
    # Carried from the review CSV: the audit deliberately over-samples the hard
    # strata, so without these no rate computed off this store can be weighted
    # back to the Tier-3 population.
    "decision_stage", "sample_stratum", "sample_weight",
    # The reviewer's raw judgement. Not redundant with matched_term: `no_cancer`,
    # `uncertain`, and `correct`-on-a-blank-suggestion all resolve to empty
    # matched_* fields, so without this column the three are indistinguishable and
    # the audit's per-stratum questions ("is hedged wording genuinely
    # unclassifiable?", "how many No Match rows hide a real cancer?") cannot be
    # answered from the store at all.
    "verdict", "notes",
]


def _taxonomy_index(labels_csv: str):
    """Build lookup maps for resolving a confirmed (group, term) to its ICD-O code.

    Keys are lowercased so a reviewer's casing never causes a spurious rejection;
    the values keep the taxonomy's own spelling, which is what reaches the gold
    store. See `_resolve_wrong`.
    """
    by_group_term: dict[tuple[str, str], tuple[str, str, str]] = {}
    by_term: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for lbl in load_labels_taxonomy(labels_csv):
        entry = (lbl.group, lbl.term, lbl.code)
        by_group_term[(lbl.group.lower(), lbl.term.lower())] = entry
        by_term[lbl.term.lower()].append(entry)
    return by_group_term, by_term


def _resolve_wrong(group: str, term: str, by_group_term, by_term) -> tuple[str, str, str, str]:
    """Return (matched_group, matched_term, matched_code, error) for a corrected label.

    The group and term returned are the **taxonomy's** spelling, not the reviewer's.
    Matching is case-insensitive, but `evaluate.py` compares terms by exact string
    membership, so storing `papillary adenocarcinoma` where the taxonomy says
    `Papillary adenocarcinoma` would make a correct gold row score as a miss forever.
    """
    gt = (group.lower(), term.lower())
    if gt in by_group_term:
        c_group, c_term, c_code = by_group_term[gt]
        return c_group, c_term, c_code, ""
    # Group left blank or not matching: fall back to term lookup.
    cands = by_term.get(term.lower(), [])
    if not cands:
        return "", "", "", (f"'Actual Diagnosis' {term!r} is not a term in the taxonomy — "
                            f"it must be copied exactly from the Term column")
    groups = sorted({g for g, _, _ in cands})
    if len(groups) == 1:
        c_group, c_term, c_code = cands[0]
        return c_group, c_term, c_code, ""   # unambiguous term → backfill group + code
    return "", "", "", (f"'Actual Diagnosis' {term!r} is ambiguous — it exists in more than "
                        f"one group ({groups}), so it cannot be coded from the term alone")


def _read_existing_store(out_path: Path) -> dict[tuple[str, str], dict]:
    """Load the current gold store keyed on (case_id, diagnosis_number)."""
    if not out_path.exists():
        return {}
    with open(out_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    unknown = set(rows[0]) - set(_GOLD_FIELDNAMES) if rows else set()
    if unknown:
        print(
            f"ERROR — existing gold store {out_path} has unexpected columns {sorted(unknown)}; "
            f"refusing to merge into it. Move it aside or pass --replace-store.",
            file=sys.stderr,
        )
        sys.exit(1)
    return {(r.get("case_id", ""), r.get("diagnosis_number", "")): r for r in rows}


def ingest(
    review_csv: str,
    key_csv: str,
    out_csv: str,
    verified_by: str,
    labels_csv: str,
    provenance: str = "round0",
    replace_store: bool = False,
) -> None:
    """Read a filled review CSV and merge it into the gold annotation store.

    Merging (rather than overwriting) is the whole point: the plan issues the audit
    in 200-row batches, and a plain rewrite would silently destroy every earlier
    batch's human review the moment batch 2 landed.
    """
    today = date.today().isoformat()
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    by_group_term, by_term = _taxonomy_index(labels_csv)

    if not Path(key_csv).exists():
        print(f"ERROR — no key CSV at {key_csv}. It is written alongside the review CSV "
              f"by `sample`, and carries the case identity the reviewer's copy omits.",
              file=sys.stderr)
        sys.exit(1)
    key = read_key_rows(key_csv)

    errors: list[str] = []
    out_rows: list[dict] = []
    seen_row_ids: set[str] = set()

    for idx, row in enumerate(read_review_rows(review_csv), start=2):  # +1 header
        row_id = row.get("row_id", "").strip()
        if not row_id:
            errors.append(f"Row {idx}: row_id is blank — the join column was edited or deleted")
            continue
        if row_id in seen_row_ids:
            errors.append(f"Row {idx}: row_id {row_id!r} appears more than once")
            continue
        seen_row_ids.add(row_id)
        if row_id not in key:
            errors.append(f"Row {idx}: row_id {row_id!r} is not in {key_csv} — the review "
                          f"CSV and the key are from different batches")
            continue

        k = key[row_id]
        cid = k.get("case_id", "")
        actual = row.get("Actual Diagnosis", "").strip()
        predicted = (row.get("Predicted Match", "").strip()
                     or k.get("cascade_matched_term", "").strip())
        has_prediction = bool(k.get("cascade_matched_term", "").strip())

        if not actual:
            # Blank means "the prediction is right". With one fill-in column that also
            # makes a skipped row look like an agreed one — accepted for the pilot, where
            # the batch is short enough to confirm how far the reviewer actually got.
            verdict = "correct" if has_prediction else "no_cancer"
            matched_term = k.get("cascade_matched_term", "") if has_prediction else ""
            matched_group = k.get("cascade_matched_group", "") if has_prediction else ""
            matched_code = k.get("cascade_matched_code", "") if has_prediction else ""
        elif actual.lower().rstrip(".") in UNCLEAR_MARKERS:
            verdict = "uncertain"
            matched_term = matched_group = matched_code = ""
        else:
            verdict = "wrong"
            matched_group, matched_term, matched_code, err = _resolve_wrong(
                "", actual, by_group_term, by_term)
            if err:
                errors.append(f"Row {idx}: {err} (row_id={row_id}, case_id={cid})")
                continue

        out_rows.append({
            "case_id": cid,
            "diagnosis_number": k.get("diagnosis_number", ""),
            "diagnosis": k.get("diagnosis", ""),
            "matched_term": matched_term,
            "matched_group": matched_group,
            "matched_code": matched_code,
            "matched_keyword": "",
            "method": k.get("cascade_method", ""),
            "confidence": "",
            "tier": "gold",
            "verified_by": verified_by,
            "verified_date": today,
            "provenance": provenance,
            "decision_stage": k.get("decision_stage", ""),
            "sample_stratum": k.get("sample_stratum", ""),
            "sample_weight": k.get("sample_weight", ""),
            "verdict": verdict,
            # Verbatim record of what the reviewer typed, kept even when it resolved
            # cleanly — the gold store should show their wording, not just our mapping.
            "notes": actual,
        })

    if errors:
        print("ERROR — ingestion aborted due to validation failures:", file=sys.stderr)
        for msg in errors:
            print(f"  {msg}", file=sys.stderr)
        sys.exit(1)

    store = {} if replace_store else _read_existing_store(out_path)
    pre_existing = len(store)

    added = replaced = 0
    for row in out_rows:
        key = (row["case_id"], row["diagnosis_number"])
        if key in store:
            replaced += 1
        else:
            added += 1
        store[key] = row

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_GOLD_FIELDNAMES)
        writer.writeheader()
        writer.writerows(store.values())

    if replace_store and pre_existing:
        print(f"--replace-store: discarded {pre_existing} pre-existing gold row(s).")
    print(
        f"Ingested {len(out_rows)} reviewed row(s): {added} new, {replaced} re-reviewed. "
        f"Gold store now holds {len(store)} row(s) across "
        f"{len({k[0] for k in store})} case(s): {out_csv}"
    )
