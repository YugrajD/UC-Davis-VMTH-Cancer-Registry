"""Split a review CSV into a short pilot and the remainder.

Every failure mode left in the audit after the tooling is verified is a *human*
one, and none of them are recoverable after the fact: if the reviewer reads a
question differently than intended, all 200 rows are contaminated the same way and
nothing in the store reveals it. The pilot is the cheap guard — the professional
fills in ~30 rows, we ingest and read them together, and only then do they spend a
sitting on the other 170.

The split is **stratified**, because the review CSV is written grouped by stratum:
taking the literal first N rows would hand the reviewer one stratum and exercise
one of the five questions. Allocation is proportional with largest-remainder
rounding, floored so no stratum drops out of the pilot entirely.

The stratum is read from the **key CSV**, not the reviewer's copy — the reviewer's
sheet deliberately carries no sampling bookkeeping.

Rows keep their original order within each stratum, and `pilot + remainder` is
exactly the input CSV — no row is duplicated or lost. `ingest` is cumulative and
keyed on (case_id, diagnosis_number), so ingesting the pilot and later the
remainder builds one store.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .csv_io import read_key_rows, write_instructions

_STRATUM_COL = "sample_stratum"
_ROW_ID = "row_id"

_PILOT_PREAMBLE = """\
> **This is a short first pass — {n} rows, not the full batch.**
>
> The point is to check that these instructions are clear before you spend a full
> sitting on the rest. Fill these in, send them back, and we will go through them
> together; the remaining rows follow once we agree the questions are landing the
> way they read. If anything below is unclear or awkward, please just tell us when
> you send it back — that is a useful result, not a complaint."""


def _allocate(counts: dict[str, int], n_rows: int, min_per_stratum: int) -> dict[str, int]:
    """Proportional allocation with largest-remainder rounding and a per-stratum floor."""
    total = sum(counts.values())
    if n_rows > total:
        print(f"ERROR — asked for {n_rows} pilot rows but the review CSV only has {total}.",
              file=sys.stderr)
        sys.exit(1)
    strata = sorted(counts)
    floor_needed = sum(min(min_per_stratum, counts[s]) for s in strata)
    if floor_needed > n_rows:
        print(f"ERROR — {n_rows} pilot rows cannot cover {len(strata)} strata at "
              f"--min-per-stratum {min_per_stratum} (needs {floor_needed}).", file=sys.stderr)
        sys.exit(1)

    exact = {s: n_rows * counts[s] / total for s in strata}
    alloc = {s: min(counts[s], max(int(exact[s]), min(min_per_stratum, counts[s]))) for s in strata}

    # Hand out (or claw back) the rounding slack, largest remainder first.
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
    review_csv: str,
    key_csv: str,
    out_pilot_csv: str,
    out_remainder_csv: str,
    n_rows: int = 30,
    min_per_stratum: int = 3,
    out_instructions_md: str | None = None,
) -> None:
    """Write a stratified pilot slice of `review_csv` plus everything left over."""
    src = Path(review_csv)
    if not src.exists():
        print(f"ERROR — no review CSV at {review_csv}. Run `sample` first.", file=sys.stderr)
        sys.exit(1)
    with open(src, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows:
        print(f"ERROR — review CSV {review_csv} has no rows.", file=sys.stderr)
        sys.exit(1)
    if _ROW_ID not in fieldnames:
        print(f"ERROR — review CSV {review_csv} has no {_ROW_ID!r} column.", file=sys.stderr)
        sys.exit(1)

    if not Path(key_csv).exists():
        print(f"ERROR — no key CSV at {key_csv}. It is written alongside the review CSV "
              f"by `sample`.", file=sys.stderr)
        sys.exit(1)
    key = read_key_rows(key_csv)
    missing = [r[_ROW_ID] for r in rows if r[_ROW_ID] not in key]
    if missing:
        print(f"ERROR — {len(missing)} row_id(s) in {review_csv} are absent from {key_csv} "
              f"(first few: {missing[:5]}). The two files are from different batches.",
              file=sys.stderr)
        sys.exit(1)

    filled = [r for r in rows if (r.get("Actual Diagnosis") or "").strip()]
    if filled:
        print(f"ERROR — {len(filled)} row(s) in {review_csv} are already filled in. "
              f"Split the batch before it is reviewed, not after.", file=sys.stderr)
        sys.exit(1)

    stratum_of = {r[_ROW_ID]: key[r[_ROW_ID]].get(_STRATUM_COL, "") for r in rows}
    counts = Counter(stratum_of.values())
    alloc = _allocate(dict(counts), n_rows, min_per_stratum)

    taken: dict[str, int] = defaultdict(int)
    pilot_rows, remainder_rows = [], []
    for r in rows:                                   # preserves within-stratum order
        s = stratum_of[r[_ROW_ID]]
        if taken[s] < alloc.get(s, 0):
            taken[s] += 1
            pilot_rows.append(r)
        else:
            remainder_rows.append(r)

    # The batch must survive the split intact: no row duplicated, none dropped.
    assert len(pilot_rows) + len(remainder_rows) == len(rows)
    assert len(pilot_rows) == n_rows

    for path, out_rows in ((out_pilot_csv, pilot_rows), (out_remainder_csv, remainder_rows)):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(out_rows)

    print(f"Split {len(rows)} review rows into a {len(pilot_rows)}-row pilot "
          f"and a {len(remainder_rows)}-row remainder.\n")
    print("Rows per stratum (pilot / batch):")
    for s in sorted(counts, key=lambda s: -counts[s]):
        print(f"  {s:<24} {alloc.get(s, 0):>3} / {counts[s]:<3}")
    # The shared sidecar names the full review CSV, which is not the file the reviewer
    # is being sent. Emit a pilot-specific copy so the handoff is self-consistent.
    if out_instructions_md:
        write_instructions(
            out_instructions_md,
            review_filename=Path(out_pilot_csv).name,
            preamble=_PILOT_PREAMBLE.format(n=n_rows),
        )

    print(f"\nPilot:     {out_pilot_csv}")
    print(f"Remainder: {out_remainder_csv}")
    if out_instructions_md:
        print(f"Pilot instructions: {out_instructions_md}")
    print("\nSend the reviewer the pilot CSV + its instructions + tier3_audit_taxonomy.csv.")
    print("Ingest it, read it together, then release the remainder — both build one store.")
