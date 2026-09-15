"""Read/write the review surface for gold annotation.

The reviewer's sheet is deliberately **three working columns**: the clinical
diagnosis, what the pipeline predicted, and — only when the prediction is wrong —
what the diagnosis actually is. Everything else the audit needs (case_id, stratum,
sampling weight, decision stage, the cascade's group/code) lives in a separate
**key CSV** that stays with us and is joined back on `row_id` at ingest.

That separation is the point. A reviewer reading eleven columns of pipeline
internals is being asked to audit the pipeline; a reviewer reading two is being
asked a clinical question, which is the only thing they are actually expert in.

Corrections are **exact taxonomy terms**, copied from the taxonomy sidecar. Free
text was measured and rejected: on Tier-2/Tier-3 rows the diagnosis wording contains
a taxonomy term ~0% of the time (25% on `tier1_exact`), so free text would not
self-resolve and every correction would need a second round-trip with the clinician.
There are no dropdowns in a CSV, so `ingest` validates every term against the
taxonomy and refuses to write the gold store if any row is unresolvable.
"""

from __future__ import annotations

import csv
from pathlib import Path

# utf-8-sig: a double-click still opens cleanly in Excel without mangling accents.
_ENCODING = "utf-8-sig"

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


def write_review_csv(path: str, header: list[str], rows: list[dict]) -> None:
    """Write the review CSV the professional fills in."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding=_ENCODING) as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def write_instructions(
    path: str,
    review_filename: str = "tier3_audit_review.csv",
    preamble: str = "",
) -> None:
    """Write the reviewer-facing instructions sidecar."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = (_INSTRUCTIONS
            .replace("__PREAMBLE__", f"{preamble}\n\n" if preamble else "")
            .replace("__REVIEW_CSV__", review_filename))
    out.write_text(text, encoding="utf-8")


def write_taxonomy_csv(path: str, taxonomy_rows: list[tuple[str, str, str]]) -> None:
    """Write the (Group, Term, Code) reference.

    No longer part of the reviewer handoff — corrections are free text now. Kept for
    our own reconciliation of what the reviewer wrote.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding=_ENCODING) as f:
        writer = csv.writer(f)
        writer.writerow(["Group", "Term", "Code"])
        writer.writerows(taxonomy_rows)


def read_review_rows(path: str) -> list[dict]:
    """Return the filled review CSV as a list of header-keyed dicts (values as str)."""
    with open(path, encoding=_ENCODING) as f:
        return [
            {k: ("" if v is None else str(v)) for k, v in row.items()}
            for row in csv.DictReader(f)
        ]


def write_key_csv(path: str, header: list[str], rows: list[dict]) -> None:
    """Write the internal key CSV that joins `row_id` back to the corpus.

    Never sent to the reviewer: this is every column the audit needs and they do not —
    case identity, the cascade's full answer, and the sampling bookkeeping.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding=_ENCODING) as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def read_key_rows(path: str) -> dict[str, dict]:
    """Return the key CSV indexed by `row_id`."""
    with open(path, encoding=_ENCODING) as f:
        rows = [
            {k: ("" if v is None else str(v)) for k, v in row.items()}
            for row in csv.DictReader(f)
        ]
    return {r["row_id"]: r for r in rows}
