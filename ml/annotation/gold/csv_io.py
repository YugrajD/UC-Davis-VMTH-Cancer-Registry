"""Read/write the CSV review surface for gold annotation.

``sample`` emits three plain-text files side by side:
  - the review CSV     one row per diagnosis.  The cascade suggestion is
                       prefilled; the reviewer fills verdict / confirmed_term /
                       confirmed_group / notes.  Diagnosis text only — see
                       ``sample`` for why no report sections are included.
  - instructions .md   how to fill it in.
  - taxonomy CSV       the full (Group, Term, Code) reference to pick from.

A CSV has no dropdowns, so typo-catching moves downstream: ``ingest`` validates
every confirmed term/group against the taxonomy and refuses to write the gold
store if any row is unresolvable.

``read_review_rows`` returns the review CSV as a list of dicts keyed by column
header, with every value a string.
"""

from __future__ import annotations

import csv
from pathlib import Path

# utf-8-sig: a double-click still opens cleanly in Excel without mangling accents.
_ENCODING = "utf-8-sig"

_INSTRUCTIONS = """\
# Tier-3 audit review — instructions

Open `tier3_audit_review.csv` in Excel, LibreOffice, or any spreadsheet tool.
Each row is one diagnosis, and **every row here is one the pipeline found hard** —
there is no filler to skim past. Rows are independent; you can stop anywhere.

Every column except the four you fill in is **read-only context**.

## What you are being asked

`cascade_matched_term` / `_group` / `_code` are the label the pipeline settled on
(blank means it settled on "no cancer"). `decision_stage` and `cascade_method`
tell you how it got there — and that changes the question you're answering:

| decision_stage | cascade_method | what happened | your question |
| --- | --- | --- | --- |
| `tier3_llm` | `LLM` | the model picked this term from a shortlist | is the term right? |
| `tier3_llm` | `No Match` | the model was asked and **refused to label it** | **is there a cancer here it missed?** |
| `tier3_llm` | `Uncertain` | the model judged the wording too hedged | is it genuinely unclassifiable? |
| `tier3_no_candidates` | `No Match` | cancer wording present, but the pipeline built no shortlist, so **the model was never asked** | **is there a cancer here it missed?** |
| `tier2_fuzzy` | `Fuzzy` | matched on partial word overlap, not an exact term | is this really the same disease? |

The `No Match` rows are the point of this batch. The pipeline currently drops
those diagnoses silently as non-cancer, and nobody has ever checked whether that
is right. If one of them *is* a reportable neoplasm, mark it `wrong` and give the
correct label — that single row is worth more than a hundred easy confirmations.

`sample_stratum` and `sample_weight` are bookkeeping for the statistics. Ignore them.

## Filling it in

**Judge each row on the `diagnosis` text alone.** That single line is the only
thing the annotation pipeline is given, so the question is always "is this the
right label *for this wording*" — not "is this the right label for the patient".
Do not consult the wider report, the case history, or other rows of the same
case, even if you have them to hand. Negation ("no evidence of neoplasia") and
hedging ("suspected", "consistent with") count only when they appear in the
diagnosis line itself.

For every row, fill in the **verdict** column with exactly one of:

| verdict | meaning | also fill in |
| --- | --- | --- |
| `correct` | the pipeline got it right — including when it correctly labelled nothing | nothing |
| `wrong` | a cancer label applies, but the pipeline's answer isn't it | `confirmed_group` **and** `confirmed_term` |
| `no_cancer` | this diagnosis is not a reportable neoplasm | nothing |
| `uncertain` | hedged, or cannot be determined from the diagnosis text | nothing |

On a blank `No Match` row, `correct` and `no_cancer` mean nearly the same thing —
prefer `no_cancer`, and reserve `correct` for rows where the pipeline proposed a
term you agree with.

If the diagnosis line is too thin to place confidently, that is `uncertain` —
it is a real finding about the pipeline's input, not a failure on your part.

`confirmed_group` and `confirmed_term` must be copied exactly from
`tier3_audit_taxonomy.csv` — spelling matters, because there are no dropdowns to
catch a typo. Ingestion fails with a per-row report if a term is not found.
The ICD-O code is derived automatically from your group + term, so there is no
code column to fill in.

The `notes` column is free text and is not parsed — use it for anything the
verdict can't express.

**Do not leave a verdict blank**, and do not reorder, rename, or delete columns.
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


def write_instructions(path: str) -> None:
    """Write the reviewer-facing instructions sidecar."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_INSTRUCTIONS, encoding="utf-8")


def write_taxonomy_csv(path: str, taxonomy_rows: list[tuple[str, str, str]]) -> None:
    """Write the (Group, Term, Code) reference the reviewer picks corrections from."""
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
