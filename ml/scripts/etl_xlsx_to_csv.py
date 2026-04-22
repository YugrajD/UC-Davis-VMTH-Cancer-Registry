"""Convert raw diagnostics xlsx exports into ml/data/{diagnoses,report}.csv.

Input xlsx schema (same for both files):
  DtOfRq, Sex, Species, Breed, Diagnoses (labels), Text (pathology report)

Case blocks: each case starts at a row whose 'Diagnoses (labels)' cell begins
with '1.'. Subsequent rows belong to the same case until the next '1.' row.
Within a block, additional non-null diag rows ('2.', '3.', ...) are extra
diagnoses for the same case; rows with null diag hold the continuation lines
of the case's pathology report text.

Report text uses '|H|<SECTION>:||' markers for section headers — we split
on those to build report.csv with columns HISTOPATHOLOGICAL SUMMARY,
FINAL COMMENT, ANCILLARY TESTS (plus the full merged text).
"""

import re
import sys
from pathlib import Path

import pandas as pd

INPUTS = [
    Path("/Users/yugrajdhillon/Downloads/2017-2023 diagnostics.xlsx"),
    Path("/Users/yugrajdhillon/Downloads/2024-2025 diagnostics.xlsx"),
]
OUT_DIAG = Path("ml/data/diagnoses.csv")
OUT_REPORT = Path("ml/data/report.csv")

DIAG_NUM_RE = re.compile(r"^\s*(\d+)\.\s*(.*)$", re.DOTALL)
SECTION_RE = re.compile(r"\|H\|([^|]+?):\|\|")

# Sections we want as their own columns in report.csv. Names match
# ml/model/constants.py::DEFAULT_TEXT_COLS.
WANTED_SECTIONS = [
    "HISTOPATHOLOGICAL SUMMARY",
    "FINAL COMMENT",
    "ANCILLARY TESTS",
    "CLINICAL ABSTRACT",
    "GROSS DESCRIPTION",
]


def load_all() -> pd.DataFrame:
    frames = [pd.read_excel(p) for p in INPUTS]
    df = pd.concat(frames, ignore_index=True)
    return df


def _clean_text_cell(val) -> str:
    if pd.isna(val):
        return ""
    s = str(val)
    if s == "NaT":
        return ""
    return s


def _parse_sections(full_text: str) -> dict[str, str]:
    """Split concatenated report text on '|H|SECTION:||' markers."""
    if not full_text:
        return {}
    matches = list(SECTION_RE.finditer(full_text))
    out: dict[str, str] = {}
    for i, m in enumerate(matches):
        name = m.group(1).strip().upper()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        body = full_text[start:end].strip()
        out[name] = body
    return out


def main() -> int:
    df = load_all()
    print(f"Loaded {len(df):,} xlsx rows from {len(INPUTS)} files")

    diag_rows: list[dict] = []
    cases: dict[int, dict] = {}
    case_id = 0
    current_case: dict | None = None
    text_buf: list[str] = []

    def _flush_text() -> None:
        if current_case is None:
            return
        merged = " ".join(t for t in text_buf if t).strip()
        current_case["_text"] = merged

    for _, row in df.iterrows():
        diag_cell = row["Diagnoses (labels)"]
        text_cell = _clean_text_cell(row["Text (pathology report)"])

        if pd.notna(diag_cell):
            m = DIAG_NUM_RE.match(str(diag_cell))
            if not m:
                # Non-numbered diag — treat as extra diagnosis of current case.
                diag_num = None
                diag_text = str(diag_cell).strip()
            else:
                diag_num = int(m.group(1))
                diag_text = m.group(2).strip()

            if diag_num == 1 or current_case is None:
                # Start new case.
                _flush_text()
                case_id += 1
                current_case = {
                    "case_id": case_id,
                    "dt_of_rq": row.get("DtOfRq"),
                    "sex": row.get("Sex"),
                    "species": row.get("Species"),
                    "breed": row.get("Breed"),
                }
                cases[case_id] = current_case
                text_buf = []

            diag_rows.append({
                "case_id": current_case["case_id"],
                "diagnosis_number": diag_num if diag_num is not None else len(
                    [d for d in diag_rows if d["case_id"] == current_case["case_id"]]
                ) + 1,
                "diagnosis": diag_text,
            })
            # The text cell on a diag row is usually the literal header
            # artifact "Text" — skip it if so.
            if text_cell and text_cell != "Text":
                text_buf.append(text_cell)
        else:
            if text_cell and current_case is not None:
                text_buf.append(text_cell)

    _flush_text()

    # --- Build diagnoses.csv --------------------------------------------
    diag_df = pd.DataFrame(diag_rows, columns=["case_id", "diagnosis_number", "diagnosis"])
    OUT_DIAG.parent.mkdir(parents=True, exist_ok=True)
    diag_df.to_csv(OUT_DIAG, index=False)
    print(f"Wrote {OUT_DIAG}  rows={len(diag_df):,}  cases={diag_df['case_id'].nunique():,}")

    # --- Build report.csv -----------------------------------------------
    report_rows: list[dict] = []
    for cid, meta in cases.items():
        full_text = meta.get("_text", "")
        sections = _parse_sections(full_text)
        row = {
            "case_id": cid,
            "dt_of_rq": meta.get("dt_of_rq"),
            "sex": meta.get("sex"),
            "species": meta.get("species"),
            "breed": meta.get("breed"),
            "full_text": full_text,
        }
        for sec in WANTED_SECTIONS:
            row[sec] = sections.get(sec, "")
        report_rows.append(row)

    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(OUT_REPORT, index=False)
    print(f"Wrote {OUT_REPORT}  cases={len(report_df):,}")

    # Summary stats
    with_hp = (report_df["HISTOPATHOLOGICAL SUMMARY"].str.len() > 0).sum()
    with_fc = (report_df["FINAL COMMENT"].str.len() > 0).sum()
    with_at = (report_df["ANCILLARY TESTS"].str.len() > 0).sum()
    print(f"  HISTOPATHOLOGICAL SUMMARY populated: {with_hp:,}")
    print(f"  FINAL COMMENT populated            : {with_fc:,}")
    print(f"  ANCILLARY TESTS populated          : {with_at:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
