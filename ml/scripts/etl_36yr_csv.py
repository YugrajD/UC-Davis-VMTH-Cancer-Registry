"""Convert the 36-year archive CSV (1990-2025) into pipeline-ready CSVs.

Input schema (one row per case):
  Date Of Birth, Sex, Species, Breed, Zipcode Zipcode, RfrrVtrn Zipcode Zipcode,
  DtOfRq, Clinical Diagnoses, Text

Section markers in `Text` come in two flavors that coexist across the archive:
  - 1990-2014:  '|B|<SECTION>: ||'  (B, space before closing pipes)
  - 2015-2025:  '|H|<SECTION>:||'   (H, no space — same format as legacy exports)
The regex matches either letter and tolerates the optional whitespace.

Header normalization for `report_36yr.csv`:
  - 'COMMENT'                                 -> 'FINAL COMMENT'
  - 'MICROBIOLOGY' / 'VIROLOGY' / 'IMMUNOHISTOLOGY' / 'IMMUNOHISTOCHEMISTRY'
                                              -> 'ANCILLARY TESTS' (concatenated)

Outputs:
  ml/data/report_36yr.csv      — pipeline-ready (case_id + 3 text cols + meta)
  ml/data/diagnoses_36yr.csv   — case_id + diagnosis_number + diagnosis (for
                                 annotation later)
"""

import re
from pathlib import Path

import pandas as pd

INPUT = Path("/Users/yugrajdhillon/Downloads/all_years_1990_2025.csv")
OUT_REPORT = Path("ml/data/report_36yr.csv")
OUT_DIAG = Path("ml/data/diagnoses_36yr.csv")

# |H|HEADER:|| or |B|HEADER: || — letter is H or B, optional whitespace before close.
SECTION_RE = re.compile(r"\|[HB]\|([^|]+?):\s*\|\|")
# Inside Clinical Diagnoses cell, split on '\n' or numbered prefixes ('1.', '2.', …)
DIAG_NUM_RE = re.compile(r"(?:^|\n)\s*(\d+)\.\s*", re.MULTILINE)

WANTED_SECTIONS = ["HISTOPATHOLOGICAL SUMMARY", "FINAL COMMENT", "ANCILLARY TESTS"]
ANCILLARY_ALIASES = {
    "ANCILLARY TESTS",
    "MICROBIOLOGY",
    "VIROLOGY",
    "IMMUNOHISTOLOGY",
    "IMMUNOHISTOCHEMISTRY",
}
COMMENT_ALIASES = {"COMMENT", "FINAL COMMENT", "HISTOPATHOLOGICAL SUMMARY/COMMENT"}


def _parse_sections(full_text: str) -> dict[str, str]:
    """Split concatenated report text on '|[HB]|SECTION:||' markers."""
    if not full_text:
        return {}
    matches = list(SECTION_RE.finditer(full_text))
    out: dict[str, str] = {}
    for i, m in enumerate(matches):
        name = m.group(1).strip().upper()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        body = full_text[start:end].strip()
        # Drop a stray "Text" prefix some rows have leading the first marker.
        if body.startswith("Text "):
            body = body[5:].lstrip()
        if name in out:
            out[name] += "\n" + body
        else:
            out[name] = body
    return out


def _normalize_sections(raw: dict[str, str]) -> dict[str, str]:
    """Map raw section names into WANTED_SECTIONS."""
    out = {sec: "" for sec in WANTED_SECTIONS}
    ancillary_parts: list[str] = []
    comment_parts: list[str] = []
    for name, body in raw.items():
        if not body:
            continue
        if name == "HISTOPATHOLOGICAL SUMMARY":
            out["HISTOPATHOLOGICAL SUMMARY"] = body
        elif name in COMMENT_ALIASES:
            comment_parts.append(body)
        elif name in ANCILLARY_ALIASES:
            ancillary_parts.append(f"{name}: {body}")
    if comment_parts:
        out["FINAL COMMENT"] = "\n".join(comment_parts)
    if ancillary_parts:
        out["ANCILLARY TESTS"] = "\n".join(ancillary_parts)
    return out


def _split_diagnoses(cell: str) -> list[tuple[int, str]]:
    """Parse 'Clinical Diagnoses' cell into [(diag_number, text), ...]."""
    if pd.isna(cell):
        return []
    s = str(cell).strip()
    if not s:
        return []
    # Find numbered diagnoses
    matches = list(DIAG_NUM_RE.finditer(s))
    if not matches:
        # No numbering — treat the whole cell as diagnosis #1
        return [(1, s)]
    out: list[tuple[int, str]] = []
    for i, m in enumerate(matches):
        num = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(s)
        text = s[start:end].strip()
        if text:
            out.append((num, text))
    return out


def _make_case_id(year: int, idx: int) -> str:
    return f"Y{year}-{idx:05d}"


def main() -> int:
    print(f"Loading {INPUT}…")
    df = pd.read_csv(INPUT, encoding="latin-1", low_memory=False)
    df["DtOfRq"] = pd.to_datetime(df["DtOfRq"], errors="coerce")
    df["year"] = df["DtOfRq"].dt.year.fillna(0).astype(int)
    print(f"  rows: {len(df):,}")

    # Per-year sequential case_id index (stable, sortable, reflects intake order).
    df = df.sort_values(["year", "DtOfRq"], kind="stable").reset_index(drop=True)
    df["case_idx_in_year"] = df.groupby("year").cumcount() + 1
    df["case_id"] = df.apply(
        lambda r: _make_case_id(int(r["year"]), int(r["case_idx_in_year"])), axis=1
    )

    # ---- Build report_36yr.csv -----------------------------------------------
    report_rows: list[dict] = []
    for _, row in df.iterrows():
        text = row["Text"] if pd.notna(row["Text"]) else ""
        sections = _normalize_sections(_parse_sections(text))
        report_rows.append(
            {
                "case_id": row["case_id"],
                "year": int(row["year"]) if row["year"] else None,
                "dt_of_rq": row["DtOfRq"],
                "sex": row["Sex"],
                "species": row["Species"],
                "breed": row["Breed"],
                "full_text": text,
                "HISTOPATHOLOGICAL SUMMARY": sections["HISTOPATHOLOGICAL SUMMARY"],
                "FINAL COMMENT": sections["FINAL COMMENT"],
                "ANCILLARY TESTS": sections["ANCILLARY TESTS"],
            }
        )
    report_df = pd.DataFrame(report_rows)
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(OUT_REPORT, index=False)
    print(f"Wrote {OUT_REPORT}  cases={len(report_df):,}")

    # Summary stats
    for col in WANTED_SECTIONS:
        n = (report_df[col].str.len() > 0).sum()
        print(f"  {col:<30} populated: {n:>6,}  ({n/len(report_df)*100:.1f}%)")

    # ---- Build diagnoses_36yr.csv --------------------------------------------
    diag_rows: list[dict] = []
    for _, row in df.iterrows():
        for num, text in _split_diagnoses(row["Clinical Diagnoses"]):
            diag_rows.append(
                {
                    "case_id": row["case_id"],
                    "diagnosis_number": num,
                    "diagnosis": text,
                }
            )
    diag_df = pd.DataFrame(
        diag_rows, columns=["case_id", "diagnosis_number", "diagnosis"]
    )
    diag_df.to_csv(OUT_DIAG, index=False)
    print(
        f"Wrote {OUT_DIAG}  rows={len(diag_df):,}  "
        f"cases_with_diag={diag_df['case_id'].nunique():,}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
