"""
Pre-process visit CSVs: merge HTML-like continuation rows into single
patient records, concatenating Text (pathology reports) and Clinical
Diagnoses across continuation rows.

The uploaded file may have one 'header row' per patient followed by zero or
more 'continuation rows' where all demographic fields are NA and only the
Text and/or Clinical Diagnoses columns carry additional content. This
script collapses each group into a single row, mirroring the server-side
merge in backend/app/routers/ingest.py:_merge_continuation_rows but
extending it to also concatenate Clinical Diagnoses.

A row is a *patient header* when at least one of these sentinel columns has
a real value: Date of Birth, Sex, Species, Breed. Otherwise it is a
continuation row appended to the preceding patient. Orphan continuation
rows (before any header) are dropped.

Usage:
    python scripts/preprocess_visits.py INPUT.csv OUTPUT.csv
"""

import csv
import re
import sys
from pathlib import Path


# Column name aliases → canonical name (matches backend/app/routers/ingest.py)
_COLUMN_RENAMES = {
    "diagnoses (labels)": "Clinical Diagnoses",
    "diagnoses": "Clinical Diagnoses",
    "clinical diagnoses": "Clinical Diagnoses",
    "text (pathology report)": "Text",
    "text": "Text",
}

# At least one must be non-NA for a row to be treated as a patient header.
_HEADER_SENTINEL_COLS = ("Date of Birth", "Sex", "Species", "Breed")

# Columns concatenated from continuation rows into the preceding header.
_CONTINUATION_COLS = ("Text", "Clinical Diagnoses")

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _is_na(value) -> bool:
    if value is None:
        return True
    s = str(value).strip()
    return s == "" or s.lower() in ("na", "nan")


def _strip_html(s: str) -> str:
    return _HTML_TAG_RE.sub("", s).strip()


def _canonicalize_headers(fieldnames):
    seen = set(fieldnames)
    rename: dict[str, str] = {}
    new_names: list[str] = []
    for col in fieldnames:
        canonical = _COLUMN_RENAMES.get(col.strip().lower())
        if canonical and canonical != col and canonical not in seen:
            rename[col] = canonical
            seen.add(canonical)
            new_names.append(canonical)
        else:
            new_names.append(col)
    return new_names, rename


def main():
    if len(sys.argv) != 3:
        print(
            "Usage: python scripts/preprocess_visits.py INPUT.csv OUTPUT.csv",
            file=sys.stderr,
        )
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    with open(input_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            print(f"Error: no header row in {input_path}", file=sys.stderr)
            sys.exit(1)

        fieldnames, rename_map = _canonicalize_headers(reader.fieldnames)
        sentinel_present = [c for c in _HEADER_SENTINEL_COLS if c in fieldnames]
        cont_cols = [c for c in _CONTINUATION_COLS if c in fieldnames]

        if not sentinel_present:
            print(
                f"Warning: none of {_HEADER_SENTINEL_COLS} found — nothing to merge.",
                file=sys.stderr,
            )
        if not cont_cols:
            print(
                f"Warning: none of {_CONTINUATION_COLS} found — no columns to concatenate.",
                file=sys.stderr,
            )

        patients: list[dict] = []
        current: dict | None = None
        current_merged = 0
        raw_count = 0
        orphan_count = 0
        multi_visit_count = 0

        for row in reader:
            raw_count += 1
            for old, new in rename_map.items():
                if old in row:
                    row[new] = row.pop(old)

            is_header = any(not _is_na(row.get(c)) for c in sentinel_present)

            if is_header:
                if current is not None:
                    patients.append(current)
                    if current_merged > 0:
                        multi_visit_count += 1
                current = dict(row)
                current_merged = 0
                for col in cont_cols:
                    if not _is_na(current.get(col)):
                        current[col] = _strip_html(str(current[col]))
            else:
                if current is None:
                    orphan_count += 1
                    continue
                merged_any = False
                for col in cont_cols:
                    if not _is_na(row.get(col)):
                        fragment = _strip_html(str(row[col]))
                        if not fragment:
                            continue
                        existing = str(current.get(col, "") or "").strip()
                        current[col] = (
                            f"{existing} {fragment}".strip() if existing else fragment
                        )
                        merged_any = True
                if merged_any:
                    current_merged += 1

        if current is not None:
            patients.append(current)
            if current_merged > 0:
                multi_visit_count += 1

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in patients:
            writer.writerow({k: p.get(k, "") for k in fieldnames})

    print(
        f"Read {raw_count} rows, wrote {len(patients)} patient records to {output_path}"
    )
    print(f"  {multi_visit_count} patients had continuation rows merged")
    if orphan_count:
        print(f"  Skipped {orphan_count} orphan continuation rows (before first header)")


if __name__ == "__main__":
    main()
