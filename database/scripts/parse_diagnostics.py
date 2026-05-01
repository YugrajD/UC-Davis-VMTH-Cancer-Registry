#!/usr/bin/env python3
"""
Parse all per-year CSVs in database/data/input/ into three output files:
  - demographics.csv
  - diagnoses.csv
  - report.csv
"""

import csv
import glob
import re
import os
from collections import OrderedDict

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'output')

# Canonical columns written as dedicated columns in report.csv.
# All other sections are serialised into ADDITIONAL INFORMATION.
CANONICAL_HEADINGS = [
    'CLINICAL ABSTRACT',
    'GROSS DESCRIPTION',
    'HISTOPATHOLOGICAL SUMMARY',
    'COMMENT',
    'FINAL COMMENT',
    'ANCILLARY TESTS',
    'ADDENDUM',
]
CANONICAL_SET = set(CANONICAL_HEADINGS)

# Map known typos/variants to canonical heading names
HEADING_NORMALIZATIONS = {
    # Pre-existing normalizations
    'CLINICAL ABS TRACT': 'CLINICAL ABSTRACT',
    'COM/MENTS AND COMMUNICATIONS': 'COMMENTS AND COMMUNICATIONS',
    'FINAL COMMENTS': 'FINAL COMMENT',
    'GROSS FINDINGS': 'GROSS DESCRIPTION',
    'GROSS NECROPSY FINDINGS': 'GROSS DESCRIPTION',
    'ANCILLARY TESTING': 'ANCILLARY TESTS',
    'ADDITIONAL TESTS': 'ANCILLARY TESTS',
    'FINAL COPLOW DIAGNOSES>': 'FINAL COPLOW DIAGNOSES',
    # Typos and OCR errors
    'HISTOPATHOLOGY SUMMARY': 'HISTOPATHOLOGICAL SUMMARY',
    'HISTOPATHOLOGIC SUMMARY': 'HISTOPATHOLOGICAL SUMMARY',
    'HISTOPATH0LOGICAL SUMMARY': 'HISTOPATHOLOGICAL SUMMARY',   # zero not O
    'HISTOPATHOLOGICAL SUMMARY/COMMENT': 'HISTOPATHOLOGICAL SUMMARY',
    'HISTOPATHOLOGICAL SUMMARY AND COMMENT': 'HISTOPATHOLOGICAL SUMMARY',
    'HISTOLOGICAL SUMMARY': 'HISTOPATHOLOGICAL SUMMARY',
    'HISTOPATHOLOGICAL DESCRIPTION': 'HISTOPATHOLOGICAL SUMMARY',
    'HISTOLOGIC DESCRIPTION': 'HISTOPATHOLOGICAL SUMMARY',
    'SPECIAL STAIN': 'SPECIAL STAINS',
    'COMMUNICATIONS': 'COMMENTS AND COMMUNICATIONS',
    'COMMENTS': 'COMMENT',
    'ADDEDNUM': 'ADDENDUM',
    'CLINICAL ABSTRAST': 'CLINICAL ABSTRACT',
    'CLINICAL ABSRACT': 'CLINICAL ABSTRACT',
    'LINICAL ABSTRACT': 'CLINICAL ABSTRACT',
    'GROSS DESCRIPTIONL': 'GROSS DESCRIPTION',
    'GOI?/ROSS DESCRIPTION': 'GROSS DESCRIPTION',
    'GROSS DESCRIPTIO': 'GROSS DESCRIPTION',
    'GROSS NECROPSY FINDING': 'GROSS DESCRIPTION',
    'FINAL COMMENTO': 'FINAL COMMENT',
    'COM4ERMENTS AND COMMUNICATIONS': 'COMMENTS AND COMMUNICATIONS',
    'COMMENTS AND CO/MMUNICATIONS': 'COMMENTS AND COMMUNICATIONS',
    'COMMENTS AND COMMUNICATION': 'COMMENTS AND COMMUNICATIONS',
    'ANCIL': 'ANCILLARY TESTS',
    'ANCILLARY': 'ANCILLARY TESTS',
    'ANCILLARY TEST': 'ANCILLARY TESTS',
    'ANCILLARY DIAGNOSTICS': 'ANCILLARY TESTS',
    'IMMUNOHISTOCHEMISTRY (IHC)': 'IMMUNOHISTOCHEMISTRY',
    'REFERENCE': 'REFERENCES',
    'ORIGINAL COMMENT': 'COMMENT',
    'HISTORY': 'CLINICAL HISTORY',
    'POLARIZING FILTERS (T3, T2)': 'ANCILLARY TESTS',
    # Malformed pipe markers — discard (content folds into previous section)
    '|B': '',
    '|U': '',
    'N/A': '',
    'NONE': '',
    ',': '',
    # Case-specific content that bled into a heading marker — fold into COMMENT
    '|. THE GREEN': 'COMMENT',
    '| AND G': 'COMMENT',
    'HYPERGLOBULINEMIA (4.6), EOSINOPHILIA (2620), BASOPHILS (728)': 'COMMENT',
    'TWO CHARS - LABELED "FINN" -': 'COMMENT',
    'CLINICAL CONSULT BY DR. OUTERBRIDGE 8/25/2020': 'COMMENT',
    'MALASSEZIA': 'COMMENT',
    'T1, LIVER, TWO SECTIONS': 'COMMENT',
    'GINGIVAL ENLARGEMENT PALATAL TO LMAXI1': 'COMMENT',
    'FUNGAL VS NEOPLASIA, VS OTHER': 'COMMENT',
    'FUNGAL (ASPERGILLUS) SINUSITIS': 'COMMENT',
}


def normalize_heading(raw):
    """Strip pipe markers, uppercase, strip whitespace, and apply known normalizations."""
    # Remove any |X| pipe markers that may have bled into the captured heading
    h = re.sub(r'\|[A-Za-z]\|', '', raw).strip().upper()
    # Strip common leading/trailing punctuation artifacts from OCR or formatting errors
    h = h.lstrip('/').rstrip(';.').strip()
    # Any ADDENDUM variant (e.g. "ADDENDUM (9-11-25)") → ADDENDUM
    if h.startswith('ADDENDUM'):
        return 'ADDENDUM'
    # Date-as-heading (e.g. " 5/30/2025: ordered GROSS DESCRIPTION") — map to GROSS DESCRIPTION
    if re.match(r'^\d', h):
        return 'GROSS DESCRIPTION'
    return HEADING_NORMALIZATIONS.get(h, h)


def extract_heading(cell):
    """
    If cell contains a |H| or |B| heading marker, extract the heading name and any inline content.
    Returns (heading, inline_content, text_after_marker).
    Returns (None, None, cell) if no heading found.
    """
    m = re.search(r'\|[HB]\|(.*?)\|\|', cell)
    if not m:
        return None, None, cell
    raw = m.group(1)
    # Split on ":" (with or without trailing space) to separate heading from inline content.
    # e.g. "ANCILLARY TESTS: NA", "ANCILLARY TESTS:NA", "HISTOPATHOLOGICAL SUMMARY:T1..."
    if ':' in raw:
        idx = raw.index(':')
        heading_candidate = raw[:idx]
        inline = raw[idx + 1:].strip()
        # If nothing before the colon (malformed), fall back to treating whole thing as heading
        heading = normalize_heading(heading_candidate) if heading_candidate.strip() else normalize_heading(raw)
    else:
        heading = normalize_heading(raw)
        inline = ''
    rest = cell[m.end():].strip()
    return heading, inline, rest


def clean_section_text(text):
    """
    Remove formatting markers from section text:
    - |U|label||: and |U|label: || → plain "label:"
    - |X| single-letter markers → removed
    - Blank lines and extra whitespace → collapsed
    """
    # Sub-headings: |U|label||: or |U|label: ||
    text = re.sub(
        r'\|U\|(.*?)(?:\|\|:?|:\s*\|\|)',
        lambda m: m.group(1).rstrip(': ') + ':',
        text
    )
    # Remove remaining pipe markers like |B|, |H|, etc.
    text = re.sub(r'\|[A-Za-z]\|', '', text)
    # Strip each line and drop empties
    lines = [line.strip() for line in text.splitlines()]
    lines = [l for l in lines if l]
    return ' '.join(lines)


def parse_diagnoses(diagnosis_cells):
    """
    Parse numbered diagnosis strings into (number, text) tuples.
    Each cell may contain one numbered diagnosis (e.g. "1. SKIN: LYMPHOMA").
    Returns a list of (diagnosis_number_str, diagnosis_text) tuples.
    """
    diagnoses = []
    for cell in diagnosis_cells:
        cell = cell.strip()
        if not cell:
            continue
        m = re.match(r'^(\d+)[.)]\s*(.*)', cell, re.DOTALL)
        if m:
            diagnoses.append((m.group(1), m.group(2).strip()))
        else:
            diagnoses.append(('', cell))
    return diagnoses


def parse_report_sections(text_cells):
    """
    Split report text cells into sections by |H|/|B| heading markers.
    Returns an OrderedDict of {heading: cleaned_content}.
    Headings are discovered dynamically.
    """
    sections = OrderedDict()
    current_heading = None
    current_lines = []

    for cell in text_cells:
        stripped = cell.strip()
        # Skip the placeholder "Text" that appears on the first row of each case
        if not stripped or stripped.lower() == 'text':
            continue

        heading, inline, rest = extract_heading(stripped)

        if heading:
            # Save the previous section before starting a new one
            if current_heading is not None:
                existing = sections.get(current_heading, '')
                addition = clean_section_text('\n'.join(current_lines))
                sections[current_heading] = (existing + ' ' + addition).strip() if existing else addition
            current_heading = heading
            current_lines = [inline] if inline else []
            if rest:
                current_lines.append(rest)
        else:
            current_lines.append(stripped)

    # Save the final section
    if current_heading is not None:
        existing = sections.get(current_heading, '')
        addition = clean_section_text('\n'.join(current_lines))
        sections[current_heading] = (existing + ' ' + addition).strip() if existing else addition

    return sections


def build_additional_info(sections):
    """
    Serialise all non-canonical sections into a single string.
    Each section is prefixed with its heading as a label: "HEADING: content"
    """
    parts = []
    for heading, text in sections.items():
        if heading not in CANONICAL_SET and heading and text.strip():
            parts.append(f'{heading}: {text.strip()}')
    return '  '.join(parts)


def _na(value):
    """Return empty string for 'NA' sentinel, otherwise the stripped value."""
    s = value.strip()
    return '' if s == 'NA' else s


def parse_cases(input_file):
    """Read the CSV and group rows into individual cases."""
    cases = []
    current_case = None

    with open(input_file, newline='', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        next(reader)  # skip header row

        for row in reader:
            # Ensure row has at least 9 columns
            while len(row) < 9:
                row.append('')
            dob = _na(row[0])
            sex = _na(row[1])
            species = _na(row[2])
            breed = _na(row[3])
            zipcode = _na(row[4])
            rfrr_zipcode = _na(row[5])
            dt = _na(row[6])
            diagnosis_cell = row[7]
            text_cell = row[8]

            if dt:
                # New case starts
                if current_case:
                    cases.append(current_case)
                current_case = {
                    'dob': dob,
                    'dt': dt,
                    'sex': sex,
                    'species': species,
                    'breed': breed,
                    'zipcode': zipcode,
                    'rfrr_zipcode': rfrr_zipcode,
                    'diagnosis_cells': [diagnosis_cell] if _na(diagnosis_cell) else [],
                    'text_cells': [text_cell],
                }
            else:
                if current_case is None:
                    continue
                if _na(diagnosis_cell):
                    current_case['diagnosis_cells'].append(diagnosis_cell)
                current_case['text_cells'].append(text_cell)

    if current_case:
        cases.append(current_case)

    return cases


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    input_files = sorted(glob.glob(os.path.join(DATA_DIR, 'input', '*.csv')))
    if not input_files:
        print(f"No CSV files found in {os.path.join(DATA_DIR, 'input')}")
        return

    all_raw_cases = []
    for path in input_files:
        file_cases = parse_cases(path)
        print(f"  {os.path.basename(path)}: {len(file_cases)} cases")
        all_raw_cases.extend(file_cases)

    print(f"Total: {len(all_raw_cases)} cases across {len(input_files)} file(s)")

    parsed_cases = []
    skipped = 0
    case_number = 1
    for case in all_raw_cases:
        sections = parse_report_sections(case['text_cells'])
        if not any(text.strip() for text in sections.values()):
            skipped += 1
            continue
        diagnoses = parse_diagnoses(case['diagnosis_cells'])
        parsed_cases.append({
            'case_id': f'CASE-{case_number:04d}',
            'case': case,
            'diagnoses': diagnoses,
            'sections': sections,
        })
        case_number += 1

    print(f"Skipped {skipped} cases with no report text")

    # --- demographics.csv ---
    with open(os.path.join(OUTPUT_DIR, 'demographics.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['case_id', 'DateOfBirth', 'DtOfRq', 'Sex', 'Species', 'Breed', 'Zipcode', 'RfrrVtrnZipcode'])
        for pc in parsed_cases:
            c = pc['case']
            writer.writerow([pc['case_id'], c['dob'], c['dt'], c['sex'], c['species'], c['breed'], c['zipcode'], c['rfrr_zipcode']])

    # --- diagnoses.csv ---
    with open(os.path.join(OUTPUT_DIR, 'diagnoses.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['case_id', 'diagnosis_number', 'diagnosis'])
        for pc in parsed_cases:
            for num, text in pc['diagnoses']:
                writer.writerow([pc['case_id'], num, text])

    # --- report.csv ---
    report_columns = CANONICAL_HEADINGS + ['ADDITIONAL INFORMATION']
    with open(os.path.join(OUTPUT_DIR, 'report.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['case_id'] + report_columns)
        for pc in parsed_cases:
            sections = pc['sections']
            row = [pc['case_id']]
            for h in CANONICAL_HEADINGS:
                row.append(sections.get(h, ''))
            row.append(build_additional_info(sections))
            writer.writerow(row)

    print(f"Output written to {OUTPUT_DIR}/")
    print(f"  demographics.csv       — {len(parsed_cases)} rows")
    print(f"  diagnoses.csv          — {sum(len(pc['diagnoses']) for pc in parsed_cases)} rows")
    print(f"  report.csv             — {len(parsed_cases)} rows, {len(report_columns)} columns")


if __name__ == '__main__':
    main()
