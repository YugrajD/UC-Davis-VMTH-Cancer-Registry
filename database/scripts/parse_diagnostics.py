#!/usr/bin/env python3
"""
Parse and de-identify 2024-2025 diagnostics.csv into three output files:
  - demographics.csv
  - diagnoses.csv
  - reportText.csv
"""

import csv
import re
import os
from collections import OrderedDict

INPUT_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', '2024-2025 diagnostics.csv')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'output')

# Map known typos/variants to canonical heading names
HEADING_NORMALIZATIONS = {
    'CLINICAL ABS TRACT': 'CLINICAL ABSTRACT',
    'COM/MENTS AND COMMUNICATIONS': 'COMMENTS AND COMMUNICATIONS',
    'FINAL COMMENTS': 'FINAL COMMENT',
    'GROSS FINDINGS': 'GROSS DESCRIPTION',
    'GROSS NECROPSY FINDINGS': 'GROSS DESCRIPTION',
    'ANCILLARY TESTING': 'ANCILLARY TESTS',
    'ADDITIONAL TESTS': 'ANCILLARY TESTS',
    'FINAL COPLOW DIAGNOSES>': 'FINAL COPLOW DIAGNOSES',
}


def normalize_heading(raw):
    """Strip pipe markers, uppercase, strip whitespace, and apply known normalizations."""
    # Remove any |X| pipe markers that may have bled into the captured heading
    h = re.sub(r'\|[A-Za-z]\|', '', raw).strip().upper()
    # Any ADDENDUM variant (e.g. "ADDENDUM (9-11-25)") → ADDENDUM
    if h.startswith('ADDENDUM'):
        return 'ADDENDUM'
    # Date-as-heading (e.g. " 5/30/2025: ordered GROSS DESCRIPTION") — map to GROSS DESCRIPTION
    if re.match(r'^\d', h):
        return 'GROSS DESCRIPTION'
    return HEADING_NORMALIZATIONS.get(h, h)


def extract_heading(cell):
    """
    If cell contains |H|...:||, extract the heading name and any inline content.
    Returns (heading, inline_content, text_after_marker).
    Returns (None, None, cell) if no heading found.
    """
    m = re.search(r'\|H\|(.*?)\|\|', cell)
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
    Split report text cells into sections by |H| heading markers.
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


def parse_cases(input_file):
    """Read the CSV and group rows into individual cases."""
    cases = []
    current_case = None

    with open(input_file, newline='', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        next(reader)  # skip header row

        for row in reader:
            # Ensure row has at least 6 columns
            while len(row) < 6:
                row.append('')
            dt = row[0].strip()
            sex = row[1].strip()
            species = row[2].strip()
            breed = row[3].strip()
            diagnosis_cell = row[4]
            text_cell = row[5]

            if dt:
                # New case starts
                if current_case:
                    cases.append(current_case)
                current_case = {
                    'dt': dt,
                    'sex': sex,
                    'species': species,
                    'breed': breed,
                    'diagnosis_cells': [diagnosis_cell] if diagnosis_cell.strip() else [],
                    'text_cells': [text_cell],
                }
            else:
                if current_case is None:
                    continue
                if diagnosis_cell.strip():
                    current_case['diagnosis_cells'].append(diagnosis_cell)
                current_case['text_cells'].append(text_cell)

    if current_case:
        cases.append(current_case)

    return cases


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cases = parse_cases(INPUT_FILE)
    print(f"Found {len(cases)} cases")

    # Parse each case and collect all heading names (preserving first-seen order)
    all_headings = []
    seen_headings = set()
    parsed_cases = []

    for i, case in enumerate(cases):
        case_id = f'CASE-{i + 1:04d}'
        diagnoses = parse_diagnoses(case['diagnosis_cells'])
        sections = parse_report_sections(case['text_cells'])
        parsed_cases.append({
            'case_id': case_id,
            'case': case,
            'diagnoses': diagnoses,
            'sections': sections,
        })
        for h in sections:
            if h not in seen_headings:
                all_headings.append(h)
                seen_headings.add(h)

    print(f"Headings discovered: {all_headings}")

    # --- demographics.csv ---
    with open(os.path.join(OUTPUT_DIR, 'demographics.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['case_id', 'DtOfRq', 'Sex', 'Species', 'Breed'])
        for pc in parsed_cases:
            c = pc['case']
            writer.writerow([pc['case_id'], c['dt'], c['sex'], c['species'], c['breed']])

    # --- diagnoses.csv ---
    with open(os.path.join(OUTPUT_DIR, 'diagnoses.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['case_id', 'diagnosis_number', 'diagnosis'])
        for pc in parsed_cases:
            for num, text in pc['diagnoses']:
                writer.writerow([pc['case_id'], num, text])

    # --- reportText.csv ---
    with open(os.path.join(OUTPUT_DIR, 'report.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['case_id'] + all_headings)
        for pc in parsed_cases:
            row = [pc['case_id']] + [pc['sections'].get(h, '') for h in all_headings]
            writer.writerow(row)

    print(f"Output written to {OUTPUT_DIR}/")
    print(f"  demographics.csv  — {len(parsed_cases)} rows")
    print(f"  diagnoses.csv     — {sum(len(pc['diagnoses']) for pc in parsed_cases)} rows")
    print(f"  reportText.csv    — {len(parsed_cases)} rows, {len(all_headings)} heading columns")


if __name__ == '__main__':
    main()
