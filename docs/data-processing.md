# Data Processing

This document describes the raw CSV parsing step that feeds the ML pipeline. It is aligned
with the current ML documentation in `ml/documentation/` and with the parser implementation
in `database/scripts/parse_diagnostics.py`.

## Purpose

The source `* diagnostics.csv` exports are semi-structured. One case can span multiple rows:

- Demographics only appear on the first row of a case.
- Diagnosis entries may be numbered across multiple rows.
- Report text is stored in a single free-text column with embedded heading markers such as
  `|H|...||` and subheading markers such as `|U|...||:`.

The parser de-identifies each case, assigns a stable internal `case_id`, and produces the
two ML input files plus a demographics file for downstream database work.

## Source Format

The raw CSV has 6 columns:

- `DtOfRq`
- `Sex`
- `Species`
- `Breed`
- `Diagnoses`
- `Text`

The first populated `DtOfRq` starts a new case. Continuation rows for the same case leave
the demographics columns blank and append more diagnosis or report text.

## Output Files

The parser writes three files to `database/data/output/`:

- `demographics.csv`
- `diagnoses.csv`
- `report.csv`

### `demographics.csv`

One row per parsed case.

Columns:

- `case_id`
- `DtOfRq`
- `Sex`
- `Species`
- `Breed`

### `diagnoses.csv`

One row per diagnosis item. This is the input used by the ML annotation pipelines described in
`ml/documentation/label-annotation.md`.

Columns:

- `case_id`
- `diagnosis_number`
- `diagnosis`

### `report.csv`

One row per parsed case. This is the input used by the production PetBERT pipeline described in
`ml/documentation/petbert-pipeline.md`.

Columns:

- `case_id`
- One column per discovered report heading

Common headings include:

- `ADDENDUM`
- `FINAL COMMENT`
- `CLINICAL ABSTRACT`
- `GROSS DESCRIPTION`
- `HISTOPATHOLOGICAL SUMMARY`
- `ANCILLARY TESTS`

Additional headings are preserved dynamically when they appear in source data.

## Heading Normalization

The parser normalizes common heading variants to the canonical names used throughout the ML docs.
Examples:

- `FINAL COMMENTS` -> `FINAL COMMENT`
- `GROSS FINDINGS` -> `GROSS DESCRIPTION`
- `ANCILLARY TESTING` -> `ANCILLARY TESTS`
- `ADDITIONAL TESTS` -> `ANCILLARY TESTS`
- `CLINICAL ABS TRACT` -> `CLINICAL ABSTRACT`

`ADDENDUM` variants are collapsed to `ADDENDUM`, and malformed date-prefixed headings are mapped
to `GROSS DESCRIPTION`.

## Text Cleaning

For each report section, the parser:

- removes pipe-formatting markers such as `|H|`, `|U|`, and similar inline markup
- converts embedded subheadings into plain text labels
- removes blank lines
- trims repeated whitespace
- preserves section-level content in separate columns rather than flattening the whole report

## Case IDs and De-identification

Each parsed case receives a generated internal identifier:

- `CASE-0001`
- `CASE-0002`
- `CASE-0003`

This `case_id` is the join key between `diagnoses.csv` and `report.csv` for ML workflows.

## Relationship To ML Docs

This preprocessing step feeds the current ML pipeline as follows:

- `database/data/output/diagnoses.csv` -> `ml/data/diagnoses.csv`
- `database/data/output/report.csv` -> `ml/data/report.csv`

From there:

- `ml/data/diagnoses.csv` is consumed by the keyword and LLM annotation pipelines
- `ml/data/report.csv` is consumed by the production PetBERT pipeline

For the current ML architecture and commands, use `ml/documentation/README.md` as the canonical
reference.
