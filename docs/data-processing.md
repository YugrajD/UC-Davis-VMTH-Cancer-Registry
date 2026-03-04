# Data De-identifier & Parser

This process is designed to parse and de-identify the source data "2024-2025 diagnosis.csv"

## Parser
The source file has 6 columns:
- DtOfRq — Date of request
- Sex — Patient sex (e.g. FS = female spayed)
- Species — e.g. K9 (canine)
- Breed
- Diagnoses (labels) — Numbered diagnosis list
- Text (pathology report) — Full pathology report text
The data is structured so that a single case spans many rows — the date/sex/species/breed only appear on the first row of each case, with the remaining rows continuing the pathology report text in the last column. 

The first 180 rows cover just 2 cases:
- Jan 8, 2025 — FS K9 Vizsla: Skin pinnae angiomatosis / proliferative thrombovascular necrosis
- Jan 9, 2025 — FS K9 Mix: Multiple skin masses (infundibular acanthoma, sebaceous hyperplasia, follicular cyst, sebaceous adenoma)

## De-identification
Each case should be assigned with its own case_id 
The Case ID should hide the personal info  

## Output
The parser should output 3 files;
- demographics.csv 
- diagnoses.csv
- reportText.csv

### demographics.csv
This file should contain the patient info and demographics:
- Case ID
- DtOfRq — Date of request
- Sex — Patient sex (e.g. FS = female spayed)
- Species — e.g. K9 (canine)
- Breed

### diagnoses.csv
This file should contain the diagnosis labels, for the purpose of verification and training for PetBERT model
- Case ID
- Diagnoses (labels) — Numbered diagnosis list

### reportText.csv
This file should contain the pathology report text. Within the text, there're headings (|H|xxx:||) and sub-headings (|U|xxx||:)
Instead of having the entire text in one column, the plan is to separate them by headings
Get rid of all empty lines, extra spaces, and formatting stuff
- Case ID
- ADDENDUM
- FINAL COMMENT
- CLINICAL ABSTRACT
- GROSS DESCRIPTION
- HISTOPATHOLOGICAL SUMMARY
- ANCILLARY TESTS
- (Any other heading that's missing)



