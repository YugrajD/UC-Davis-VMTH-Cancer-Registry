# Cancer Classification Systems Reference

Background on the coding systems behind this project's labels (ICD-11, ICD-O-3.2 and Vet-ICD-O-Canine-1) and how the project uses them.

Covers the three classification systems relevant to this project:
- **ICD-11** — what PetBERT-ICD was trained on
- **ICD-O-3.2** — the human oncology coding standard this project's taxonomy derives from
- **Vet-ICD-O-Canine-1** — the canine-specific taxonomy used as prediction targets

---

## ICD-11

### What it is

The International Classification of Diseases, 11th Revision (ICD-11) is the global standard
for recording and comparing health information. Maintained by the World Health Organization
(WHO). Officially came into effect 1 January 2022.

### Structure

| Detail | Value |
|--------|-------|
| Chapters | 28 |
| Unique diagnostic codes (MMS) | ~17,000 |
| Full foundation layer entities | ~85,000 |
| Code format | Alphanumeric stem (e.g. 2A00) + optional extension codes prefixed "X" |

ICD-11 is nominally a single-axis hierarchy but supports **post-coordination**: codes can be
combined into clusters to capture site + histology + stage + laterality together, giving it
effective multi-axial capacity.

### Chapter 2 — Neoplasms

The chapter relevant to this project and to PetBERT-ICD.

| Detail | Value |
|--------|-------|
| Code range | 2A00–2F9Z |
| Leaf codes | 1,037 (36.6% expansion over ICD-10) |
| Behaviour encoding | Embedded in stem codes or extension codes |

Chapter 2 is one of the 20 ICD-11 chapters that PetBERT-ICD classifies into. It is a broad
category covering all neoplasms — not specific cancer types or histologies.

### Relationship to this project

PetBERT-ICD predicts ICD-11 chapters. This project predicts Vet-ICD-O-Canine-1 terms.
These label spaces are **incompatible** — ICD-11 chapter assignment cannot be converted to
Vet-ICD-O term assignment.

**Official reference:** https://icd.who.int/en/

---

## ICD-O (International Classification of Diseases for Oncology)

### What it is

A specialty classification designed for cancer registries and pathology departments.
Published jointly by WHO and IARC (International Agency for Research on Cancer).
Unlike ICD-11, ICD-O is **dual-axis**: every tumor receives two independent codes.

### Two axes

| Axis | Code format | What it describes |
|------|------------|-------------------|
| Topography | C00–C80 (from ICD-10 Chapter II) | Anatomical site of origin |
| Morphology | M-XXXX/B | Cell type (4 digits) + behaviour (1 digit after slash) |

**Behaviour codes (the slash digit):**

| Code | Meaning |
|------|---------|
| /0 | Benign |
| /1 | Uncertain whether benign or malignant |
| /2 | In situ |
| /3 | Malignant, primary |
| /6 | Malignant, metastatic |
| /9 | Malignant, uncertain whether primary or metastatic |

### Current version

ICD-O-3.2 was published by IARC/WHO in April 2019 and is recommended for all cases
diagnosed on or after 1 January 2021. Changes from ICD-O-3 affected only morphology codes;
topography codes are unchanged.

**SEER coding materials:** https://seer.cancer.gov/icd-o-3/
**WHO ICD-O page:** https://www.who.int/standards/classifications/other-classifications/international-classification-of-diseases-for-oncology

### ICD-O vs ICD-11

| | ICD-O | ICD-11 |
|--|-------|--------|
| Purpose | Cancer registry / pathology | General clinical / mortality reporting |
| Axes | Two (topography + morphology) | One (hierarchical, with post-coordination) |
| Specificity | High — specific histologies + site combinations | Lower — chapter/block level for neoplasms |
| Used by | Cancer registries (SEER, IARC, etc.) | Clinicians, hospitals, mortality statistics |

---

## Vet-ICD-O-Canine-1

### What it is

A peer-reviewed veterinary adaptation of ICD-O-3.2 for canine neoplasms. It is a published
standard, not a project-internal taxonomy.

**Citation:**
Pinello K, Baldassarre V, Steiger K, Paciello O, Pires I, Laufer-Amorim R, Oevermann A,
Niza-Ribeiro J, Aresu L, Rous B, Znaor A, Cree IA, Guscetti F, Palmieri C, Zaidan Dagli ML.
"Vet-ICD-O-Canine-1, a System for Coding Canine Neoplasms Based on the Human ICD-O-3.2."
*Cancers (Basel).* 2022 Mar 16;14(6):1529.
DOI: 10.3390/cancers14061529 · PMID: 35326681 · PMC: PMC8946502

### Who developed it

Developed by a coding subgroup within the **Global Initiative for Veterinary Cancer
Surveillance (GIVCS)**, in collaboration with IARC. The working group comprised 9 veterinary
pathologists and 2 veterinary epidemiologists from 6 countries (Portugal, Brazil, Australia,
Italy, Germany, Switzerland) plus IARC staff co-authors.

### Structure

| Detail | Value |
|--------|-------|
| Topography codes | 335 |
| Morphology codes | 534 |
| Chapters | 12 (organ-system based) |
| Axes | Dual (topography + morphology/behaviour), same as ICD-O-3.2 |
| Behaviour codes | Same slash notation as ICD-O-3.2 (see above) |

Vet-ICD-O-Canine-1 is explicitly designed to be **compatible with ICD-O-3.2**, enabling
comparative oncology studies and cross-referencing with human cancer registries under a
One Health framework. Canine-specific additions and modifications are made where human
equivalents do not exist.

### How this project uses it

`ml/taxonomy/labels.csv` (`config.LABELS_CSV`) holds the Vet-ICD-O-Canine-1 taxonomy as this project
uses it: 845 terms in 52 groups, covering 534 distinct morphology codes (counted with
`taxonomy.load_labels_taxonomy`, which drops duplicate rows). The file has a title row above its
header row. Each row has:

| Column | Example | Description |
|---|---|---|
| `Vet-ICD-O-canine-1 code` | `8000/3` | Morphology code (cell type / behaviour) |
| `Group` | `Neoplasms, NOS` | Broader cancer category (52 groups) |
| `Term` | `Neoplasm, malignant` | Specific diagnostic label |
| `level` | `Preferred` | Preferred vs. synonym |
| `Topography` | none | Anatomical site (where specified) |

Several terms can share one code (186 of the 534 codes map to more than one term, always within the
same group), so results and reviews are recorded as taxonomy terms, not bare codes.

Production predicts these terms in four stages plus a post-step: a case-presence gate decides whether
a report describes a cancer, a group classifier (with a tail gate) picks the group or groups, a
per-group label-presence head picks the terms inside each group, and a keyword correction filters
those terms; a lipoma rescue runs last. Groups with too few training cases share one `Uncommon` head, so
every term stays reachable. The design and recipe are in
[report-mapping.md](../concepts/report-mapping.md); the strategy for labels and review is in
[icd-mapping-strategy.md](../concepts/icd-mapping-strategy.md).

**GIVCS:** https://www.givcs.org/

_Last verified against code: 2026-09-29_
