# Cancer Classification Systems Reference

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

The `ml/ICD-labels/labels.csv` file contains the Vet-ICD-O-Canine-1 taxonomy as used by
this project. It has 846 term rows organized across 44 cancer groups. Each row has:

| Column | Example | Description |
|--------|---------|-------------|
| `Vet-ICD-O-canine-1 code` | `8000/3` | Morphology code (cell type / behaviour) |
| `Group` | `Neoplasms, NOS` | Broader cancer category (44 groups) |
| `Term` | `Neoplasm, malignant` | Specific diagnostic label |
| `level` | `Preferred` | Preferred vs. synonym |
| `Topography` | — | Anatomical site (where specified) |

The pipeline embeds each term (as a short text string) through PetBERT and matches report
embeddings against them via cosine similarity or the trained PresenceClassifier.

The 44 groups in this project's taxonomy map to varying depths of Vet-ICD-O-Canine-1.
Only groups with ≥100 keyword-confirmed training cases (17 of 44) are usable by the
GroupClassifier; the binary PresenceClassifier can reach all 846 terms.

**GIVCS:** https://www.givcs.org/
