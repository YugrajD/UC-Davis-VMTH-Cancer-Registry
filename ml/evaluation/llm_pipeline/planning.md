# Evaluation - LLM Pipeline

## Goal
The keyword pipeline has a number of misses and inaccuracies. This pipeline uses LLMs to produce a more complete evaluation table by catching what keyword matching missed or was uncertain about.

## Concept
This is an **independent pipeline** that runs on all diagnosis rows directly. It uses keyword matching as its first step, then escalates to LLM only for cases that keyword matching cannot resolve.

LLM is only invoked as a last resort for ambiguous or unmappable diagnoses.

## Terminology
- **Report Diagnosis** — the diagnosis string(s) for each case in diagnoses.csv
- **ICD Term** — the canonical or synonym term found in labels.csv
- **ICD Group** — the group/category term found in labels.csv

## Pipeline

### Pre-pass — Normalization & Abbreviation Expansion (no LLM)
Normalize text and expand known clinical abbreviations before any matching.
Reuse `_normalize()` logic from the keyword pipeline, with abbreviation expansion and
synonym substitutions added on top. Normalization steps (in order):
1. Lowercase
2. Collapse hyphens, underscores, slashes to spaces
3. Strip commas, parentheses, semicolons, colons (replace with spaces)
4. Collapse repeated whitespace
5. Apply synonym substitutions (see Abbreviation & Synonym Table below)
6. Expand clinical abbreviations (see Abbreviation & Synonym Table below)

### Tier 1 — Exact Match (no LLM)
For each ICD Term, generate candidate strings and match against the normalized diagnosis:
- **Full term** — normalized ICD Term (including synonyms and related terms from labels.csv)
- **Core term** — qualifier words stripped from the end: `NOS`, `NEC`, `malignant`, `benign`,
  `conventional`, `well differentiated`, `spindle cell`, `atypical`, and other descriptor words
- **Permutations** — for 2–3 word candidates, try all word-order permutations
  (e.g. `"neoplasm malignant"` also matches `"malignant neoplasm"`)
- **Plurals** — each pattern allows an optional trailing `s` (e.g. `carcinoma` matches `carcinomas`)
- Longest match wins; FAB M0–M7 codes matched by exact string (no signal suffix needed)
- Label: **Exact**

### Tier 2 — Fuzzy Match (no LLM)
- Try partial / substring matches of the ICD Term core against the Report Diagnosis
- If similarity score exceeds threshold → **likely match**
- Assign a confidence score; flag for review if below a second threshold
- Key target: descriptor-heavy strings (e.g. "moderately differentiated hepatocellular carcinoma", "metastatic osteosarcoma, high-grade")
- Label: **Fuzzy**

### Tier 3 — Signal Fallback + LLM
Triggered when Tiers 1 and 2 both fail but the Report Diagnosis contains a signal term
(see Signal Terms below), including named outlier terms.

Steps:
1. Identify a candidate **ICD Group** using exact or fuzzy match on group-level terms
2. **If an ICD Group is found:** pass the Report Diagnosis and candidate group(s) to the LLM; LLM selects the best-matching ICD Term within that group, or returns "no match"
3. **If no ICD Group is identified:** extract all `-oma` and `-emia` words from the diagnosis (longest first) and look them up in the ICD Term index; pass those candidates to the LLM for disambiguation
4. **LLM prompt must check for negation** — phrases like "no evidence of lymphoma" or "rule out carcinoma" should return "no match" even though a signal term is present
5. Label: **LLM**

### Tier 4 — No Match
- Report Diagnosis did not match any ICD Term and contains no signal terms
- Marked as **No Match** — may be benign, non-neoplastic, or a data quality issue


---

## Signal Terms (Tier 3 Trigger)

These patterns in a Report Diagnosis suggest a neoplastic diagnosis worth escalating to the LLM.
Named outlier terms are also included here so they trigger Tier 3 rather than falling silently to Tier 4.

### Suffix Patterns (~96% of catalog)
| Pattern | Notes |
|---|---|
| `-oma` (any form) | Covers ~79.5% of catalog — carcinoma, sarcoma, lymphoma, blastoma, adenoma, melanoma, etc. |
| `tumor` / `tumour` | ~11.5% of catalog — mast cell tumor, GIST, granulosa cell tumor, etc. |
| `leukemia` / `leukaemia` | ~4.6% of catalog |
| `neoplasm` | ~1% of catalog |
| `cancer` / `malignancy` / `malignant` | Generic terms not tied to a specific ICD suffix — escalate to LLM |
| `metastasis` / `metastatic` | Known keyword pipeline miss — "metastasis" ≠ "metastatic neoplasm" in the keyword index |

### Named Outliers (~2.7% of catalog — explicit keyword triggers)
These have no standard suffix and must be matched by keyword to reach Tier 3:

| Trigger Keyword | Maps To | Notes |
|---|---|---|
| `carcinoid` | Neuroendocrine carcinoma | |
| `mycosis fungoides` | Cutaneous epitheliotropic lymphoma | |
| `polycythemia vera` | Myeloproliferative neoplasm | |
| `paget disease` | Paget carcinoma | |
| `mastocytosis` | Systemic/visceral mastocytosis | Covers both systemic and visceral variants |
| `ganglioneuromatosis` | Diffuse ganglioneuroma-like proliferation | |
| `gliomatosis` | Diffuse astrocytoma | Obsolete synonym |
| `lipomatosis` | Diffuse lipoma proliferation | |
| `refractory anemia` | Myelodysplastic syndrome | Covers NOS and "with excess blasts" variants |
| `acanthomatous epulis` | Acanthomatous ameloblastoma | |
| `fibromatosis` | Multilobular tumor of bone | Cartilage analogue synonym |
| `PNET` | Primitive Neuroectodermal Tumor | Also matches CPNET |

### FAB Codes (handled in Tier 1 only)
FAB M0–M7 contain no signal terms and are matched exclusively by exact string in Tier 1.
They do not escalate to Tier 3.

---

## Abbreviation & Synonym Table (Pre-pass)

### Synonym Substitutions (applied during normalization)
These normalize free-text variants that don't appear verbatim in labels.csv:

| Input | Normalized To |
|---|---|
| `neoplasia` | `neoplasm` |
| `plasma cell tumor` | `plasmacytoma` |
| `metastasis` | `metastatic neoplasm` |

### Clinical Abbreviation Expansions (applied after normalization)
| Abbreviation | Expanded Term |
|---|---|
| MCT | mast cell tumor |
| SCC | squamous cell carcinoma |
| HCC | hepatocellular carcinoma |
| OSA | osteosarcoma |
| HSA | hemangiosarcoma |
| TVT | transmissible venereal tumor |
| DLBCL | diffuse large B-cell lymphoma |
| GIST | gastrointestinal stromal tumor |
| PNET | primitive neuroectodermal tumor |
| CPNET | central primitive neuroectodermal tumor |
