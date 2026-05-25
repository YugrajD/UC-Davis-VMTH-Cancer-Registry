# Development of a standardized Cancer Registry for dogs: illustrating the value using data from the UC Davis-VMTH

# Introduction/Background 

## (we do not recommend placing your abstract on your poster unless required) Provides context. Clearly states your hypothesis/aim/goal here or under another section title.  Explains why are you asking the research question.

Cancer is a leading cause of mortality in companion animals, yet the US currently lacks a centralized veterinary cancer registry, making cancer type prevalence and geospatial trend estimation difficult. This study aims to illustrate the value of such a registry. Institutions like the UC Davis Veterinary Medical Teaching Hospital (VMTH) possess extensive records that would aid in development of a registry, but said records often contain inconsistent diagnostic coding and fragmented storage across veterinary labs and clinics. Establishing a standardized registry would help resolve this issue, enabling population-level analysis and epidemiological research.

Vet-ICD-O-canine-1 is a recently published canine adaptation of the WHO ICD-O oncology taxonomy, providing 846 standardized terms grouped under 52 cancer groups with behavior codes (benign / uncertain / in situ / malignant). It supplies the controlled vocabulary that a canine registry needs in order to make diagnoses comparable across cases, institutions, and time.

The objective of this project is to develop a standardized cancer registry for dogs using deidentified, unstructured clinical data and an AI-driven NLP pipeline that maps free-text diagnoses to Vet-ICD-O-canine-1 codes. We will integrate the newly structured registry with a geospatial dashboard to enable spatial mapping and exploratory epidemiological analysis. Furthermore, we aim to rigorously test and validate this end-to-end framework to establish a scalable foundation that supports future multi-institutional expansion.

# Methods 

## Outlines your research strategy.  Shows how you carried out your research, materials and methods used.  

**Data.** Deidentified canine records from UC Davis VMTH (1990–2026): 119,895 dogs with demographic metadata (breed, age, sex, ZIP, diagnosis date); 58,208 had diagnosis-bearing reports. Cancer diagnoses appear only as free-text across three report sections — *History*, *Findings & Comments*, *Ancillary*.

**Text representation.** PetBERT (a veterinary-domain BERT) was adapted to Vet-ICD-O-canine-1 via **per-section contrastive fine-tuning**: each report section was paired with its label's textual description under an InfoNCE objective, aligning section semantics with code semantics. Each report is then a **2304-dim "concat-3" vector** — three section embeddings concatenated. This adaptation alone shifted pipeline G+S from 13.1% → 24.0% before any downstream retraining.

**Annotation supervision.** A **three-tier cascade** generated training labels: (1) exact term match, (2) fuzzy token overlap, (3) local LLM ensemble (Gemma-3-27B + Qwen3-30B). Negation masking and behavior-code awareness (benign / uncertain / in situ / malignant) blocked out-of-context term lifting; an ensemble verification pass demoted non-unanimous labels to *Uncertain*. The cascade labeled **26,791 / 58,208 cases (46.0%)** across 50 of 52 cancer groups.

**Four-stage classifier pipeline** (independently retrainable):
1. **CasePresence** — binary gate, val F1 = 0.94 (t = 0.85).
2. **Group** — 2304→512→25 multi-label MLP, macro F1 = 0.57; Stage-2 tail-gate suppresses low-confidence trailing predictions (K = 2, prob-gap = 0.08).
3. **Per-group LabelPresence** — 17 within-group disambiguators over `[report | label]` pairs, trained with 5 hard negatives per positive; per-LP thresholds calibrated post-hoc (13 / 17 at t ≥ 0.85).
4. **Keyword correction** — rule-based ICD-O behavior-code and subtype filter (not learned).

All classifiers trained with binary cross-entropy on the auto-annotated training split, with class-rebalancing (per-class loss weights capped at 50) to counteract long-tail group imbalance. Per-stage operating thresholds were tuned on a held-out calibration set, and a focal-loss alternative was evaluated and discarded after collapsing Group macro F1 from 0.57 → 0.23.

**Evaluation.** Case-wise split — train (46,652) / held-out test (11,661), stratified by label group with patient-level leakage prevention. Metrics reported on **common labels** (Vet-ICD-O-canine-1 terms with frequency > 20 in the test set; n = 33 labels, *Neoplasms, NOS* excluded from macro averaging as a taxonomically ambiguous catch-all): **exact-term recall** (predicted = expected), **group-level recall** (predicted in correct group), and macro precision / F1.

# Results

## Shares what was found out and shows the data.  

Among 58,208 annotated cases, the automated annotation cascade assigned at least one Vet-ICD-O-canine-1 label to **26,791 (46.0%)**, covering **50 of 52 (96.2%) groups** in the taxonomy. The five highest-volume groups — Adenomas/adenocarcinomas (15.6%), Blood vessel tumors (9.4%), Epithelial neoplasms NOS (9.3%), Lipomatous neoplasms (8.1%), and Mast cell neoplasms (8.0%) — account for over half of all labeled cases. Most cases concentrate within a single cancer group (38.8% one group, 7.2% multi-group), though 5.8% exhibit same-group multi-diagnosis collisions reflecting differential or comorbid lesions.

To assess clinically meaningful performance, we report results on common labels — Vet-ICD-O-canine-1 terms with more than 20 cases in the held-out test set (n = 33 labels; *Neoplasms, NOS* excluded from macro averaging as a non-specific catch-all). Across this set, the 4-stage pipeline reaches **77.2% exact-term recall** and **83.1% group-level recall** (macro precision 0.632, macro F1 0.688), meaning the predicted code matches the expected label outright in over three quarters of common-cancer cases, and falls within the correct cancer group in more than four out of five.

For the three most common labels:

| Label | Group | n cases | Exact recall | Group recall |
|---|---|---|---|---|
| Hemangiosarcoma, NOS | Blood vessel tumors | 195 | **90.8%** | 95.9% |
| Mast cell tumor, NOS | Mast cell neoplasms | 192 | **90.6%** | 91.7% |
| Lipoma, NOS | Lipomatous neoplasms | 174 | **60.9%** | 61.5% |

Hemangiosarcoma and mast cell tumor — two of the most clinically important canine cancers — are recovered with >90% exact-term accuracy. Lipoma is harder because its free-text descriptions overlap heavily with other lipomatous neoplasms (e.g., infiltrative or atypical variants), so most errors are within-group subtype confusions rather than wrong-group predictions.

Each stage was instrumented independently: the case-presence gate reaches F1 = 0.94 on cancer/non-cancer separation, the group classifier reaches macro F1 = 0.57 across 25 groups, and per-group label disambiguation benefits from calibrated per-LP thresholds and a Stage-2 tail-gate that suppresses low-confidence trailing predictions. Behavior-code keyword correction further preserves benign-vs-malignant distinctions that the embedding layer alone cannot reliably encode.

Residual error is concentrated in lexical heterogeneity across decades of records, differential diagnoses presented without pathology confirmation, and rare cancer groups for which training signal is sparse. Planned next steps include incorporation of pathology-confirmed cases to anchor noisy labels, expansion of the group taxonomy beyond the current 25-group set, and multi-institutional data ingestion to test generalization.

# Conclusions/Discussion/Next Steps

## Focuses on main takeaway points.  Shares what the significance of the research is and future directions.  

AI-assisted standardization of veterinary free-text diagnoses allows for scalable construction of a canine cancer registry. A staged classifier pipeline operating over domain-adapted PetBERT embeddings — with per-section contrastive alignment to the Vet-ICD-O-canine-1 taxonomy — recovers the exact Vet-ICD-O term in 77.2% of common-cancer cases (>20 cases per label, n = 33) at UC Davis VMTH, including >90% exact recall on hemangiosarcoma and mast cell tumor, two of the most clinically consequential canine cancers. The modular architecture (gate → group → per-group label → keyword correction) makes individual stages independently retrainable, supports future integration across institutions, and lays the groundwork for geospatial dashboards and broader comparative oncology research.

# References

## Cites sources and credits any figures used in your poster

# Acknowledgements 

thank individuals who contributed to your project and always acknowledge funding sources

Ruwini, Beatriz, Michael Kent