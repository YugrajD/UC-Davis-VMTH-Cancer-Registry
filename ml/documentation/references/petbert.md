# PetBERT Reference

## Overview

PetBERT is a large language model pre-trained on veterinary electronic health records (EHRs)
from the UK. It serves as the embedding backbone for this project's production pipeline,
generating 768-dimensional vector representations of pathology report text.

---

## Model Family

| Model | HuggingFace ID | Description |
|-------|---------------|-------------|
| PetBERT | `SAVSNET/PetBERT` | Domain-adapted base model |
| PetBERT-ICD | `SAVSNET/PetBERT_ICD` | Fine-tuned ICD-11 multi-label classifier |
| PetBERT-mortality | `SAVSNET/PetBERT_mortality` | Mortality-prediction variant |

**License:** openrail (permissive with responsible-use guidelines)
**GitHub:** https://github.com/SAVSNET/PetBERT
**Organisation:** SAVSNET (Small Animal Veterinary Surveillance Network), University of Liverpool

---

## PetBERT

### Training

Built on top of `bert-base-uncased` (110M parameters, 12 encoder layers, 768-dim hidden states).
Domain-adaptive pre-training was performed using masked language modelling (MLM) and next
sentence prediction (NSP) on a corpus of 5.1 million UK first-opinion veterinary EHRs
containing over 500 million words.

| Detail | Value |
|--------|-------|
| Base model | `bert-base-uncased` |
| Pre-training tasks | MLM + NSP |
| Corpus | 5.1M UK veterinary EHRs, 500M+ words |
| Compute | Single NVIDIA A100, ~450 GPU-hours (~30 kg CO₂) |
| Output dimension | 768-dim vectors |

### Why it matters for this project

Standard BERT was pre-trained on Wikipedia and BookCorpus. PetBERT was further pre-trained
on veterinary clinical language, giving tokens like "carcinoma", "lymph node", "excision",
"grade II", and "IHC" richer, domain-appropriate representations. This improves the quality
of embedding-space similarity comparisons between report text and taxonomy labels.

### How it is used here

Each report column (HISTOPATHOLOGICAL SUMMARY, FINAL COMMENT, ANCILLARY TESTS) is passed
through PetBERT's transformer body (not the MLM head). Tokens are mean-pooled over
non-padding positions to produce one 768-dim vector per column per case. Taxonomy labels
are embedded the same way. The embedding model runs once and results are cached in
`ml/data/embedding_cache.npz`.

---

## PetBERT-ICD

### What it is

PetBERT fine-tuned end-to-end as a **multi-label ICD-11 classifier** predicting 20 ICD-11
disease chapters. The classifier head is a linear layer on the `[CLS]` token:
`Linear(768 → 20)` with sigmoid activations and BCEWithLogitsLoss.

The system is architecturally an ensemble: 20 separate binary (Case/Control) classifiers
are trained first, then a combined multi-label model (`AutoModelForSequenceClassification`,
`problem_type="multi_label_classification"`) is fine-tuned end-to-end.

### Training details

| Detail | Value |
|--------|-------|
| Starting weights | `SAVSNET/PetBERT` |
| Fine-tuning scope | **Full model, end-to-end** (no frozen layers) |
| Output classes | 20 ICD-11 disease chapters |
| Loss | BCEWithLogitsLoss with per-class inverse-frequency weights |
| Optimizer | AdamW, lr=5e-5 |
| Epochs | Up to 10, early stopping (patience=3) on F1 |
| Reported performance | F1 > 83% across 20 disease codings |

Because fine-tuning was end-to-end with no frozen layers, the transformer backbone
weights in PetBERT-ICD have drifted from base PetBERT — particularly in the upper
encoder layers which specialise toward ICD-11 chapter-level distinctions.

### ICD-11 chapters predicted

PetBERT-ICD predicts 20 broad disease categories. Chapter 2 (Neoplasms, codes 2A00–2F9Z)
is one of them. These are chapter-level labels, not specific cancer types.

### Why PetBERT-ICD is not used here

#### The classifier head is incompatible

The output label space (20 ICD-11 chapters) is completely incompatible with the
Vet-ICD-O-canine-1 taxonomy (~857 terms across 44 groups). There is no mapping between
the two systems that would allow the head to be reused or adapted.

#### A two-stage ICD-11 → Vet-ICD-O approach does not work

An appealing idea is to use PetBERT-ICD as a first stage to narrow the prediction space,
then use a second classifier to map from ICD-11 chapters to Vet-ICD-O groups:

```
Stage 1: PetBERT-ICD  →  one of 20 ICD-11 chapters
Stage 2: second classifier  →  one of 44 Vet-ICD-O groups
```

This would only be valid if the 44 Vet-ICD-O groups were distributed across multiple
ICD-11 chapters. They are not. Every group in the Vet-ICD-O-canine-1 taxonomy
(Adenomas, Mast cell neoplasms, Gliomas, Lymphoid leukemias, Blood vessel tumors, etc.)
is a type of neoplasm — they all fall under **ICD-11 Chapter 2** (Neoplasms, 2A00–2F9Z).

```
44 Vet-ICD-O groups  ⊂  ICD-11 Chapter 2  ⊂  ICD-11 (20 chapters)
```

PetBERT-ICD's output for any cancer case would be "Chapter 2: Neoplasms". This is a
binary cancer/non-cancer signal that provides no discrimination between the 44 groups —
which is the hard part of the problem.

#### The backbone is also not ideal

PetBERT-ICD's upper encoder layers have been steered toward coarse chapter-level
distinctions during end-to-end fine-tuning. This compresses exactly the fine-grained
terminology differences (specific cell types, grades, subtypes) that this project
depends on. Base PetBERT (MLM pre-training only, no downstream task) is the better
embedding source for this use case.

---

## Papers

| Paper | DOI | Notes |
|-------|-----|-------|
| Farrell et al. 2023 — PetBERT-ICD for disease surveillance | 10.1038/s41598-023-44047-8 | Primary PetBERT paper |
| Companion/related SAVSNET paper | 10.1038/s41598-023-45155-7 | Same project, different framing |

**Citation:**
Farrell S, Appleton C, Noble P-JM, Al Moubayed N. "PetBERT: automated ICD-11 syndromic
disease coding for outbreak detection in first opinion veterinary electronic health records."
*Scientific Reports* 13, 18015 (2023).
