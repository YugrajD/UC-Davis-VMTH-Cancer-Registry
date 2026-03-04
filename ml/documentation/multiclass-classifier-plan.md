# Multi-Class Group Classifier — Design Plan

## Problem Statement

The current binary `PresenceClassifier` MLP scores each `(report_embedding, label_embedding)` pair
independently and selects the winner by argmax. Because groups compete implicitly rather than
explicitly, the classifier cannot redirect a wrong-group cosine match — it can only accept or
reject individual pairs. This produces a hard **~33% completely-off (CO) floor** and a **~33%
Good+Slight ceiling** (Phase 11, 5,788 confirmed cases), regardless of how long training continues.

The goal is to replace the binary pair-wise classifier with a **multi-class group classifier**
that makes a single global decision per report: which cancer group(s) does this report belong to,
or is it non-cancer (Uncategorized)?

---

## Current Architecture vs. Proposed Architecture

### Current (binary PresenceClassifier)

```
report text
    → PetBERT (frozen) → mean_embedding (768-dim)
    → for each of ~857 labels:
        concat(mean_embedding, label_embedding) → binary MLP → present/absent score
    → argmax across all label scores → predicted label
```

**Problem:** Labels compete implicitly through argmax after independent binary decisions.
The classifier has no information about competing labels when it scores any single pair.
Result: ~33% CO floor, ~33% Good+Slight ceiling (Phase 11).

### Proposed (multi-class GroupClassifier)

```
report text
    → PetBERT (frozen) → mean_embedding (768-dim)
    → GroupClassifier MLP → score per group (45 outputs, sigmoid)
    → threshold → predicted group(s)
    → for each predicted group:
        cosine similarity against only terms within that group → best term + ICD code
```

**Why this works:** The group classifier sees the full output space simultaneously.
Groups compete explicitly during training via multi-label binary cross-entropy.
CO drops because a wrong-group assignment is directly penalized in the loss function.
Term selection within the predicted group is a much easier sub-problem (fewer candidates,
semantically tighter cluster).

---

## Training Data

The keyword pipeline (training only) provides ground-truth labels via
`ml/output/diagnoses/keyword_predictions.csv`.

| Split | Cases | Label |
|-------|-------|-------|
| Cancer cases (keyword-matched) | ~5,788 unique cases (44 groups) | Multi-hot encoding over matched_group values |
| Non-cancer cases (no keyword match) | ~6,832 unique cases | Uncategorized (all zeros except Uncategorized bit) |
| **Total** | **~12,620 cases** | |

**Ground-truth assumption:** Cases not matched by the keyword scan are non-cancer.
This is valid for a general veterinary clinic population.
As keyword coverage improves, training data quality improves automatically — no
architectural changes needed.

### Multi-hot encoding

A case with two cancer diagnoses (e.g. mast cell tumor + squamous cell carcinoma) gets
a target vector with both group bits set:

```python
target = torch.zeros(num_groups + 1)  # +1 for Uncategorized
for group in matched_groups_for_case:
    target[group_to_index[group]] = 1.0
# Non-cancer case:
target[uncategorized_index] = 1.0
```

### Class distribution

5,788 confirmed cancer cases across 44 groups (distribution varies by group; re-train
`train_group_classifier.py` whenever keyword coverage improves). Class imbalance is
handled via **per-class BCE loss weights** (inverse frequency weighting).

---

## Model Architecture

```
GroupClassifier(
    input_dim  = 768,        # PetBERT mean_embedding
    hidden_dim = 256,
    num_classes = 45,        # 44 cancer groups + 1 Uncategorized
    dropout = 0.3
)

forward(x):
    x = ReLU(Linear(768 → 256))
    x = Dropout(0.3)
    x = Sigmoid(Linear(256 → 45))   # independent probability per class (multi-label)
    return x
```

- **Sigmoid** (not softmax): a report can belong to multiple groups simultaneously.
- **Loss:** binary cross-entropy per class, with inverse-frequency class weights to
  compensate for rare groups and the large Uncategorized majority.
- **Inputs:** cached `mean_embeddings` from `ml/data/embedding_cache.npz` — no PetBERT
  inference needed during training.

### Why not fine-tune PetBERT?

PetBERT fine-tuning requires significantly more compute, risks catastrophic forgetting,
and is unnecessary at current data volumes. The GroupClassifier trains on frozen 768-dim
embeddings already in the cache — training runs in seconds on MPS.

As keyword coverage and case counts grow, fine-tuning becomes more viable and can be
explored in a future phase.

---

## Inference Flow

Replaces `PresenceClassifier.score_matrix()` in the PetBERT pipeline:

```
1. Load cached mean_embedding for the report (or compute via PetBERT if not cached)
2. Forward pass through GroupClassifier → 45 group probabilities
3. Apply threshold (e.g. 0.3) → predicted group(s)
   - If no group exceeds threshold → predict Uncategorized
4. For each predicted group:
   a. Filter taxonomy labels to only terms within that group
   b. Cosine similarity between mean_embedding and those term embeddings
   c. Select top-1 term → predicted_term, predicted_code
5. Output: up to k predictions, one per predicted group above threshold
```

This preserves the existing output format (term + group + code per prediction) while
eliminating the CO problem at the group level.

---

## Evaluation

The existing evaluation framework (`training/binary/evaluate.py`) applies directly.

### Results by data volume

| Data | Metric | Binary (Phase 11) | GroupClassifier @ 0.3 | GroupClassifier @ 0.8 |
|------|--------|-------------------|-----------------------|-----------------------|
| 1,273 cases (Phase 9) | Good+Slight | 20.4% | 14.3% | 23.4% |
| 1,273 cases (Phase 9) | CO% | 42.7% | 55.9% | 50.7% |
| 1,273 cases (Phase 9) | FP% | ~30% | 28.0% | 8.4% |
| 1,273 cases (Phase 9) | FN% | 4% | 1.8% | 17.6% |
| **5,788 cases (Phase 11)** | **Good+Slight** | **33.1%** | 13.9% | 21.9% |
| **5,788 cases (Phase 11)** | **CO%** | **31.8%** | 57.5% | 54.5% |
| **5,788 cases (Phase 11)** | **FN%** | **1.3%** | — | 15.6% |

**Findings at 5,788 cases:**
- Binary PresenceClassifier is the clear winner at current data volumes
- GroupClassifier still overfits (val loss >> train loss) despite 5,788 cases
- The MLP memorises rather than generalises across 44 groups with ~132 examples per group on average
- Re-train whenever keyword coverage improves — no architecture changes needed

### Expected trajectory as keyword coverage grows

| Keyword-confirmed cases | Expected Good+Slight | Notes |
|------------------------|---------------------|-------|
| ~1,273 | ~20–23% | Comparable to Phase 9 binary |
| ~5,788 (current) | GroupClassifier still overfits | Binary wins at 33.1% |
| ~10,000 | GroupClassifier starts generalising | May match or exceed binary |
| ~15,000+ | Meaningful CO reduction expected | GroupClassifier should pull ahead |

Re-train by running `ml/scripts/run_training.py --mode group` whenever keyword coverage improves.
No architecture changes needed.

---

## Implementation Status

All phases A–E are implemented. Re-run training after keyword coverage improves.

### Phase A — GroupClassifier model ✓

- [x] `ml/model/group_classifier.py` — `GroupClassifier` nn.Module with sigmoid output, inverse-frequency class weights, save/load

### Phase B — Training data preparation ✓

- [x] `ml/training/group/build_training_data.py` — loads keyword_predictions.csv, builds multi-hot targets, computes class weights

### Phase C — Training script ✓

- [x] `ml/training/group/train.py` — CLI with `--epochs`, `--lr`, `--hidden-dim`, `--threshold`, `--device`, `--max-class-weight`, `--weight-decay`

### Phase D — Inference integration ✓

- [x] `ml/petbert_pipeline/categorization.py` — two-stage group → cosine-within-group inference
- [x] `ml/petbert_pipeline/pipeline.py` — `--group-classifier` CLI flag; uses group-based categorization when present

### Phase E — Evaluation and iteration ✓

- [x] Evaluation via `ml/training/binary/evaluate.py` — results documented in `classifier.md`

---

## File Locations

| File | Purpose |
|------|---------|
| `ml/model/group_classifier.py` | GroupClassifier model definition |
| `ml/training/group/build_training_data.py` | Build embeddings + multi-hot targets from cache + keyword CSV |
| `ml/training/group/train.py` | Training script |
| `ml/model/checkpoints/group_classifier_best.pt` | Saved checkpoint |
| `ml/petbert_pipeline/categorization.py` | Inference integration |
| `ml/petbert_pipeline/pipeline.py` | CLI integration (`--group-classifier` flag) |

---

## Future Improvements

- **More keyword coverage:** As the keyword expert expands coverage, training data quality and
  rare-group sample counts improve automatically. Re-train the GroupClassifier periodically.
- **PetBERT fine-tuning:** Once sufficient labeled data exists for rare groups (>30 cases each),
  fine-tuning PetBERT end-to-end as a direct classifier will likely push accuracy above 90%.
- **Multi-label term prediction:** Currently the term selector uses cosine similarity within the
  predicted group. A separate term-level classifier (per group) could improve term precision once
  per-group sample counts are large enough.
