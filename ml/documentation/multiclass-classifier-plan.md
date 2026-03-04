# Multi-Class Group Classifier — Design Plan

## Problem Statement

The current binary `PresenceClassifier` MLP scores each `(report_embedding, label_embedding)` pair
independently and selects the winner by argmax. Because groups compete implicitly rather than
explicitly, the classifier cannot redirect a wrong-group cosine match — it can only accept or
reject individual pairs. This produces a hard **~42% completely-off (CO) floor** and a **~20%
Good+Slight ceiling**, regardless of how long training continues.

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
Result: ~42% CO, ~20% Good+Slight ceiling.

### Proposed (multi-class GroupClassifier)

```
report text
    → PetBERT (frozen) → mean_embedding (768-dim)
    → GroupClassifier MLP → score per group (40 outputs, sigmoid)
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

The keyword scan (training pipeline) provides ground-truth labels via
`ml/output/diagnoses/keyword_predictions.csv`.

| Split | Cases | Label |
|-------|-------|-------|
| Cancer cases (keyword-matched) | ~1,273 unique cases | Multi-hot encoding over matched_group values |
| Non-cancer cases (no keyword match) | ~1,510 unique cases | Uncategorized (all zeros except Uncategorized bit) |
| **Total** | **~2,783 cases** | |

**Ground-truth assumption:** Cases not matched by the keyword scan are non-cancer.
This is valid for a general veterinary clinic population (~18% cancer prevalence).
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

### Class distribution (current keyword coverage)

| Group | Cases | Notes |
|-------|-------|-------|
| Adenomas and adenocarcinomas | 209 | |
| Blood vessel tumors | 153 | |
| Osseous and chondromatous neoplasms | 122 | |
| Epithelial neoplasms, NOS | 110 | |
| Lipomatous neoplasms | 96 | |
| Soft tissue tumors and sarcomas, NOS | 82 | |
| Mast cell neoplasms | 82 | |
| Malignant lymphomas, NOS or diffuse | 76 | |
| ... | ... | |
| 14 groups with <10 cases | 1–8 each | Underrepresented until keyword coverage grows |
| **Uncategorized** | **~1,510** | Non-cancer majority class |

Class imbalance is handled via **per-class BCE loss weights** (inverse frequency weighting).

---

## Model Architecture

```
GroupClassifier(
    input_dim  = 768,        # PetBERT mean_embedding
    hidden_dim = 256,
    num_classes = 40,        # 39 cancer groups + 1 Uncategorized
    dropout = 0.3
)

forward(x):
    x = ReLU(Linear(768 → 256))
    x = Dropout(0.3)
    x = Sigmoid(Linear(256 → 40))   # independent probability per class (multi-label)
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
2. Forward pass through GroupClassifier → 40 group probabilities
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

The existing evaluation framework (`evaluate_predictions.py`) applies directly.

### Baseline results (current keyword coverage, ~1,273 cancer cases)

Tested 2026-03-04 with 50 epochs, 39 groups, MPS device:

| Metric | Phase 9 (binary) | GroupClassifier @ 0.3 | GroupClassifier @ 0.8 |
|--------|------------------|-----------------------|-----------------------|
| Good+Slight | **20.4%** | 14.3% | **23.4%** |
| CO% | 42.7% | 55.9% | 50.7% |
| FP% | ~30% | 28.0% | **8.4%** |
| FN% | 4% | 1.8% | 17.6% |
| Predictions | ~9,500 | 8,608 | 3,255 |

**Findings:**
- At threshold 0.8: marginally better Good+Slight, much lower FP, but significantly higher FN
- At threshold 0.3: more predictions, lower FN, but worse CO and Good+Slight than Phase 9
- CO is not eliminated because 1,273 labeled cases is insufficient for the 768→39 MLP to
  generalise — val loss (2.6) far exceeds train loss (0.38), indicating clear overfitting
- The model correctly identifies ~90% of cancer cases at val (high recall) but with very
  low precision (~0.1), flooding the output with wrong-group predictions

**Why the CO floor persists at current data volumes:**
The frozen PetBERT embedding space does not have sufficient discriminative power across
39 cancer groups with ~33 examples per group on average. The MLP (~200k parameters,
~2,276 training cases) memorises rather than generalises.

### Expected trajectory as keyword coverage grows

| Keyword-confirmed cases | Expected Good+Slight | Notes |
|------------------------|---------------------|-------|
| ~1,273 (current) | ~20–23% | Comparable to Phase 9 |
| ~3,000 | 35–50% | Model starts generalising |
| ~5,000+ | 60–80% | Meaningful CO reduction |

Re-train by running `build_group_training_data.py` + `train_group_classifier.py` whenever
keyword coverage improves. No architecture changes needed.

---

## Implementation Checklist

### Phase A — GroupClassifier model

- [ ] Create `ml/model/group_classifier.py`
  - `GroupClassifier` nn.Module (architecture above)
  - `train_group_classifier(embeddings, targets, class_weights, ...)` training function
  - `predict(embeddings, threshold)` inference function
  - `save / load` checkpoint utilities

### Phase B — Training data preparation

- [ ] Create `ml/scripts/build_group_training_data.py`
  - Load `keyword_predictions.csv` → multi-hot group targets per case
  - Load `embedding_cache.npz` → mean_embeddings per case
  - Identify non-cancer cases (cases in report.csv not in keyword_predictions.csv)
  - Assign Uncategorized target to non-cancer cases
  - Compute inverse-frequency class weights
  - Output: `(embeddings, targets, class_weights, case_ids)` tensors

### Phase C — Training script

- [ ] Create `ml/scripts/train_group_classifier.py`
  - CLI args: `--epochs`, `--lr`, `--hidden-dim`, `--threshold`, `--device`
  - Train/val split (80/20, stratified by group)
  - Per-epoch logging: loss, per-group precision/recall
  - Checkpoint: save best model by macro F1 on validation set

### Phase D — Inference integration

- [ ] Update `ml/petbert_scan/categorization.py`
  - Add `run_categorization_group(group_classifier, mean_embeddings, taxonomy, threshold)`
  - For each predicted group: cosine similarity against group-restricted term list → top term
  - Preserve existing output format

- [ ] Update `ml/petbert_scan/pipeline.py`
  - Add `--group-classifier` CLI flag (path to GroupClassifier checkpoint)
  - When present: use group-based categorization instead of binary PresenceClassifier

### Phase E — Evaluation and iteration

- [ ] Run evaluation with existing `evaluate_predictions.py`
- [ ] Tune threshold on validation set (precision/recall tradeoff for CO vs FN)
- [ ] Document results in `classifier.md`

---

## File Locations

| File | Purpose |
|------|---------|
| `ml/model/group_classifier.py` | GroupClassifier model definition |
| `ml/scripts/build_group_training_data.py` | Build embeddings + multi-hot targets from cache + keyword CSV |
| `ml/scripts/train_group_classifier.py` | Training script |
| `ml/model/checkpoints/group_classifier_best.pt` | Saved checkpoint |
| `ml/petbert_scan/categorization.py` | Inference integration (update existing) |
| `ml/petbert_scan/pipeline.py` | CLI integration (update existing) |

---

## Future Improvements

- **More keyword coverage:** As the keyword expert expands coverage, training data quality and
  rare-group sample counts improve automatically. Re-train the GroupClassifier periodically.
- **PetBERT fine-tuning:** Once sufficient labeled data exists for rare groups (>30 cases each),
  fine-tuning PetBERT end-to-end as a direct classifier will likely push accuracy above 90%.
- **Multi-label term prediction:** Currently the term selector uses cosine similarity within the
  predicted group. A separate term-level classifier (per group) could improve term precision once
  per-group sample counts are large enough.
