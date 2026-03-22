# Model Training — Approaches and Architecture

The goal is to map veterinary pathology report text to standardized Vet-ICD-O-canine-1
cancer labels (term, group, ICD code). Three classifier approaches have been explored.

| Approach | Status | Best result |
|---|---|---|
| Binary PresenceClassifier | Production best | 40.0% Good+Slight |
| GroupClassifier | Implemented, not yet competitive | 21.9% Good+Slight |
| Fine-tuned PetBERT | Work in progress | Not benchmarked |

---

## Background: Ground Truth and Evaluation

### Ground Truth Generation

No labelled cancer dataset exists — only free-text diagnosis strings written by
pathologists. A separate keyword pipeline scans these diagnosis strings against the
Vet-ICD-O taxonomy to produce ground-truth labels. Cases with no keyword match are
treated as non-cancer (Uncategorized).

In production, only the full pathology report text is available (not the structured
diagnosis field). All three classifier approaches must therefore predict from report
text alone, using the keyword-matched labels only for training supervision.

### Evaluation Verdicts

Predictions are scored against keyword ground truth per case:

| Verdict | Meaning |
|---------|---------|
| `good` | Predicted term exactly matches a keyword-matched term |
| `slightly_off` | No exact term match, but predicted group matches a keyword group |
| `completely_off` | Neither term nor group matches any keyword label for this case |
| `false_positive` | Case has no keyword labels but a cancer label was predicted |
| `false_negative` | Confirmed cancer case with no good/slightly_off prediction |

**Good+Slight** is the primary performance metric. The CO rate (completely off) is the
key failure mode — the classifier is confidently predicting the wrong cancer group.

---

## Approach 1 — Binary PresenceClassifier

### Architecture

Each report section (Histopathological Summary, Final Comment, Ancillary Tests) is
embedded independently through PetBERT, producing a 768-dim vector per column. The
MLP then scores every (report, label) pair independently by concatenating the three
column embeddings with the label embedding:

```
report text
    ↓
PetBERT (frozen)
    ↓
per-column embeddings (3 × 768)
    ↓
for each of ~857 taxonomy labels:
    concat(col1_emb ‖ col2_emb ‖ col3_emb ‖ label_emb)
    ↓
    binary MLP → present/absent score
    ↓
argmax across all label scores → predicted label
```

Labels compete implicitly — the one with the highest present/absent score wins.

### Training Strategy

Training is **iterative**. Each cycle:

1. Run the pipeline with the current checkpoint → predictions
2. Evaluate predictions → identify completely-off (CO) and false-positive (FP) cases
3. Accumulate CO predictions into a rolling bank
4. Retrain using positives + CO negatives + FP negatives from the bank
5. Repeat

The rolling CO bank is the key mechanism. It ensures the classifier trains on the
specific wrong-group pairs that fool cosine similarity, compounding across cycles.

| Training data source | Description |
|---|---|
| Positives | Keyword-confirmed (report, term) pairs |
| CO negatives | Completely-off predictions from the rolling CO bank |
| FP negatives | Labels sampled for false-positive cases |
| Easy negatives | Random wrong labels for confirmed cancer cases |

### Advantages

- Works at low data volumes — competitive from ~1,273 confirmed cases upward
- Iterative CO feedback produces steady improvement across cycles
- Fast training — MLP trains on cached PetBERT embeddings, no re-embedding needed

### Disadvantages

- **Hard CO floor (~30%)**: labels compete implicitly via argmax. A wrong-group
  prediction cannot be redirected — only the individual pair can be rejected.
- Pairwise scoring over all ~857 labels is slow at inference
- Label enrichment attempts caused regressions and are off by default

---

## Approach 2 — GroupClassifier

### Motivation

The binary classifier's CO floor arises because labels compete implicitly through
argmax after independent pairwise scoring. There is no mechanism to say "wrong group
entirely" — the classifier can only lower the score of a specific pair.

The GroupClassifier addresses this directly: predict the cancer *group* first (explicit
group competition in the loss), then select the specific *term* within that group.

### Architecture

```
report text
    ↓
PetBERT (frozen)
    ↓
mean embedding (768-dim, averaged across all columns)
    ↓
GroupClassifier MLP → sigmoid score per group (45 outputs)
    ↓
threshold → predicted group(s)
    ↓
for each predicted group:
    cosine similarity against terms within that group only
    ↓
    best term + ICD code
```

The MLP produces one sigmoid probability per group simultaneously. Sigmoid (not
softmax) because a report can belong to multiple groups. Loss is binary cross-entropy
per class with inverse-frequency class weights.

```
GroupClassifier MLP:
    Linear(768 → 256) → ReLU → Dropout(0.3) → Linear(256 → 45) → Sigmoid
```

Training is **one-shot**: build multi-hot targets from keyword-matched cases, train
on cached embeddings, evaluate. Re-run whenever keyword coverage improves.

| Training data | Cases | Label |
|---|---|---|
| Cancer (keyword-matched) | ~5,788 (44 groups) | Multi-hot over matched groups |
| Non-cancer | ~6,832 | Uncategorized (all zeros) |

### Evaluation Results

| Data | Metric | Binary | GroupClassifier @ 0.3 | GroupClassifier @ 0.8 |
|---|---|---|---|---|
| 1,273 cases | Good+Slight | 20.4% | 14.3% | 23.4% |
| 1,273 cases | CO% | 42.7% | 55.9% | 50.7% |
| **5,788 cases** | **Good+Slight** | **33.1%** | 13.9% | 21.9% |
| **5,788 cases** | **CO%** | **31.8%** | 57.5% | 54.5% |
| **5,788 cases** | **FN%** | **1.3%** | — | 15.6% |

The GroupClassifier overfits at current data volumes (~132 cases/group average).
Binary wins at 5,788 cases. Expected crossover at ~10,000 confirmed cases.

| Confirmed cases | Expected outcome |
|---|---|
| ~5,788 (current) | Overfits — binary wins |
| ~10,000 | GroupClassifier starts generalising — may match binary |
| ~15,000+ | Meaningful CO reduction expected — GroupClassifier should pull ahead |

### Advantages

- Wrong-group assignments are directly penalised in the loss — designed to eliminate
  the CO floor once sufficient data is available
- Faster inference: cosine over ~20 terms per group instead of all ~857 labels
- Simple re-training: one-shot, seconds on cached embeddings

### Disadvantages

- Overfits at current data volume (5,788 cases across 44 groups)
- Uses mean embedding — per-column signal is averaged away
- A mis-predicted group guarantees a wrong term (no recovery path within-group)

### Potential Improvement: Discriminating-Keyword Term Selection

Instead of cosine similarity within the predicted group, the group's own taxonomy
labels can be used to auto-derive discriminating keywords, which are then scanned
for in the report text to pick the specific term.

**Example — "Neoplasms, NOS" group:**

| Term | Discriminating words |
|---|---|
| Neoplasm, benign | `benign` |
| Neoplasm, malignant | `malignant` |
| Neoplasm, NOS | *(fallback)* |

Report says "...consistent with a **malignant** neoplasm..." → "Neoplasm, malignant"

**Example — "Mast cell neoplasms" group:**

| Term | Discriminating words |
|---|---|
| Mast cell tumor, grade I | `grade i`, `grade 1` |
| Mast cell tumor, grade II | `grade ii`, `grade 2` |
| Mast cell tumor, grade III | `grade iii`, `grade 3` |
| Mast cell leukemia | `leukemia` |
| Mast cell tumor, NOS | *(fallback)* |

**Proposed inference flow:**
```
group_classifier → predicted group
        ↓
discriminating keyword scan on report text
        ↓ (match)                    ↓ (no match)
specific term             cosine similarity within group (fallback)
```

Discriminating words are words that appear in *some but not all* terms within a group
— auto-derivable from the taxonomy CSV without manual curation. They are short
qualifiers (`benign`, `grade ii`) rather than full tumor names, making them less prone
to false matches. This idea applies equally to Approach 3, which shares the same
within-group term selection step.

---

## Approach 3 — Fine-tuned PetBERT

### Architecture

Fine-tunes PetBERT end-to-end as a group classifier, removing the frozen-embedding
bottleneck. The keyword pipeline still generates training labels from diagnosis text;
at inference, the fine-tuned model predicts the group directly from report text.

```
Training:
    (report text, group label from keyword pipeline)
        ↓
    fine-tune PetBERT (AutoModelForSequenceClassification)

Inference:
    report text
        ↓
    fine-tuned PetBERT → softmax over 45 groups
        ↓
    predicted group probabilities
        ↓
    cosine similarity within predicted group
    (using base, unfinetuned PetBERT for label embeddings)
        ↓
    best term + ICD code
```

### Relationship to Approach 2

Approach 3 is architecturally the same two-stage design as Approach 2 (group
prediction → within-group term selection). The only difference is that the MLP head
operating on frozen embeddings is replaced by a full transformer fine-tuned end-to-end.
The two are not complementary — Approach 3 is an upgrade of the group prediction
stage, not an addition to it.

### Advantages

- **No frozen-embedding bottleneck**: the transformer weights adapt to veterinary
  diagnostic language, potentially capturing nuances that frozen embeddings miss
- More expressive capacity than an MLP on fixed embeddings
- Mirrors the GroupClassifier training strategy — same supervision signal, stronger model

### Disadvantages

- **Computationally expensive**: full transformer fine-tuning requires significantly
  more GPU time and memory than training an MLP on cached embeddings
- **Risk of catastrophic forgetting**: PetBERT's pretrained veterinary language
  representations may degrade if the learning rate or epoch count is not managed carefully
- Cached embeddings cannot be reused — every run requires a full forward pass
- Still shares the group-level ceiling: term selection within the predicted group
  remains cosine-based (unless discriminating keywords are used)

---

## Comparison

| | Binary PresenceClassifier | GroupClassifier | Fine-tuned PetBERT |
|---|---|---|---|
| **Status** | Production best | Implemented, not competitive | Work in progress |
| **Best result** | 40.0% Good+Slight | 21.9% Good+Slight | Not benchmarked |
| **PetBERT** | Frozen | Frozen | Fine-tuned end-to-end |
| **Training style** | Iterative (CO feedback) | One-shot | One-shot |
| **Data requirement** | Works from ~1,273 cases | Needs ~10,000+ cases | Needs ~10,000+ cases |
| **Training speed** | Fast (MLP on cached embeddings) | Fast (MLP on cached embeddings) | Slow (full transformer) |
| **Inference speed** | Slow (~857 pair scores/report) | Fast (~45 group scores + cosine) | Fast (~45 group scores + cosine) |
| **CO floor** | ~30% | Designed to eliminate it | Designed to eliminate it |
| **Main constraint** | CO floor, keyword data ceiling | Overfits at current data volume | Compute cost |

### Roadmap

- **Now**: Binary PresenceClassifier — best results at current data volume
- **When keyword coverage reaches ~10,000 cases**: Re-train GroupClassifier and benchmark against binary; implement discriminating-keyword term selection
- **After GroupClassifier proves competitive**: Consider fine-tuning PetBERT for additional gains
