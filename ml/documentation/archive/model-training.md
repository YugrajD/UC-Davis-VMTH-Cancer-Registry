# Model Training Approaches

This document describes the three approaches to training the cancer classification model, covering how each works, its advantages, disadvantages, and constraints.

All three approaches use the same ground truth: the keyword pipeline scans the diagnosis text field and maps it to Vet-ICD-O labels. Those labels are the training signal for every approach. In production, no diagnosis text is available — only report text.

---

## Approach 1 — Binary PresenceClassifier

**Status: Implemented. Current best: 40.0% Good+Slight (Phase 13).**

### How it works

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
argmax across all label scores
    ↓
predicted label
```

PetBERT is frozen throughout. The three report text columns (HISTOPATHOLOGICAL SUMMARY, FINAL COMMENT, ANCILLARY TESTS) are each embedded separately and concatenated rather than averaged, giving the MLP column-specific signal. The classifier scores every (report, label) pair independently and the highest-scoring label wins.

Training is **iterative**: each cycle runs the pipeline, evaluates predictions, accumulates the wrong-group (completely-off) predictions into a rolling bank, and retrains the classifier on that feedback. This self-correcting loop drives down the CO rate over successive cycles.

### Advantages

- Works at low data volumes — competitive from ~1,273 confirmed cases upward
- Stable training with the rolling CO bank and `--recall-weight 0.25`; no bad/degenerate cycles
- Simple, fast training — runs on the cached embeddings, no PetBERT inference needed per cycle
- Backward-compatible checkpoint format; `n_cols` is serialised into the checkpoint

### Disadvantages

- **Hard CO floor (~30%)**: labels compete implicitly via argmax after independent scoring. The classifier cannot redirect a wrong-group match — it can only reject individual pairs, not compare groups against each other.
- Pairwise scoring scales with the number of labels (~857 pairs per report at inference)
- Enrichment attempts (Fix 6, Fix 9) caused regressions — label enrichment is off by default
- Cross-column interactions may not be learnable at current data volumes — per-column independent scoring (one MLP pass per column + aggregate) is an untested alternative that reduces input dimensionality and avoids zeroed columns polluting the input; see [presence-classifier-optimizations.md](../../planning/presence-classifier-optimizations.md) Idea D

### Constraints

- `--co-neg-per-case 5` — do **not** raise to 10 with the per-column architecture; causes regression (Phase 13 c2 experience)
- Requires embedding cache (`ml/data/embedding_cache.npz`) and a CO bank (`evaluation_co_bank.csv`)
- Cold start required after any architecture change (embedding cache and CO bank must be deleted)
- Keyword coverage is the main ceiling: 5,788 confirmed cases produced 40% Good+Slight; further improvement requires more labeled data or a group-level architecture

---

## Approach 2 — Multi-class GroupClassifier

**Status: Implemented. Not yet competitive (21.9% @ t=0.8 vs binary 40.0%). Needs ~10,000+ confirmed cases.**

### How it works

```
report text 
    ↓
PetBERT (frozen) 
    ↓
mean_embedding (768-dim)
    ↓
GroupClassifier MLP 
    ↓
sigmoid score per group (45 outputs)
    ↓
threshold → predicted group(s)
    ↓
for each predicted group:
    cosine similarity against terms within that group only
            ↓
     best term + ICD code
```

Instead of scoring each label independently, the GroupClassifier makes one global decision per report: which cancer group(s) does this report belong to? Groups compete explicitly in the loss function (multi-label BCE). Term selection is then a simpler sub-problem — cosine similarity within only the ~20 terms of the predicted group rather than across all 857 labels.

Training is **one-shot**: build multi-hot targets from `keyword_predictions.csv`, train the MLP on cached embeddings, evaluate. Re-run whenever keyword coverage improves — no architectural changes needed.

### Advantages

- Explicit group competition: wrong-group assignments are directly penalized in the loss, not just rejected pair-wise after the fact
- Designed to break the CO floor — a report assigned to the right group cannot produce a completely-off term
- Faster inference: cosine search across ~20 terms instead of ~857
- Simple re-training: one-shot, trains in seconds on the cached embeddings

### Disadvantages

- **Overfits at current data volume**: with 5,788 cases across 44 groups (~132 cases/group average), the MLP memorises rather than generalises. Validation loss diverges from training loss.
- Currently worse than the binary classifier at all thresholds (21.9% vs 40.0% Good+Slight)
- Uses mean embedding (not per-column concatenation) — some column-specific signal is lost
- A mis-predicted group at the group stage guarantees a wrong-term at the term stage (no recovery)

### Constraints

- Requires ~10,000+ confirmed cancer cases across 44 groups to generalise reliably; rare groups need at least 30–100 examples each
- Must be re-trained from scratch whenever the embedding cache is rebuilt or keyword coverage changes
- Group threshold is a tunable parameter: lower threshold (0.3) produces fewer FN but more CO; higher threshold (0.8) cuts CO at the cost of more FN

---

## Approach 3 — Fine-tuned PetBERT

**Status: Work in progress. Not yet benchmarked.**

### How it works

```
Training:
    diagnosis text
        ↓
    keyword pipeline
        ↓
    group label
        ↓
    fine-tune PetBERT (AutoModelForSequenceClassification) on (report text, group label)
        ↓
    save HuggingFace checkpoint

Inference:
    report text
        ↓
    fine-tuned PetBERT
        ↓
    softmax over 45 groups
        ↓
    predicted group probabilities
        ↓
    cosine similarity within predicted group
        ↓
    best term + ICD code
```

PetBERT is fine-tuned end-to-end as a sequence classifier, replacing the frozen-embedding + separate classifier-head architecture. The keyword pipeline still generates the training labels from the existing diagnosis text — but at inference, the diagnosis field is not needed. PetBERT has learned to predict cancer groups directly from report text.

Training uses the HuggingFace Trainer API (`WeightedTrainer` with class-weighted `CrossEntropyLoss`). Within-group term selection still uses cosine similarity against taxonomy label embeddings produced by the base (unfinetuned) PetBERT.

### Advantages

- **No frozen-embedding bottleneck**: PetBERT's weights adapt to the task, potentially learning clinical vocabulary and diagnostic patterns that generic masked-LM pre-training did not
- A single model replaces PetBERT + GroupClassifier — simpler inference pipeline
- HuggingFace Trainer handles mixed precision, gradient accumulation, and checkpoint selection automatically
- The training strategy mirrors the GroupClassifier (keyword labels → group supervision) but with more expressive capacity

### Disadvantages

- **Computationally expensive**: full PetBERT fine-tuning requires significantly more GPU time and memory than training an MLP head on cached embeddings
- **Risk of catastrophic forgetting**: fine-tuning may degrade PetBERT's general veterinary language representations if learning rate or epoch count is not carefully managed
- **No cached embeddings**: inference cannot reuse the existing `embedding_cache.npz` — every run requires a full PetBERT forward pass
- More complex training pipeline (HuggingFace `datasets`, `Trainer`, checkpoint format) vs. a plain PyTorch MLP
- Still shares the group-level ceiling: term selection within the predicted group remains cosine-based

### Constraints

- Known code issues to resolve before a full training run (see `classifier.md` WIP section):
  - `WeightedTrainer` constructor argument order is fragile
  - Class weights tensor moved to device during `__init__` before device is resolved
  - No stratified val split in `build_dataset.py`
  - No guard against `--presence-classifier` and `--finetuned-model-path` being set simultaneously
- Needs sufficient labeled data — likely requires at least the same data volume as the GroupClassifier (~10,000+ cases) to fine-tune reliably and avoid memorisation
- `build_dataset.py` uses `padding="max_length"` at build time, producing a large on-disk dataset; switching to dynamic padding (`DataCollatorWithPadding`) would reduce memory and disk usage
- MPS (Apple Silicon) fallback must be enabled: `PYTORCH_ENABLE_MPS_FALLBACK=1`

---

## Comparison Summary

| | Binary PresenceClassifier | GroupClassifier | Fine-tuned PetBERT |
|---|---|---|---|
| **Status** | ✅ Production best | ✅ Implemented, not competitive | 🚧 WIP |
| **Best result** | 40.0% Good+Slight | 21.9% Good+Slight | Not benchmarked |
| **PetBERT** | Frozen | Frozen | Fine-tuned end-to-end |
| **Training style** | Iterative (CO feedback) | One-shot | One-shot |
| **Data requirement** | Works from ~1,273 cases | Needs ~10,000+ cases | Needs ~10,000+ cases |
| **Training speed** | Fast (MLP on cached embeddings) | Fast (MLP on cached embeddings) | Slow (full transformer) |
| **Inference speed** | Slow (~857 pair scores/report) | Fast (~45 group scores + cosine within group) | Fast (~45 group scores + cosine within group) |
| **CO floor** | ~30% | Designed to eliminate it (not yet demonstrated) | Designed to eliminate it (not benchmarked) |
| **Main constraint** | CO floor, keyword data ceiling | Overfits at current data volume | Compute cost, known code issues |

### When to use each

- **Now**: Binary PresenceClassifier — best results at current data volume
- **When keyword coverage reaches ~10,000 cases**: Re-train GroupClassifier and benchmark against binary
- **After GroupClassifier proves competitive**: Consider fine-tuning PetBERT as the next improvement, after resolving the known code issues
