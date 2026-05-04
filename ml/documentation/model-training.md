# Model Training — Approaches and Architecture

The goal is to map veterinary pathology report text to standardized Vet-ICD-O-canine-1
cancer labels (term, group, ICD code). Three classifier approaches have been explored.

| Approach | Status | Best result |
|---|---|---|
| Binary PresenceClassifier | Superseded | 41.9% G+S (Phase 16, `hidden_dim=512`) |
| GroupClassifier | **Part of 3-stage pipeline (Phase 26)** | **54.6% G+S (Phase 26, 3-stage, per-label eval, test set)** |
| Contrastive fine-tuned PetBERT + 3-stage pipeline | **Production best** | **54.6% G+S test set (Phase 26, per-label eval, gate=0.5, group-t=0.85)** |
| End-to-end fine-tuned PetBERT | WIP, blocked on data volume | — |

> For full current results and Phase 25 details see [classifiers.md](classifiers.md).

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

### Evaluation Pipeline Flow

Standard evaluation compares model predictions against verified annotation labels and
assigns each case to an outcome bucket before logging cycle history.

```mermaid
flowchart TB
    subgraph IN4["Input"]
        A["predictions.csv"]
        B["annotation labels / verified labels"]
    end
    subgraph P4["Process"]
        C["Compare predicted labels to verified labels"]
        E["Good<br>Exact term match"]
        F["Slightly Off<br>Correct group, wrong term"]
        G["Completely Off<br>Wrong group"]
        H["false_positive"]
        I["false_negative"]
    end
    subgraph OUT4["Output"]
        J["evaluation.csv"]
        L["evaluation_history.csv"]
    end

    A --> C
    B --> C
    C --> E & F & G & H & I
    E --> J
    F --> J
    G --> J
    H --> J
    I --> J
    J --> L
```

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

Early results (768-dim mean embedding, incorrect pipeline input — archived for reference):

| Data | Metric | Binary | GroupClassifier @ 0.3 | GroupClassifier @ 0.8 |
|---|---|---|---|---|
| 1,273 cases | Good+Slight | 20.4% | 14.3% | 23.4% |
| 5,788 cases | Good+Slight | 33.1% | 13.9% | 21.9% |

End-to-end results after fixing the pipeline to use 2304-dim `col_emb_concat`:

| Metric | Binary (Phase 16) | GroupClassifier (best, t=0.3) |
|---|---|---|
| **Good+Slight** | **41.9%** | 9.3% |
| CO% | 29.6% | ~37% |
| FP% | 27.2% | 33.3% |
| FN% | 1.2% | 16.8% |

The results above are from Phase 16 (keyword annotation, 5,788 cases). GroupClassifier
became competitive in Phase 23 with ~21,853 LLM-annotated train cases:

| Phase | Train cases | GroupClassifier G+S @ t=0.90 | Notes |
|-------|-------------|------------------------------|-------|
| Phase 16 | 5,788 (keyword) | 9.3% | Severely overfits |
| Phase 23 | 21,853 (LLM, 46,652 total) | **50.1%** | Beats binary (+2.9pp), FP −15.3pp |

See [classifiers.md](classifiers.md) for Phase 23+ full results and the three-stage pipeline (Phase 25).

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

## Approach 3 — Contrastive Fine-tuning (InfoNCE)

### Motivation

The binary classifier's ~30% CO floor comes from labels competing implicitly via
arg max over an embedding space that was never optimized for this task. PetBERT was
pre-trained with masked-language-modeling — its weights have no signal pulling report
embeddings toward their correct label embeddings and away from wrong ones.

Contrastive fine-tuning directly optimizes this geometry. For each (report, label)
positive pair, the report embedding is pulled toward the correct label embedding and
pushed away from all other labels in the batch. The fine-tuned backbone then produces
better per-column embeddings, which the PresenceClassifier (retrained from scratch
after a cold start) uses as input.

Unlike Approach 4 (end-to-end group classification), this does not require group-level
generalization and works at current data volumes.

### Architecture

```
Fine-tuning:
    for each batch of N (report_text, label_text) pairs:
        report_emb = PetBERT.base_model(report_text) → mean pool → 768-dim → L2-norm
        label_emb  = PetBERT.base_model(label_text)  → mean pool → 768-dim → L2-norm
        sim_matrix = report_emb @ label_emb.T / temperature    # (N, N)
        loss = symmetric cross-entropy (diagonal = positives)   # InfoNCE
        backprop through PetBERT base transformer only

    save full AutoModelForMaskedLM checkpoint

Inference (after cold start + PresenceClassifier retraining):
    identical to current pipeline — pass --model <checkpoint> --local-only
```

Training data: keyword-confirmed `(case_id, matched_term, matched_group)` pairs from
`keyword_annotation.csv` joined with report text from `report.csv`.

Label text format: `"{term} {group}"` — exactly what the pipeline uses for label embeddings.

### Results (Phase 17, 2026-03-23)

Fine-tuning config: 3 epochs, batch=32, lr=2e-5, temperature=0.07, 7,398 pairs.
InfoNCE loss: 1.90 → 1.36 → 1.22 (converged normally).

| Metric | Phase 16 (frozen PetBERT) | Phase 17 (contrastive) | Δ |
|--------|--------------------------|------------------------|---|
| **Good+Slight** | 41.9% | **69.0%** | **+27.1pp** |
| CO% | 29.6% | **6.9%** | **−22.7pp** |
| FP% | 27.2% | 23.7% | −3.5pp |
| FN% | 1.2% | 0.3% | −0.9pp |

The CO floor — the core failure mode for the frozen-backbone approach — was shattered.
Contrastive training directly fixed the wrong-group problem: labels that had nearly
identical cosine similarity under the frozen backbone became separable after fine-tuning.

PresenceClassifier cycle trajectory (cold start, hd=512, co=5):

| Cycle | Good+Slight | CO% | Notes |
|-------|-------------|-----|-------|
| c1 | 49.6% | 22.3 | Already above Phase 16 best |
| c2 | 54.5% | 22.1 | |
| c3 | 64.5% | 8.8 | Large jump — CO bank kicking in |
| c4 | 68.0% | 7.9 | |
| c8 | **69.0%** | **6.9** | **Best checkpoint** — plateau confirmed |
| c10 | 68.8% | 7.0 | Plateau oscillation — stopped here |

Best checkpoint: `ml/output/checkpoints/binary/presence_classifier_best.pt`
Backbone: `ml/output/checkpoints/contrastive/`

### Advantages

- Works at current data volumes (~7,398 pairs) — no group-level generalization needed
- Directly optimizes the embedding geometry the PresenceClassifier operates on
- No new inference code — fine-tuned checkpoint is a drop-in replacement for `SAVSNET/PetBERT`
- MLM head weights are unchanged (never called during contrastive forward pass)
- **Proven**: reduced CO floor from ~30% to ~7% in Phase 17

### Disadvantages

- Requires a full cold start after fine-tuning (embedding space changes)
- False negatives in-batch: if the same label appears twice in a batch, the off-diagonal
  entry is wrongly treated as a negative (~4% collision rate at batch_size=32 — acceptable)
- FP floor (~24%) remains — driven by non-cancer cases with similar vocabulary to cancer reports

### How to Run

```bash
# Step 1: Adapt the embedding backbone
ml/.venv/Scripts/python.exe ml/scripts/run_training.py \
  --mode adapt-backbone \
  --epochs 3 --batch-size 32 --lr 2e-5 --temperature 0.07 \
  --device xpu --local-only

# Step 2: Cold start
rm -f ml/output/training/embedding_cache.npz
rm -f ml/output/training/contrastive/evaluation_co_bank.csv
rm -f ml/output/checkpoints/contrastive/presence_classifier_current.pt

# Step 3: Retrain label classifier with the adapted backbone
ml/.venv/Scripts/python.exe ml/scripts/run_training.py \
  --mode train-classifier \
  --model ml/output/checkpoints/contrastive \
  --label "adapted backbone c1" \
  --co-neg-per-case 5 --fp-neg-per-case 10 \
  --embedding-min-sim 0.05 --epochs 25 \
  --recall-weight 0.25 --hidden-dim 512 \
  --device xpu --local-only
```

---

## Approach 4 — End-to-end Fine-tuned PetBERT (WIP)

Fine-tunes PetBERT as a group sequence classifier — architecturally the same two-stage
design as the GroupClassifier (group prediction → within-group cosine term selection),
but replaces the frozen-embedding MLP with a full transformer fine-tuned end-to-end.

This approach shares the GroupClassifier's data ceiling: it needs ~10,000+ confirmed
cases before it can generalize across the 44 groups. It is not recommended until the
GroupClassifier proves competitive.

> **Known code issues** must be resolved before running. See [training-log/training-log-finetune.md](training-log/training-log-finetune.md) for the full list.

---

## Comparison

| | Binary PresenceClassifier | GroupClassifier | Contrastive fine-tuned + 3-stage | End-to-end fine-tuning |
|---|---|---|---|---|
| **Status** | Superseded | **Competitive (Phase 23)** | **Production best** | WIP, blocked |
| **Best result** | 41.9% G+S (Phase 16) | **50.1% G+S @ t=0.90 (Phase 23)** | **62.6% G+S test set (Phase 25, 3-stage)** | Not benchmarked |
| **PetBERT** | Frozen | Contrastive fine-tuned | Fine-tuned (InfoNCE) | Fine-tuned (classification) |
| **Training style** | Iterative (CO feedback) | One-shot | One-shot fine-tune + iterative PresenceClassifier | One-shot |
| **Data requirement** | Works from ~1,273 cases | Competitive at ~21,853 LLM cases | Works at ~5,788 cases | Needs ~10,000+ cases |
| **Training speed** | Fast (MLP on cached embeddings) | Fast (MLP on cached embeddings) | Slow once (full transformer) + fast iterative | Slow (full transformer) |
| **Inference speed** | Slow (~857 pair scores/report) | Fast (~42 group scores + cosine) | Fast (3-stage: gate + ~42 group scores) | Fast (~42 group scores + cosine) |
| **CO floor** | ~30% | ~25.5% @ t=0.90 | **~26% (3-stage, Phase 25)** — backbone-level improvement | Designed to eliminate it |
| **Main constraint** | Superseded | FN trade-off at high threshold | LLM annotation ceiling | Compute cost, data volume |

### Roadmap

- **Now**: Three-stage pipeline (Phase 26) — CasePresenceClassifier + GroupClassifier (F1=0.4335) + KW correction with argmax fallback and subtype keyword discriminators. See [classifiers.md](classifiers.md) for full details.
- **Next**: Reduce 3-stage CO% (22.3%) — backbone adaptation Round 3 with hard-negative mining from CO bank (Tier 4 in `training-ideas/ideas-to-try.md`)
- **Later**: End-to-end fine-tuning (Approach 4) after ~10,000+ confirmed cases

---

## Explored Ideas

### Hybrid Binary + KNN Group Selector (2026-03-23) — Abandoned

**Motivation:** The binary classifier's ~30% CO floor comes from labels competing implicitly
via argmax — a wrong-group prediction cannot be redirected. The idea was to use a KNN group
selector to constrain which groups the binary classifier's scores can be chosen from.

**Architecture:**
```
Per-column embeddings (2304-dim)
          │
          ├──► Binary Classifier ──► (N, M) presence score matrix
          │                           (845 labels scored per case)
          │
          └──► KNN Group Selector ──► (N, G) group vote fractions
                                       (top-K confirmed neighbours vote)
                           │
                           ▼
              For each case, restrict label candidates
              to groups with vote fraction ≥ threshold
                           │
                           ▼
              Pick highest binary score within those groups
```

**Evaluation results** (LLM ground truth, baseline binary-only = 37.8% Good+Slight):

| Run | Config | Good+Slight | CO% | FP% | FN% |
|---|---|---|---|---|---|
| Baseline | Binary only | **37.8%** | 30.1% | 30.3% | 1.8% |
| KNN only | threshold=0.1 | 5.6% | — | — | 25.8% |
| Hybrid | threshold=0.1 | 5.6% | 37.6% | 53.0% | 3.8% |
| Hybrid | threshold=0.2 | 7.5% | 31.9% | 47.7% | 13.0% |
| Hybrid | threshold=0.3 | ~6.1% | ~28% | ~39% | 26.6% |

**Root causes of failure:**
1. "Slightly off" collapses when KNN excludes the correct group.
2. KNN FP floor (53% at threshold=0.1) is not reduced by the binary gate — non-cancer cases still find cancer neighbours in embedding space.
3. KNN sparsity (~150 confirmed cases/group) misses correct groups → FN spikes.

**Conclusion:** Approach abandoned. Revisit when the database grows past ~15,000 confirmed cases.
Code is preserved: `run_categorization_hybrid()` in `categorization.py`, wired in `pipeline.py`.
Full root-cause analysis in [training-log/training-log-binary.md](training-log/training-log-binary.md).
