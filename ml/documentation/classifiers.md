# Classifiers

The classifier layer sits on top of frozen PetBERT embeddings and decides which cancer
label(s) a report belongs to. Three approaches exist; the binary PresenceClassifier is
currently the best performer.

| Classifier | Status | Best result |
|---|---|---|
| Binary PresenceClassifier | Production best | 41.9% Good+Slight (Phase 16, `hidden_dim=512`) |
| GroupClassifier | Implemented, not yet competitive | 21.9% Good+Slight @ t=0.8 |
| Fine-tuned PetBERT | WIP — see `petbert-pipeline.md` | Not benchmarked |

> **Training ground truth:** All three approaches derive training labels from the keyword
> pipeline (`keyword_predictions.csv`). The keyword pipeline maps diagnosis field text to
> Vet-ICD-O labels. Cases with no keyword match are treated as non-cancer (Uncategorized).
> In production, no diagnosis text is available — classifiers predict from report text alone.

---

## Evaluation Verdicts

All approaches are evaluated by `ml/training/binary/evaluate.py`:

| Verdict | Meaning |
|---------|---------|
| `good` | Predicted term exactly matches a keyword-matched term |
| `slightly_off` | No exact term match but predicted group matches a keyword group |
| `completely_off` | Neither term nor group matches any keyword label for this case |
| `false_positive` | Case has no keyword labels (should be Uncategorized) |
| `false_negative` | Confirmed cancer case with no good/slightly_off prediction |

---

## Approach 1 — Binary PresenceClassifier

### Architecture

```
report text
    ↓
PetBERT (frozen)
    ↓
per-column embeddings (3 × 768)
    ↓
for each of ~857 taxonomy labels:
    concat(col1_emb ‖ label_emb) → MLP → score1
    concat(col2_emb ‖ label_emb) → MLP → score2   (shared weights)
    concat(col3_emb ‖ label_emb) → MLP → score3
    max(score1, score2, score3)
    ↓
argmax across all label scores
    ↓
predicted label
```

Each column is paired with the label embedding independently (1536-dim input per pair)
and scored by a shared MLP. Max-pooling across the three per-column logits gives the
final score — the most informative column wins. Empty columns are zeroed before pairing.
Labels compete implicitly through argmax.

**Why per-pair over concat:** The Phase 13 concat architecture compressed 3072-dim into
a 256-dim hidden layer (12:1 ratio). Per-pair halves the input to 1536-dim (6:1 ratio),
directly addressing the known compression bottleneck. Column-label relationships are
also learned independently rather than entangled.

Two modes are stored in the checkpoint (`col_pair_mode`):

| `col_pair_mode` | Input dim | Aggregation | Notes |
|---|---|---|---|
| `True` (default) | 2 × 768 = 1536 | max over columns | Phase 14+ |
| `False` (legacy) | 4 × 768 = 3072 | — | Phase 13 and earlier |

`n_cols` and `col_pair_mode` are saved into the checkpoint. Legacy checkpoints (no
`col_pair_mode` key) default to `False` and load without modification.

### Training Data Sources

| Source | Description |
|--------|-------------|
| `positive` | Keyword-confirmed (case, term) pairs |
| `hard_negative` | False-positive predictions from previous eval cycle |
| `fp_extra_negative` | Additional random labels sampled for FP cases |
| `co_negative` | Completely-off predictions from the rolling CO bank |
| `easy_negative` | Random wrong labels for keyword-confirmed cases |

Training is **iterative**: each cycle runs the pipeline, evaluates, accumulates
completely-off predictions into a rolling bank, and retrains. The CO bank is the key
mechanism — it ensures the classifier always trains on the specific wrong-group pairs
that fool cosine similarity, accumulated across all previous cycles.

### How to Run

> **Windows:** use `ml/.venv/Scripts/python.exe`. Adjust `--device` to your hardware (`xpu`, `cuda`, `mps`, `cpu`).

**Continuing from an existing bank (standard):**
```bash
ml/.venv/bin/python3 ml/scripts/run_training.py \
  --mode binary \
  --label "cycle N" \
  --co-neg-per-case 5 \
  --fp-neg-per-case 10 \
  --embedding-min-sim 0.05 \
  --epochs 25 \
  --recall-weight 0.25 \
  --device mps \
  --local-only
```

### Cold Start

Required any time the embedding space changes (PetBERT update, architecture change,
new keyword data). Old bank pairs are anchored to the old cosine space and will add noise.

**Prerequisites:**
1. `ml/output/diagnoses/keyword_predictions.csv` must exist. If not:
   ```bash
   ml/.venv/bin/python3 -m keyword_pipeline
   ```
2. `ml/data/report.csv` must exist.

**Files to delete:**
```bash
rm -f ml/data/embedding_cache.npz
rm -f ml/output/evaluation/evaluation_co_bank.csv
rm -f ml/model/checkpoints/presence_classifier_current.pt
```

**Cold-start c1 command** (same as standard, label it clearly):
```bash
ml/.venv/bin/python3 ml/scripts/run_training.py \
  --mode binary \
  --label "cold-start c1" \
  --co-neg-per-case 5 \
  --fp-neg-per-case 10 \
  --embedding-min-sim 0.05 \
  --epochs 25 \
  --recall-weight 0.25 \
  --device mps \
  --local-only
```

Step 0 detects the missing cache and runs PetBERT on all reports and labels — the only
time PetBERT runs. All subsequent cycles load from cache.

Continue all subsequent cycles with `--co-neg-per-case 5`. Do **not** switch to 10.

### Expected Trajectory (5,788 cancer cases, per-column architecture)

| Cycles completed | Expected Good+Slight | Notes |
|-----------------|---------------------|-------|
| c1 (co=5, cold start) | ~28–30% | Cache rebuilt; bank ~15k |
| c2 (co=5) | ~26% | May dip slightly — continue |
| c3–c4 (co=5) | ~38–39% | Large jump |
| c5–c6 (co=5) | ~39–40% | Plateau; CO floor ~30% |
| c7+ | may regress | Confirm plateau and stop |

### Key Parameters

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `--embedding-min-sim` | `0.05` | Scores are mean-subtracted — use 0.05, not 0.5 |
| `--co-neg-per-case` | `5` | Do NOT raise to 10 with per-column architecture — causes regression |
| `--fp-neg-per-case` | `10` | Keep at 10; reducing to 5 weakens FP rejection |
| `--epochs` | `25` | Beyond 25 shows diminishing returns |
| `--pos-weight` | `1.0` | Do not increase; sampler already balances training |
| `--recall-weight` | `0.25` | Prevents epoch-1 degenerate checkpoints winning. Do not raise above 0.5 |
| `--max-pos-per-group` | `0` (no cap) | Do not cap — removes signal from already-good groups |

### Advantages

- Works at low data volumes — competitive from ~1,273 confirmed cases upward
- Stable training with the rolling CO bank and `--recall-weight 0.25`
- Fast — trains on cached embeddings, no PetBERT inference needed per cycle

### Disadvantages

- **Hard CO floor (~30%)**: labels compete implicitly via argmax. Cannot redirect a
  wrong-group match — only reject individual pairs.
- Pairwise scoring scales with number of labels (~857 scores per report at inference)
- Enrichment attempts (Fix 6, Fix 9) caused regressions — off by default

### Known Limitations

- CO floor is data-limited, not architecture-limited. Further reduction requires more
  keyword-confirmed cases or a group-level architecture.
- `hidden_dim=512` (Phase 16) resolved the 12:1 compression bottleneck — confirmed +1.9pp gain.
  Trying `hidden_dim=768` may recover additional signal.

### Phase 14 Experiment — Per-pair architecture (`col_pair_mode=True`) ❌

Tested 2026-03-21: score each `[colN | label]` pair (1536-dim) independently with a shared
MLP, then max-pool across columns. Plateaued at **32.7%** Good+Slight (c6) vs Phase 13's
**40.0%** — a −7.3pp regression. Reverted. See training-log-binary.md Phase 14 for analysis.

Per-pair experiment backed up as `presence_classifier_best_phase14_perpair.pt`.

### Phase 16 Experiment — Wider hidden layer (`hidden_dim=512`) ✅

Tested 2026-03-22: same concat architecture as Phase 13 (`col_pair_mode=False`, `n_cols=3`) but
`hidden_dim=512` instead of 256, reducing the input compression ratio from 12:1 to 6:1.

Plateaued at **41.9%** Good+Slight (c2) — **+1.9pp over Phase 13 (40.0%)**.
- c1 already beat Phase 13 at 40.6% with no cold start
- c2–c4 held at ~41.7–41.9%; c5 regressed — plateau confirmed
- FP% also improved: 27.1% vs 28.5% in Phase 13

**The 12:1 compression bottleneck was real and is resolved.** `hidden_dim` is now saved in the
checkpoint so `load()` reconstructs the correct size automatically.

**Production checkpoint updated to `presence_classifier_best.pt` (Phase 16, 41.9%).**
Backed up as `presence_classifier_best_phase16_hd512.pt`.

---

## Approach 2 — GroupClassifier

### Architecture

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

Instead of scoring each label independently, the GroupClassifier makes one global
decision per report: which cancer group(s) does this report belong to? Groups compete
explicitly in the loss function (multi-label BCE). Term selection is then a simpler
sub-problem — cosine similarity within only the ~20 terms of the predicted group.

**Why this design:** The binary classifier's ~30% CO floor comes from labels competing
implicitly through argmax after independent scoring. The GroupClassifier directly
penalizes wrong-group assignments in the loss, eliminating the CO problem at the
group level (once it generalises).

### Model Architecture

```
GroupClassifier(
    input_dim   = 768,     # PetBERT mean_embedding
    hidden_dim  = 256,
    num_classes = 45,      # 44 cancer groups + 1 Uncategorized
    dropout     = 0.3
)

forward(x):
    x = ReLU(Linear(768 → 256))
    x = Dropout(0.3)
    x = Sigmoid(Linear(256 → 45))   # independent probability per class
```

Sigmoid (not softmax) because a report can belong to multiple groups simultaneously.
Loss: binary cross-entropy per class with inverse-frequency class weights.

### Training Data

| Split | Cases | Label |
|-------|-------|-------|
| Cancer (keyword-matched) | ~5,788 unique cases (44 groups) | Multi-hot over matched groups |
| Non-cancer (no match) | ~6,832 unique cases | Uncategorized (all zeros) |
| **Total** | **~12,620 cases** | |

Training is **one-shot**: build multi-hot targets, train the MLP on cached embeddings,
evaluate. Re-run whenever keyword coverage improves — no architectural changes needed.

### How to Train

```bash
ml/.venv/bin/python3 ml/scripts/run_training.py --mode group --device mps
```

Re-train by running this command whenever `keyword_predictions.csv` is updated.

### Inference Flow

1. Load cached `mean_embedding` for the report
2. Forward pass → 45 group probabilities
3. Apply threshold (e.g. 0.3) → predicted group(s); if none → Uncategorized
4. For each predicted group: cosine similarity against terms in that group only → top term + code
5. Output: up to k predictions, one per predicted group above threshold

### Evaluation Results

| Data | Metric | Binary (Phase 11) | GroupClassifier @ 0.3 | GroupClassifier @ 0.8 |
|------|--------|-------------------|-----------------------|-----------------------|
| 1,273 cases | Good+Slight | 20.4% | 14.3% | 23.4% |
| 1,273 cases | CO% | 42.7% | 55.9% | 50.7% |
| 1,273 cases | FP% | ~30% | 28.0% | 8.4% |
| 1,273 cases | FN% | 4.2% | 1.8% | 17.6% |
| **5,788 cases** | **Good+Slight** | **33.1%** | 13.9% | 21.9% |
| **5,788 cases** | **CO%** | **31.8%** | 57.5% | 54.5% |
| **5,788 cases** | **FN%** | **1.3%** | — | 15.6% |

Binary PresenceClassifier is the clear winner at current data volumes. GroupClassifier
overfits (val loss >> train loss) at 5,788 cases across 44 groups (~132 cases/group avg).

### Embedding Experiments (What Was Tried)

#### Priority embedding — FINAL COMMENT first ❌ (2026-03-21)

**Hypothesis:** The mean embedding dilutes the diagnostic signal. FINAL COMMENT is the
pathologist's conclusion — the most group-discriminating column. Using it as the sole 768-dim
input (falling back to HISTOPATHOLOGICAL SUMMARY, then ANCILLARY TESTS if empty) should
give the MLP a cleaner signal without changing the model architecture.

**Result:** Regression. Macro F1 fell from 0.1020 (mean baseline) to 0.0695. The model
reverted to the degenerate "approve everything" pattern: recall ≈ 1.0, precision ≈ 0 for
almost every group. Val loss diverged more severely than the mean baseline (train 0.64,
val 2.4+ at epoch 25). Reverted.

**Why it failed:** The problem is data volume (~132 cases/group), not embedding quality.
The 768-dim input size and total training examples are identical regardless of which column
is selected. Choosing FINAL COMMENT doesn't give the model more data — it gives it different
data, which may actually be noisier because some cases have an empty FINAL COMMENT and fall
back to a different column, making the input distribution inconsistent.

**Conclusion:** Embedding selection is not a lever for the GroupClassifier at current data
volumes. The overfitting ceiling can only be broken by more keyword-confirmed cases.

---

### Expected Trajectory as Keyword Coverage Grows

| Confirmed cases | Expected outcome |
|----------------|-----------------|
| ~5,788 (current) | Overfits — binary wins |
| ~10,000 | GroupClassifier starts generalising — may match binary |
| ~15,000+ | Meaningful CO reduction expected — GroupClassifier should pull ahead |

### Implementation Status

- [x] `ml/model/group_classifier.py` — model definition
- [x] `ml/training/group/build_training_data.py` — multi-hot targets from cache + keyword CSV
- [x] `ml/training/group/train.py` — training script
- [x] `ml/petbert_pipeline/categorization.py` — two-stage group → cosine inference
- [x] `ml/petbert_pipeline/pipeline.py` — `--group-classifier` CLI flag

### Worth Trying: Discriminating-Keyword Term Selection

Instead of cosine similarity within the predicted group, use the group's own taxonomy
labels to auto-derive discriminating keywords, then scan report text for those keywords
to pick the specific term.

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

**Why it's better than pure cosine within-group:**
- Discriminating words are short qualifiers (`benign`, `malignant`, `grade ii`) — much
  less susceptible to false matches than scanning for full tumor names
- Auto-derivable from `labels.csv`: words that appear in *some but not all* terms within
  a group are discriminating candidates; no manual curation needed for most groups
- Interpretable — the reason a term was chosen is visible

**Limitations:**
- Groups whose terms differ substantively (not just by qualifier) still need cosine as
  fallback — e.g. "Vascular tumors" (Hemangioma vs Hemangiosarcoma) are already separated
  at the group level so within-group cosine is less critical there
- Negation is still unhandled, but short qualifiers (`benign`, `grade i`) are less likely
  to appear in negation context than full tumor names

This idea applies equally to Approach 3 (fine-tuned PetBERT) since both approaches share
the same within-group term selection step.

### Advantages

- Explicit group competition — wrong-group assignments directly penalized
- Designed to eliminate the CO floor once data volumes are sufficient
- Faster inference: cosine over ~20 terms instead of ~857
- Simple re-training: one-shot, seconds on cached embeddings

### Disadvantages

- Overfits at current data volume (5,788 cases / 44 groups)
- Uses mean embedding — column-specific signal is lost
- A mis-predicted group guarantees a wrong term (no recovery path)

---

## Approach 3 — Fine-tuned PetBERT

Fine-tunes PetBERT end-to-end as a sequence classifier, removing the frozen-embedding
bottleneck. See `petbert-pipeline.md` for scripts, usage, and known code issues.

### Architecture

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
    cosine similarity within predicted group (base PetBERT for label embeddings)
        ↓
    best term + ICD code
```

PetBERT is fine-tuned end-to-end. The keyword pipeline still generates training labels
from the diagnosis text — at inference, the diagnosis field is not needed. Within-group
term selection still uses cosine similarity against base (unfinetuned) PetBERT embeddings.

### Advantages

- **No frozen-embedding bottleneck**: weights adapt to the task, potentially learning
  clinical vocabulary that generic masked-LM pre-training did not
- A single model replaces PetBERT + GroupClassifier — simpler inference pipeline
- HuggingFace Trainer handles mixed precision, gradient accumulation, and checkpoint selection
- Mirrors the GroupClassifier training strategy but with more expressive capacity

### Disadvantages

- **Computationally expensive**: full transformer fine-tuning requires significantly more
  GPU time and memory than training an MLP head on cached embeddings
- **Risk of catastrophic forgetting**: may degrade PetBERT's veterinary language
  representations if learning rate or epoch count is not carefully managed
- **No cached embeddings**: cannot reuse `embedding_cache.npz` — every run requires a full
  PetBERT forward pass
- Still shares the group-level ceiling: term selection within the predicted group is cosine-based

### Constraints

- Known code issues to resolve before a full training run (see `petbert-pipeline.md`):
  - `WeightedTrainer` constructor argument order is fragile
  - Class weights tensor moved to device during `__init__` before device is resolved
  - No stratified val split in `build_dataset.py`
  - No guard against `--presence-classifier` and `--finetuned-model-path` set simultaneously
- Needs ~10,000+ confirmed cases to fine-tune reliably without memorisation
- MPS (Apple Silicon) fallback must be enabled: `PYTORCH_ENABLE_MPS_FALLBACK=1`

---

## Comparison

| | Binary PresenceClassifier | GroupClassifier | Fine-tuned PetBERT |
|---|---|---|---|
| **Status** | Production best | Implemented, not competitive | WIP |
| **Best result** | 40.0% Good+Slight | 21.9% Good+Slight | Not benchmarked |
| **PetBERT** | Frozen | Frozen | Fine-tuned end-to-end |
| **Training style** | Iterative (CO feedback) | One-shot | One-shot |
| **Data requirement** | Works from ~1,273 cases | Needs ~10,000+ cases | Needs ~10,000+ cases |
| **Training speed** | Fast (MLP on cached embeddings) | Fast (MLP on cached embeddings) | Slow (full transformer) |
| **Inference speed** | Slow (~857 pair scores/report) | Fast (~45 group scores + cosine) | Fast (~45 group scores + cosine) |
| **CO floor** | ~30% | Designed to eliminate it | Designed to eliminate it |
| **Main constraint** | CO floor, keyword data ceiling | Overfits at current data volume | Compute cost, known code issues |

### When to Use Each

- **Now**: Binary PresenceClassifier — best results at current data volume
- **When keyword coverage reaches ~10,000 cases**: Re-train GroupClassifier and benchmark against binary
- **After GroupClassifier proves competitive**: Consider fine-tuning PetBERT, after resolving known issues

---

## Training History

Phase-by-phase results, fix descriptions, and cycle-by-cycle tables are in
[training-log.md](training-log.md).
