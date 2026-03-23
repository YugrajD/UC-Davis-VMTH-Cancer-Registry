# Classifiers

The classifier layer sits on top of frozen PetBERT embeddings and decides which cancer
label(s) a report belongs to. Three approaches exist; the binary PresenceClassifier is
currently the best performer.

| Classifier | Status | Best result |
|---|---|---|
| Binary PresenceClassifier | Production best | 41.9% Good+Slight (Phase 16, `hidden_dim=512`) |
| GroupClassifier | Implemented, not yet competitive | 9.3% Good+Slight end-to-end (MLP macro F1=0.4975 on 17 groups) |
| Fine-tuned PetBERT | WIP — see `petbert-pipeline.md` | Not benchmarked |

> **Training ground truth:** All three approaches derive training labels from the keyword
> pipeline (`keyword_annotation.csv`). The keyword pipeline maps diagnosis field text to
> Vet-ICD-O labels. Cases with no keyword match are treated as non-cancer (Uncategorized).
> In production, no diagnosis text is available — classifiers predict from report text alone.

---

## Evaluation Verdicts

All approaches are evaluated by `ml/evaluation/evaluate.py`:

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
    concat(col1_emb ‖ col2_emb ‖ col3_emb ‖ label_emb)
    → 3072-dim → Linear(3072 → 512) → ReLU → Dropout → Linear(512 → 1)
    ↓
argmax across all label scores
    ↓
predicted label
```

The three column embeddings and the label embedding are concatenated into a 3072-dim
vector and scored by an MLP with a 512-dim hidden layer. Empty columns are zeroed
before concatenation. Labels compete implicitly through argmax.

**Why hidden_dim=512 (Phase 16):** The Phase 13 architecture used hidden_dim=256,
creating a 12:1 compression bottleneck (3072→256). Widening to 512 (6:1 ratio)
recovered +1.9pp (40.0% → 41.9%). The bottleneck was confirmed real.

Two modes are stored in the checkpoint (`col_pair_mode`):

| `col_pair_mode` | Input dim | Notes |
|---|---|---|
| `False` (current) | 4 × 768 = 3072 | Phase 13, 16 — production |
| `True` (experimental) | 2 × 768 = 1536 per pair | Phase 14–15 — regressions, not used |

`n_cols`, `col_pair_mode`, and `hidden_dim` are saved into the checkpoint. Legacy
checkpoints without these keys fall back to safe defaults and load without modification.

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
ml/.venv/Scripts/python.exe ml/scripts/run_training.py \
  --mode binary \
  --label "cycle N" \
  --co-neg-per-case 5 \
  --fp-neg-per-case 10 \
  --embedding-min-sim 0.05 \
  --epochs 25 \
  --recall-weight 0.25 \
  --hidden-dim 512 \
  --device xpu \
  --local-only
```

### Cold Start

Required any time the embedding space changes (PetBERT update, architecture change,
new keyword data). Old bank pairs are anchored to the old cosine space and will add noise.

**Prerequisites:**
1. `ml/output/annotation/keyword/keyword_annotation.csv` must exist. If not, run the keyword pipeline first.
2. `ml/data/report.csv` must exist.

**Files to delete:**
```bash
rm -f ml/data/embedding_cache.npz
rm -f ml/output/training/contrastive/evaluation_co_bank.csv
rm -f ml/model/checkpoints/contrastive/presence_classifier_current.pt
```

**Cold-start c1 command** (same as standard, label it clearly):
```bash
ml/.venv/Scripts/python.exe ml/scripts/run_training.py \
  --mode binary \
  --label "cold-start c1" \
  --co-neg-per-case 5 \
  --fp-neg-per-case 10 \
  --embedding-min-sim 0.05 \
  --epochs 25 \
  --recall-weight 0.25 \
  --hidden-dim 512 \
  --device xpu \
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
| `--hidden-dim` | `512` | Phase 16 standard; widening from 256 resolved 12:1 bottleneck |
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
per-column embeddings (3 × 768) → concat → 2304-dim
    ↓
GroupClassifier MLP
    ↓
sigmoid score per group (17 outputs — groups with ≥100 training cases)
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

**Important:** The GroupClassifier uses `col_emb_concat` (2304-dim), the same
per-column concatenated embeddings used by the binary PresenceClassifier — not
`mean_embedding` (768-dim). Groups with fewer than `--min-group-cases` training
samples are excluded from training and are unreachable at inference.

### Model Architecture

```
GroupClassifier(
    input_dim   = 2304,    # per-column concat (3 × 768)
    hidden_dim  = 512,
    num_classes = 17,      # groups with ≥100 keyword-confirmed cases
    dropout     = 0.05
)

forward(x):
    x = ReLU(Linear(2304 → 512))
    x = Dropout(0.05)
    x = Sigmoid(Linear(512 → 17))   # independent probability per class
```

Sigmoid (not softmax) because a report can belong to multiple groups simultaneously.
Loss: `BCEWithLogitsLoss` with per-class inverse-frequency weights, capped at
`--max-class-weight` to prevent extreme weights from collapsing recall.

### Training Data

| Split | Cases | Label |
|-------|-------|-------|
| Cancer (keyword-matched) | ~5,788 unique cases (44 groups) | Multi-hot over matched groups |
| Non-cancer (no match) | ~6,832 unique cases | Uncategorized (all zeros) |
| **Total** | **~12,620 cases** | |

Training is **one-shot**: build multi-hot targets, train the MLP on cached embeddings,
evaluate. Re-run whenever keyword coverage improves — no architectural changes needed.

### How to Train

> Run from the `ml/` directory so relative paths resolve correctly.

```bash
cd ml
ml/.venv/Scripts/python.exe -m training.group.train \
  --epochs 1700 \
  --lr 5e-5 \
  --hidden-dim 512 \
  --dropout 0.05 \
  --max-class-weight 12 \
  --min-group-cases 100 \
  --device xpu
```

Re-train whenever `keyword_annotation.csv` is updated. The best checkpoint is saved to
`ml/model/checkpoints/group/group_classifier_best.pt` and metadata to `group_classifier_best.meta.json`.

### Inference Flow

1. Load cached `col_emb_concat` (2304-dim) for the report
2. Forward pass → 17 group probabilities
3. Apply threshold → predicted group(s); if none → Uncategorized
4. For each predicted group: cosine similarity against terms in that group only → top term + code
5. Output: up to k predictions, one per predicted group above threshold

### Evaluation Results (Early — 768-dim mean, 2026-03-21)

| Data | Metric | Binary (Phase 11) | GroupClassifier @ 0.3 | GroupClassifier @ 0.8 |
|------|--------|-------------------|-----------------------|-----------------------|
| 1,273 cases | Good+Slight | 20.4% | 14.3% | 23.4% |
| 1,273 cases | CO% | 42.7% | 55.9% | 50.7% |
| 1,273 cases | FP% | ~30% | 28.0% | 8.4% |
| 1,273 cases | FN% | 4.2% | 1.8% | 17.6% |
| **5,788 cases** | **Good+Slight** | **33.1%** | 13.9% | 21.9% |
| **5,788 cases** | **CO%** | **31.8%** | 57.5% | 54.5% |
| **5,788 cases** | **FN%** | **1.3%** | — | 15.6% |

*These results used 768-dim mean_embedding — incorrect; a pipeline bug fed the wrong
tensor to the model. See bug fix documentation below.*

### End-to-End Evaluation (2304-dim col_emb_concat, 2026-03-23)

After fixing the pipeline embedding dimension bug and using the best trained checkpoint
(macro F1=0.4975, epoch 1672), end-to-end pipeline evaluation showed:

| Metric | Binary (Phase 16) | GroupClassifier (best, t=0.3) |
|--------|-------------------|-------------------------------|
| Good+Slight | **41.9%** | 9.3% |
| CO% | 29.6% | ~37% |
| FP% | 27.2% | 33.3% |
| FN% | 1.2% | 16.8% |

The GroupClassifier is a large regression vs binary at current data volumes:
- Only 17 of 43 groups have ≥100 training cases — the other 26 are unreachable at inference
- High FP% (33.3%): the model fires positively on non-cancer cases
- High FN% (16.8%): cases in untrained groups are missed entirely
- Within-group cosine term selection is still weak even when the group is correct

Binary PresenceClassifier remains the clear winner.

### Hyperparameter Tuning Experiment (2026-03-23) ❌ End-to-end regression

**Motivation:** With natural class weights the model predicted everything positive (recall→1.0,
precision→0). The imbalance was severe: Adenoma group had 1,040 positives out of ~12,620 cases
— a 11:1 ratio, other groups as extreme as 119:1. The hypothesis was that (1) capping training
samples per group at 100 and (2) filtering groups with <100 samples would reduce overfitting.

**Per-group cap (--max-group-cases) — REGRESSION:**

Capping positives at 100 per group (row-removal approach, keeping all non-cancer rows) dropped
macro F1. The cap threw away valid training signal without enough regularization benefit. Removing
the cap and using only `--max-class-weight` to control BCE weight magnitude worked better:

| max-group-cases | max-class-weight | Best F1 |
|----------------|-----------------|---------|
| 100 (capped) | auto (up to 119x) | ~0.0 (all positive) |
| 100 (capped) | 20 | 0.245 |
| 0 (no cap) | 20 | 0.358 |
| 0 (no cap) | 12 | 0.409 (best at this LR) |

**Key finding:** `--max-group-cases` removes valid training data. Do not use it. Use
`--max-class-weight` alone to prevent weight explosion.

**Note:** When `--max-group-cases` is applied, class weights must be recalculated from
the capped dataset (not the full dataset) to avoid stale weights. This is implemented in
`train.py` but the approach itself is not recommended.

**Hyperparameter sweep — best config reached through iterative tuning:**

| Parameter | Values tested | Best |
|-----------|--------------|------|
| `--dropout` | 0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.0 | **0.05** |
| `--max-class-weight` | 5, 10, 15, 20, 40, 12, 11, 13 | **12** |
| `--lr` | 1e-3, 5e-4, 3e-4, 2e-4, 1e-4, 7e-5, 5e-5, 3e-5 | **5e-5** |
| `--epochs` | 50, 100, 200, 300, 500, 1000, 1700, 2000 | **1700** |
| `--hidden-dim` | 256, 512 | **512** |

Key insights from the sweep:
- Reducing dropout from 0.3–0.4 to 0.05 gave the largest single gain (~+0.18 F1)
- Lower LR consistently helped; each halving added ~0.01–0.03 F1
- 2-layer architecture (512→256→17) regressed vs single hidden layer (reverted)
- More epochs always helped at LR=5e-5 until epoch ~1672; stopped at 1700

**Best training result:** macro F1 = 0.4975 on 17 groups at epoch 1672.

**End-to-end result:** 9.3% Good+Slight — large regression vs binary 41.9%.
The training F1 does not translate to good end-to-end performance because:
1. Only 17 of 43 groups are trained; the other 26 are unreachable
2. High FP (33.3%): model fires positively on non-cancer reports
3. FN (16.8%): cases in untrained groups are entirely missed
4. Within-group cosine term selection remains weak regardless of group accuracy

**Conclusion:** GroupClassifier MLP requires significantly more labeled data before it
can compete with binary. Do not retry until ~15,000 keyword-confirmed cases exist.

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

### Bugs Discovered and Fixed (2026-03-23)

#### 1. Wrong embedding dimension in pipeline ❌ → ✅ Fixed

**Symptom:** Early evaluation results (the "21.9% Good+Slight @ 0.8" row) were measured
with the wrong embeddings. `pipeline.py` passed `embeddings` (768-dim mean_embedding) to
`group_clf.predict_proba()` but the model was trained on 2304-dim `col_emb_concat`.
The shapes were silently mismatched — PyTorch raised a dimension error at runtime.

**Fix:** Changed line in `pipeline.py` to pass `col_emb_concat` instead of `embeddings`.

#### 2. Double-nesting path bug in train.py ❌ → ✅ Fixed

**Symptom:** Running `python -m training.group.train` from the `ml/` directory with
the default `--out` path `"ml/model/checkpoints/group_classifier_current.pt"` resolved
to `ml/ml/model/checkpoints/...` (double-nested). Similarly for `--training-data`.

**Fix:** Changed both defaults to relative paths without the `ml/` prefix:
- `--out`: `"ml/model/checkpoints/..."` → `"model/checkpoints/..."`
- `--training-data`: `"ml/output/..."` → `"output/..."`

The misplaced checkpoint files were manually moved from `ml/ml/model/checkpoints/` to
`ml/model/checkpoints/` and `ml/ml/` was deleted.

#### 3. Stale class weights after per-group cap ❌ → ✅ Fixed

**Symptom:** When `--max-group-cases=100` was applied, the BCE weights were still
computed from the full (uncapped) dataset. After capping Adenoma from 1,040 → 100
positives, the true neg:pos ratio jumped from 11:1 to 92:1 but the weight stayed at 11,
causing the model to under-penalize false negatives for capped groups.

**Fix:** Class weights are now recalculated from `targets` after the cap is applied in
`train.py`. (The cap itself is not recommended — see tuning section above.)

---

### Expected Trajectory as Keyword Coverage Grows

| Confirmed cases | Expected outcome |
|----------------|-----------------|
| ~5,788 (current) | Overfits — binary wins by large margin (41.9% vs 9.3%) |
| ~10,000 | More groups cross 100-case threshold; GroupClassifier starts generalising |
| ~15,000+ | Meaningful CO reduction expected — GroupClassifier should pull ahead |

**Do not retry GroupClassifier in production until ~15,000 keyword-confirmed cases.**
Hyperparameter tuning has been exhausted at current volume (best F1=0.4975, epoch 1672).

### Implementation Status

- [x] `ml/model/group_classifier.py` — model definition
- [x] `ml/training/group/build_training_data.py` — multi-hot targets from cache + keyword CSV
- [x] `ml/training/group/train.py` — training script
- [x] `ml/production/petbert_pipeline/categorization.py` — two-stage group → cosine inference
- [x] `ml/production/petbert_pipeline/pipeline.py` — `--group-classifier` CLI flag

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
- Only 17 of 43 groups have enough data (≥100 cases) — 26 groups are unreachable
- A mis-predicted group guarantees a wrong term (no recovery path)
- High FP% (33.3%) and FN% (16.8%) at current data volume — large regression vs binary

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
| **Best result** | 41.9% Good+Slight (Phase 16) | 9.3% Good+Slight end-to-end (MLP F1=0.4975 on 17 groups) | Not benchmarked |
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
