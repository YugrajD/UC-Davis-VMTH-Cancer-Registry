# Classifiers

The classifier layer sits on top of PetBERT embeddings and decides which cancer
label(s) a report belongs to. Four approaches exist; contrastive fine-tuning +
PresenceClassifier (Approach 3) is currently the production best.

| Classifier | Status | Best result |
|---|---|---|
| Binary PresenceClassifier | Superseded by Approach 3 | 41.9% Good+Slight (Phase 16, `hidden_dim=512`) |
| GroupClassifier | **Beats binary at threshold=0.90 (Phase 23)** | **50.1% Good+Slight (Phase 23, Round 2 backbone, threshold=0.90)** |
| Contrastive fine-tuned PetBERT + PresenceClassifier | **Production best** | **86.5% Good+Slight (Phase 18, c16) — 56.5% Good with group-keyword mode** |
| End-to-end fine-tuned PetBERT | WIP — see `production-pipeline.md` | Not benchmarked |

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
| `false_positive` | Case has no keyword labels but model made a positive prediction |
| `false_negative` | Confirmed cancer case with no good/slightly_off prediction |
| `true_negative` | Model correctly predicted "Uncategorized" for a non-cancer case — excluded from `evaluation.csv` and from all metric totals |

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

**Continuing from an existing bank (standard — adapted backbone):**
```bash
ml/.venv/Scripts/python.exe ml/scripts/run_training.py \
  --mode train-classifier \
  --model ml/output/checkpoints/contrastive \
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
1. `ml/output/annotation/keyword/keyword_annotation.csv` must exist. If not, run annotation first.
2. `ml/data/report.csv` must exist.

**Files to delete:**
```bash
rm -f ml/data/embedding_cache.npz
rm -f ml/output/training/contrastive/evaluation_co_bank.csv
rm -f ml/output/checkpoints/contrastive/presence_classifier_current.pt
```

**Cold-start c1 command** (same as standard, label it clearly):
```bash
ml/.venv/Scripts/python.exe ml/scripts/run_training.py \
  --mode train-classifier \
  --model ml/output/checkpoints/contrastive \
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

**Phase 16 best: `presence_classifier_best_phase16_hd512.pt` (41.9%). Superseded by Phase 17 contrastive fine-tuning (69.0%).**

---

## Approach 2 — GroupClassifier

### Architecture

```
report text
    ↓
PetBERT (contrastive fine-tuned)
    ↓
fallback-chain embedding (768-dim, col_fallback_selected)
    ↓
GroupClassifier MLP
    ↓
sigmoid score per group (42 outputs — groups with ≥10 training cases)
    ↓
threshold → predicted group(s)
```

Instead of scoring each label independently, the GroupClassifier makes one global
decision per report: which cancer group(s) does this report belong to? Groups compete
explicitly in the loss function (multi-label BCE). Term selection is then a simpler
sub-problem — cosine similarity within only the ~20 terms of the predicted group.

**Why this design:** The binary classifier's CO floor comes from labels competing
implicitly through argmax after independent scoring. The GroupClassifier directly
penalises wrong-group assignments in the loss, eliminating CO at the group level.

**Phase 23 architecture:** Uses `col_fallback_selected` (768-dim) from the
contrastive fine-tuned backbone — not the old 3-column concat (2304-dim).

### Model Architecture

```
GroupClassifier(
    input_dim   = 768,     # fallback-chain embedding (col_fallback_selected)
    hidden_dim  = 512,
    num_classes = 42,      # groups with ≥10 LLM-confirmed cases (Phase 23)
    dropout     = 0.3
)

forward(x):
    x = ReLU(Linear(768 → 512))
    x = Dropout(0.3)
    x = Sigmoid(Linear(512 → 42))   # independent probability per class
```

Sigmoid (not softmax) because a report can belong to multiple groups simultaneously.
Loss: `BCEWithLogitsLoss` with per-class inverse-frequency weights, capped at
`--max-class-weight` to prevent extreme weights from collapsing recall.

### Training Data (Phase 23)

| Split | Cases | Label |
|-------|-------|-------|
| Cancer (LLM-annotated, train split) | ~21,853 unique cases (42 groups) | Multi-hot over matched groups |
| Non-cancer (no match, train split) | ~24,799 unique cases | Uncategorized (all zeros) |
| **Total (train split)** | **~46,652 cases** | |

Training is **one-shot**: build multi-hot targets from cached embeddings, train the MLP,
evaluate. Re-run whenever annotation coverage improves.

### How to Train (Phase 23)

```bash
ml/.venv/Scripts/python.exe ml/scripts/run_training.py \
  --mode train-groups \
  --epochs 50 --lr 5e-5 \
  --max-class-weight 50 --weight-decay 1e-3 \
  --device xpu --local-only \
  --train-cases ml/output/splits/train_cases.txt \
  --annotation-csv ml/output/annotation/llm/llm_annotation.csv
```

Critical hyperparameters:
- `--max-class-weight 50` — prevents extreme BCE weights (observed up to 3,587× without this)
- `--weight-decay 1e-3` — L2 regularisation; without it train/val gap is ~0.9 (overfitting)

Best checkpoint saved to `ml/output/checkpoints/group/group_classifier_best.pt`.

### Intended Pipeline Design

Three sequential stages, each with a single distinct responsibility:

1. **CasePresenceClassifier gate** — case-level binary classifier (`mean_report_emb → cancer probability`).
   Cases below `--case-presence-threshold` are rejected → Uncategorized without reaching GroupClassifier.
   Trained with `recall_weight=0.7` to err toward letting uncertain cases through rather than missing cancer.
   **Responsibility: filter non-cancer cases → reduce FP.**

2. **GroupClassifier** — for cases that passed the gate, `report_emb → 42 group probabilities`.
   Apply `--group-classifier-threshold` → predicted group(s); if none → Uncategorized.
   **Responsibility: assign cancer to the correct ICD group → reduce CO.**

3. **KW correction** — for each predicted group, ICD-O behavior keyword matching narrows candidates
   to the matching behavior digit, then cosine similarity selects the best specific term within that pool.
   **Responsibility: pick the right term within the group → convert Slight → Good.**

**Run command:**
```bash
ml/.venv/Scripts/python.exe ml/scripts/run_production.py \
  --case-presence-classifier ml/output/checkpoints/contrastive/case_presence_classifier.pt \
  --case-presence-threshold 0.5 \
  --group-classifier ml/output/checkpoints/group/group_classifier_best.pt \
  --group-classifier-threshold 0.90 \
  --embedding-cache ml/output/training/embedding_cache.npz \
  --device xpu --local-only
```

**Run 9 (2026-04-29, superseded):** Used the label-level `PresenceClassifier` score matrix (N × M)
as the gate instead of `CasePresenceClassifier`. Replaced because `CasePresenceClassifier` is simpler
(no label matrix needed), trained specifically for case-level cancer detection, and has a tunable
recall-precision trade-off via its training objective.

**Run 8 (baseline, no gate):** GroupClassifier ran unconditionally; `PresenceClassifier` was a fallback
for cases where no group cleared the threshold, not a gate.

### Phase 25 Evaluation Results (2026-05-01) — CURRENT

**3-stage pipeline (CasePresenceClassifier rw=0.85 + GroupClassifier Phase 24 + KW).**
Phase 25 fixed FN=25% from Phase 24 by retraining CasePresenceClassifier with recall_weight=0.85.

| Config | G+S | CO | FP | FN | Total |
|--------|-----|----|----|-----|-------|
| Ph24 3-stage rw=0.7 gate=0.5 | 49.1% | 22.1% | 3.7% | 25.0% | 6,748 |
| Ph25 3-stage rw=0.85 gate=0.3 | 61.1% | 26.3% | 9.7% | 3.0% | 7,324 |
| **Ph25 3-stage rw=0.85 gate=0.5** | **62.6%** | **26.2%** | **6.7%** | **4.5%** | **7,084** |

Best config: `--case-presence-threshold 0.5 --group-classifier-threshold 0.90`.

**Note:** PresenceClassifier cycles (binary mode) are NOT used in the 3-stage pipeline.
Binary best (c14): 65.7% train G+S. 3-stage uses GroupClassifier + CasePresenceClassifier only.

---

### Phase 23 Evaluation Results (2026-04-28)

**Backbone:** Round 2 contrastive fine-tuned PetBERT (InfoNCE + hard-negative loss).
**Classifier:** GroupClassifier v2 (max_class_weight=50, weight_decay=1e-3), macro F1=0.1922.
**Comparison baseline:** Binary PresenceClassifier Phase 23 c10.

| Metric | Binary (Phase 23 c10) | GroupClassifier @ 0.88 | **GroupClassifier @ 0.90** | GroupClassifier @ 0.92 |
|--------|-----------------------|------------------------|----------------------------|------------------------|
| Good+Slight | 47.2% | 49.1% | **50.1%** | 50.2% |
| CO% | 28.6% | 29.1% | **25.5%** | 21.0% |
| FP% | 24.2% | 10.0% | **8.9%** | 7.6% |
| FN% | ~0% | 11.8% | **15.5%** | 21.2% |
| Rows | 131,951 (top-k) | 40,927 | **37,760** | 34,745 |

**Recommended threshold: 0.90** — beats binary on G+S (+2.9pp), FP (−15.3pp), and CO
(−3.1pp). FN rises (+15.5pp) because the group gate abstains when uncertain; this is
the primary trade-off.

The GroupClassifier now outperforms binary end-to-end on the Phase 23 training set
(46,652 train cases, LLM annotation).

### Overfitting Failure Without Weight Guards (2026-04-28)

BCE pos_weights on a 46,652-case dataset reach up to 3,587× for rare groups without
capping. Without weight decay the model converges to "predict every group for every
case" (train loss 0.23, val loss 1.13). Both guards are required:

| | Uncapped | Fixed (Phase 23) |
|---|---|---|
| `--max-class-weight` | up to 3,587× | **50** |
| `--weight-decay` | 0.0 | **1e-3** |
| Train loss | 0.23 | 0.247 |
| Val loss | 1.13 | 0.258 |

**Key finding:** `--max-group-cases` (per-group sample cap) removes valid training
signal. Do not use it. Use `--max-class-weight` alone to prevent weight explosion.

### Early Experiments (Phase 16, keyword annotation, 5,788 cases) — Historical

At 5,788 cases with keyword annotation and the old 3-column 2304-dim architecture:
- Best macro F1 = 0.4975 (epoch 1672), but end-to-end G+S = 9.3% — large regression
- Only 17 of 43 groups had ≥100 cases; 26 groups were unreachable at inference
- High FP (33.3%) and FN (16.8%)

These results are superseded by Phase 23 with LLM annotation and the fallback-chain
768-dim architecture.

### Embedding Experiments (Historical — Phase 16)

#### Priority embedding — FINAL COMMENT first ❌ (2026-03-21)

At 5,788 cases with keyword annotation: using FINAL COMMENT as the sole 768-dim input
(falling back to HISTOPATHOLOGICAL SUMMARY, ANCILLARY TESTS) regressed macro F1 from
0.1020 to 0.0695. Val loss diverged (train 0.64, val 2.4+). Reverted.

At Phase 16 volumes, embedding selection was not a lever — the problem was data volume,
not embedding quality.

**Phase 23 resolution:** The fallback-chain 768-dim embedding (col_fallback_selected)
from the contrastive fine-tuned backbone is now the standard input. The contrastive
training step aligns report embeddings with ICD label embeddings, providing better
group discrimination than any column selection on the frozen base model.

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

### Expected Trajectory

| Annotated train cases | Expected outcome |
|----------------------|-----------------|
| ~5,788 (Phase 16, keyword) | Overfits — binary wins by large margin (41.9% vs 9.3%) |
| ~21,853 (Phase 23, LLM) | **GroupClassifier competitive — beats binary at threshold=0.90** |
| ~30,000+ | FN may decrease as more groups have sufficient coverage |

GroupClassifier became competitive at ~21,853 LLM-annotated train cases with the
contrastive fine-tuned backbone. Further annotation coverage will primarily reduce FN
by giving marginal groups more training signal.

### Implementation Status

- [x] `ml/model/group_classifier.py` — GroupClassifier model definition
- [x] `ml/model/case_presence_classifier.py` — CasePresenceClassifier model definition
- [x] `ml/training/group/build_training_data.py` — multi-hot targets from cache + annotation CSV
- [x] `ml/training/group/train.py` — GroupClassifier training script
- [x] `ml/training/binary/build_case_presence_dataset.py` — case-level cancer/no-cancer dataset builder
- [x] `ml/training/binary/train_case_presence.py` — CasePresenceClassifier training script
- [x] `ml/scripts/run_training.py --mode train-case-presence` — orchestrates dataset build + training
- [x] `ml/production/petbert_pipeline/categorization.py` — KW correction within predicted groups
- [x] `ml/production/petbert_pipeline/pipeline.py` — three-stage gate: `--case-presence-classifier` → `--group-classifier` → KW
- [x] `ml/production/petbert_pipeline/cli.py` — `--case-presence-classifier`, `--case-presence-threshold` flags

### Behavior-Keyword Term Selection — Implemented ✅ (2026-03-28)

The "group-keyword" categorization mode is now available via `--categorization-mode group-keyword`
in `run_production.py`. It replaces within-group cosine similarity with ICD-O behavior code
keyword matching to convert "slightly off" predictions (right group, wrong term) into "good".

**Design:**

Stage 1 is identical to the default mode — the PresenceClassifier's top-scoring label
determines the predicted ICD group, and the same threshold decides whether to predict
at all. CO, FP, and FN are therefore unchanged.

Stage 2 only runs when Stage 1 would have made a prediction. Within the predicted group,
the ICD-O behavior digit (the digit after `/` in the code) directly encodes the key
disambiguator: `/0`=benign, `/1`=borderline, `/2`=in situ, `/3`=malignant, `/6`=metastatic.
This is matched to weighted clinical vocabulary in the report text (e.g. `malignant` → `/3`,
`metastatic` → `/6`). Candidates are filtered to the matched behavior digit, then the
highest raw PresenceClassifier score within that pool wins. If no keyword signal is found,
all candidates in the group compete by raw score.

**Keyword module:** `ml/ICD_labels/behavior_keywords.py` — pure Python, no model dependencies.
Uses pre-compiled whole-word regex patterns. Weights: 1.0 = strong signal, 0.5 = contextual.

**Evaluation results (Phase 18 best checkpoint, corrected evaluation, 2026-03-28):**

Both modes write up to 5 top-k rows per case — percentages are directly comparable.

| Metric | Default mode (top-k) | **Group-keyword mode (top-k) — PRODUCTION** |
|---|---|---|
| Good% | 30.6% | **56.5%** |
| Slightly off% | 55.9% | **30.0%** |
| **Good + Slight** | **86.5%** | **86.5%** |
| Completely off% | 9.4% | 9.3% |
| False positive% | 1.9% | **1.9%** |
| False negative% | 2.2% | 2.2% |
| True negatives (excluded from CSV) | 6,329 | 6,330 |

Stage 2 runs per-row: Good% rose +25.9pp with no change to G+S total, CO, FP, or FN.

**Remaining Slight:** Groups where terms differ by something other than behavior code —
Meningiomas (80% Slight — topography), Osseous neoplasms (46%), Gliomas (46%) — need
topographic or histologic keyword discriminators beyond the current behavior-code vocabulary.

**Usage (production):**
```bash
ml/.venv/Scripts/python.exe ml/scripts/run_production.py \
  --categorization-mode group-keyword \
  --out-dir ml/output/production/contrastive_kw \
  --local-only --device xpu
```

This idea applies equally to Approach 2 (GroupClassifier) since both share the same
within-group term selection step.

### Advantages

- Explicit group competition — wrong-group assignments directly penalised
- FP dramatically reduced vs binary (8.9% vs 24.2% at threshold=0.90)
- Faster inference: cosine over ~20 terms instead of ~857
- Simple re-training: one-shot, seconds on cached embeddings
- Now competitive — beats binary G+S at Phase 23 data volume

### Disadvantages

- FN trade-off: cases below threshold output Uncategorized (binary never abstains)
- A mis-predicted group guarantees a wrong term (no recovery path)
- Requires capped class weights and weight decay to avoid degenerate training

---

## Approach 3 — Contrastive Fine-tuned PetBERT (Production Best)

InfoNCE fine-tuning of PetBERT on `(report_text, label_text)` pairs, then retraining
the PresenceClassifier from scratch on the improved embedding space.

- **Result:** 86.5% Good+Slight (Phase 18, c16) — 56.5% Good / 30.0% Slight with `--categorization-mode group-keyword` (production); 30.6% Good / 55.9% Slight with default mode
- **CO floor shattered:** ~30% → ~7% — contrastive training directly fixed wrong-group assignments
- **Scripts:** `ml/training/contrastive/` — see [training-log-finetune.md](training-log/training-log-finetune.md) for full details
- **Backbone checkpoint:** `ml/output/checkpoints/contrastive/`
- **Presence classifier checkpoint:** `ml/output/checkpoints/contrastive/presence_classifier_best.pt`

See [model-training.md](model-training.md#approach-3--contrastive-fine-tuning-infonce) for architecture and run commands.

### Unsupervised Contrastive Training — Considered and Rejected ❌

**What it is:** Unsupervised contrastive learning (SimCSE/SimCLR-style for text) creates positive
pairs from two augmented views of the same input — e.g., passing the same report through PetBERT
twice with different dropout masks — without any label information. The model learns to be invariant
to augmentations.

**Why it was considered:** The dataset has ~12,620 total cases but only ~5,788 with keyword
annotations. Unsupervised contrastive could use all cases, not just the annotated subset, and
requires no labels.

**Why it does not help here:**

1. **It does not solve the alignment problem.** The core difficulty is cross-modal alignment:
   report embeddings and label embeddings start in different regions of the embedding space.
   Supervised InfoNCE fixes this directly — each `(report_text, label_text)` positive pair
   explicitly pulls report embeddings toward their correct ICD label embeddings.
   Unsupervised contrastive only trains report→report similarity; it teaches the model nothing
   about where labels live relative to reports.

2. **PetBERT is already domain-adapted.** Pre-trained on UK veterinary EHRs with masked-LM,
   it already understands veterinary language. Unsupervised contrastive on this dataset would
   provide marginal additional domain benefit on top of what PetBERT already knows.

3. **The bottleneck is data, not representation quality.** The Phase 21 conclusion was explicit:
   *"~70% classifier plateau is a data ceiling. Need more labelled cases."* The Phase 22
   train/test gap (88.9% train → 74.1% test) is a generalisation problem caused by limited
   labeled cases. Changing the training paradigm does not fix a data problem.

**What would actually help:**
- More keyword-confirmed or LLM-annotated cases (direct fix for the data ceiling)
- Data augmentation on existing labeled pairs (synonym substitution, column permutation)
- Using unannotated cases as additional in-batch negatives within the existing supervised
  InfoNCE setup — this keeps cross-modal alignment signal while adding coverage

**Conclusion:** Do not replace supervised InfoNCE with unsupervised contrastive. The supervision
is the mechanism. Revisit if the annotation pipeline cannot scale to ~10,000+ cases.

---

## Approach 4 — End-to-end Fine-tuned PetBERT (WIP)

Fine-tunes PetBERT end-to-end as a sequence classifier, removing the frozen-embedding
bottleneck. See `production-pipeline.md` for the live production path and `training-log/training-log-finetune.md` for scripts, usage, and known code issues.

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

- Known code issues to resolve before a full training run (see `training-log/training-log-finetune.md`):
  - `WeightedTrainer` constructor argument order is fragile
  - Class weights tensor moved to device during `__init__` before device is resolved
  - No stratified val split in `build_dataset.py`
  - No guard against `--presence-classifier` and `--finetuned-model-path` set simultaneously
- Needs ~10,000+ confirmed cases to fine-tune reliably without memorisation
- MPS (Apple Silicon) fallback must be enabled: `PYTORCH_ENABLE_MPS_FALLBACK=1`

---

## Comparison

| | Binary PresenceClassifier | GroupClassifier | Contrastive fine-tuning | End-to-end fine-tuning |
|---|---|---|---|---|
| **Status** | Superseded by Approach 3 | **Competitive (Phase 23)** | **Production best** | WIP, blocked |
| **Best result** | 41.9% G+S (Phase 16) | **50.1% G+S at threshold=0.90 (Phase 23, Round 2 backbone)** | **86.5% G+S, 56.5% Good / 30.0% Slight (Phase 18, c16, group-keyword mode)** | Not benchmarked |
| **PetBERT** | Frozen | Contrastive fine-tuned | Fine-tuned (InfoNCE) | Fine-tuned (classification) |
| **Training style** | Iterative (CO feedback) | One-shot | One-shot fine-tune + iterative PresenceClassifier | One-shot |
| **Data requirement** | Works from ~1,273 cases | Competitive at ~21,853 LLM cases | Works at ~5,788 cases | Needs ~10,000+ cases |
| **Training speed** | Fast (MLP on cached embeddings) | Fast (MLP on cached embeddings) | Slow once (full transformer) + fast iterative | Slow (full transformer) |
| **Inference speed** | Slow (~857 pair scores/report) | Fast (~42 group scores + cosine) | Slow (~857 pair scores/report) | Fast (~42 group scores + cosine) |
| **CO floor** | ~28.6% (Phase 23) | ~25.5% @ t=0.90 | **~7%** — dramatically reduced | Designed to eliminate it |
| **FP rate** | 24.2% (Phase 23) | **8.9% @ t=0.90** | ~1.9% (Phase 18) | Unknown |
| **Main constraint** | CO floor; FP from implicit argmax | FN trade-off at high threshold | LLM annotation ceiling | Compute cost, data volume |

### When to Use Each

- **Three-stage pipeline (current design — CasePresenceClassifier + GroupClassifier + KW):**
  ```
  --case-presence-classifier ml/output/checkpoints/contrastive/case_presence_classifier.pt
  --case-presence-threshold 0.5
  --group-classifier ml/output/checkpoints/group/group_classifier_best.pt
  --group-classifier-threshold 0.90
  ```
  Stage 1 filters non-cancer cases; Stage 2 assigns the ICD group; Stage 3 selects the specific term.
  Train CasePresenceClassifier first with `--mode train-case-presence`.

- **Phase 23 GroupClassifier alone (Run 8 baseline):** `--group-classifier` only, no gate — 50.1% G+S, 8.9% FP, 15.5% FN at threshold=0.90
- **Phase 23 binary**: standard `--mode train-classifier` cycles — 47.2% G+S, higher FP, lower FN
- **Phase 18 contrastive (keyword annotation)**: highest overall G+S (86.5%) but evaluated on a smaller dataset — not directly comparable to Phase 23 LLM ground truth
- **End-to-end fine-tuning**: after three-stage pipeline proves stable and bugs are resolved

---

## Training History

Phase-by-phase results, fix descriptions, and cycle-by-cycle tables are in
[training-log-binary.md](training-log/training-log-binary.md).


