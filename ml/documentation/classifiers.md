# Classifiers

The classifier layer sits on top of PetBERT embeddings and decides which cancer
label(s) a report belongs to. The 4-stage pipeline
(CasePresenceCLF → GroupCLF → per-group LabelPresenceCLF → KW correction) is the **current production design** (2026-05-13, concat-3 stack).

| Classifier | Status | Best result |
|---|---|---|
| CasePresenceClassifier (2304-dim) | Stage 1 gate of 4-stage pipeline | **val F1=0.942** (P=0.937, R=0.947) at recall_weight=0.7. Operating threshold 0.85 in production |
| GroupClassifier (2304-dim → 25 groups) | Stage 2 of 4-stage pipeline | **macro F1=0.5712** (concat-3 + per-section contrastive backbone, epoch 258/300, dropout=0.1, lr=5e-5) — up from 0.4475 on the prior TF-IDF baseline |
| Per-group LabelPresenceClassifier (`n_cols=3, col_pair_mode=True, col_combine="learned"`) | **Stage 3a of 4-stage pipeline (concat-3, 25 LPs)** | **G+S 62.1% on unbiased eval-half** (Good 46.1, Slight 16.0, CO 14.7, FP 2.3, FN 20.8, n=4,414); 62.3% on full test (n=8,835); per-LP threshold calibration adds +0.082 Macro F1 / +0.19 Micro F1 |
| LabelPresenceClassifier (whole-corpus, all-label binary) | Removed during 4-stage refactor; preserved in `training-log/training-log-binary.md` | 65.7% G+S train (Phase 25 c14, contrastive backbone) |
| Prior TF-IDF 4-stage baseline | Preserved at `../ml-tfidf/`; G+S 56.6% on eval-half | superseded 2026-05-13 by the concat-3 stack (+5.5 pp G+S, +8.8 pp Good) |

> **Training ground truth:** Training supervision comes from the LLM annotation pipeline (`llm_annotation.csv`). It uses a three-tier cascade — exact keyword match (with negation masking) → fuzzy token overlap (behavior-code aware) → LLM resolution (group + anatomic-site aware) — to map diagnosis text to Vet-ICD-O labels. In production, no diagnosis text is available — classifiers predict from report text alone.

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

## Approach 1 — LabelPresenceClassifier (all-label binary, removed in Phase 28)

The single all-label binary classifier with iterative CO-bank training was the
production design from Phase 1 through Phase 25 (best: 65.7% G+S train, 5,788
cases). It was replaced in Phase 28 by per-group LabelPresenceClassifier
(see Approach 4 below). Architecture summary: 3 × 768 PetBERT column embeddings
concatenated with each label embedding (3072-dim → 512 → 1), with an iterative
CO bank of completely-off predictions accumulating across cycles to push the
hard-negative floor down.

**Why removed:** the implicit argmax across all ~857 labels created a hard CO
floor (~30%) — labels could only reject individual pairs, never redirect a
wrong-group match. The 4-stage pipeline solves this by gating on group first
(GroupClassifier) and scoring within-group (per-group LabelPresenceClassifier),
which removed the need for a global CO bank entirely.

The full training-cycle history (Phases 1–22) lives in
`training-log/training-log-binary.md`. The training entry point
(`--mode train-classifier`), the `binary/` build/cycle scripts, and
`model/presence_classifier.py` were deleted in Phase 28.

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
sigmoid score per group (25 outputs — post-uncommon-bucket consolidation)
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

**Current architecture (concat-3, 2026-05-13):** Uses `col_concat_3` (2304-dim) from
the per-section-adapted contrastive backbone — three section views (HIST / FC+C /
ANCILLARY) concatenated per row.

### Model Architecture

```
GroupClassifier(
    input_dim   = 2304,    # concat-3 per-row embedding (col_concat_3)
    hidden_dim  = 512,
    num_classes = 25,      # 25 groups (post-uncommon-bucket consolidation)
    dropout     = 0.1
)

forward(x):
    x = ReLU(Linear(2304 → 512))
    x = Dropout(0.1)
    x = Sigmoid(Linear(512 → 25))   # independent probability per class
```

`emb_dim` is parameter-driven and auto-detected from the training NPZ shape.

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

### Intended Pipeline Design (4-stage, Phase 28+)

Four sequential stages, each with a single distinct responsibility:

1. **CasePresenceClassifier gate** — case-level binary classifier (`mean_report_emb → cancer probability`).
   Cases below `--case-presence-threshold` are rejected → Uncategorized without reaching GroupClassifier.
   Trained with `recall_weight=0.85` to err toward letting uncertain cases through rather than missing cancer.
   **Responsibility: filter non-cancer cases → reduce FP.**

2. **GroupClassifier** — for cases that passed the gate, `report_emb → group probabilities` (sigmoid per group).
   Apply `--group-classifier-threshold`; argmax fallback ensures gate-passed cases always receive a concrete group.
   Groups below `--uncommon-threshold` cases are merged into a single "Uncommon" output class.
   **Responsibility: assign cancer to the correct ICD group → reduce CO.**

3a. **Per-group LabelPresenceClassifier (Stage 3a — concat-3, current production)** — for each active group, a learned
   `n_cols=3, col_pair_mode=True, col_combine="learned"` head. The 2304-dim report concat is viewed as three 768-dim
   section embeddings; each section forms a `[section_emb (768) ‖ label_emb (768)] → 1536` pair through a shared
   1536→512→1 MLP; the per-section logits go through a learned `Linear(3 → 1)` combiner to produce one (case, label)
   score. Labels above the per-LP threshold (`lp_thresholds.json`) or the global `--label-presence-threshold` are
   selected; argmax fallback applies otherwise. Multiple labels per group can be selected. Groups without a
   corresponding `.pt` file fall through directly to KW correction.
   **Responsibility: pick the right specific term(s) within the group → convert Slight → Good.**

3b. **KW correction (post-filter)** — within the label pool selected by Stage 3a (or the full group pool when 3a is absent),
   ICD-O behavior keyword matching narrows candidates to the matching behavior digit, then a subtype keyword filter
   (Mast cell, Blood vessel, Melanomas, Meningiomas, Osseous, Gliomas) applies group-specific discriminators
   before cosine similarity selects the final term.

**Run command (4-stage, current production, concat-3):**
```bash
ml/.venv/Scripts/python.exe ml/scripts/run_production.py \
  --csv ml/data/report.csv \
  --model ml/output/checkpoints/contrastive --local-only \
  --embedding-cache ml/output/training/embedding_cache.npz \
  --case-presence-classifier ml/output/checkpoints/case_presence/case_presence_classifier.pt \
  --case-presence-threshold 0.85 \
  --group-classifier ml/output/checkpoints/group/group_classifier_best.pt \
  --group-classifier-threshold 0.85 \
  --label-presence-classifier-dir ml/output/checkpoints/label_presence \
  --label-presence-thresholds-json ml/output/checkpoints/label_presence/lp_thresholds.json \
  --tail-max-predictions 2 --tail-max-group-prob-gap 0.08 \
  --out-dir ml/output/production --device cuda
```

The 2304-dim concat-3 view (`col_concat_3`) is built unconditionally and routed to the
case gate, group classifier, and LP head. Cosine-similarity fallbacks inside
`categorize_per_case` use the 768-dim masked-mean (`mean_embeddings`) so the comparison
dim matches the 768-dim label embeddings.

**Run 9 (2026-04-29, superseded):** Used the label-level `PresenceClassifier` score matrix (N × M)
as the gate instead of `CasePresenceClassifier`. Replaced because `CasePresenceClassifier` is simpler
(no label matrix needed), trained specifically for case-level cancer detection, and has a tunable
recall-precision trade-off via its training objective.

**Run 8 (baseline, no gate):** GroupClassifier ran unconditionally; `PresenceClassifier` was a fallback
for cases where no group cleared the threshold, not a gate.

### Phase 28 Evaluation Results (2026-05-07) — HISTORICAL (25-group setup, pre-concat-3 backbone)

> **Note:** The table below is the original Phase 28 measurement (25 LP files, Phase 27 GroupCLF on the TF-IDF-text-aligned backbone). The current production setup (`ml/` post-2026-05-13 promotion) runs **25 LPs on the concat-3 + per-section contrastive backbone** with per-LP thresholds (`lp_thresholds.json`) and the Stage-2 tail-gate, reaching **G+S 62.1% on eval-half** (Good 46.1, Slight 16.0, CO 14.7, FP 2.3, FN 20.8, n=4,414). See `training-log/training-log-label-presence.md` for the post-promotion section with the canonical reproducibility command. The threshold-sweep curve below is preserved as a historical reference for the legacy TF-IDF stack (now at `ml-tfidf/`).

**4-stage pipeline (historical): CasePresenceClassifier (rw=0.85) → GroupClassifier (Ph27, F1=0.4475) → LabelPresenceClassifier (Stage 3a, 25 LPs) → KW correction.**

`--group-classifier-threshold 0.85`, `--label-presence-threshold` swept 0.5–0.9 on held-out test set.

| lp-threshold | G+S | Good% | Slight% | CO% | FP% | FN% | Total rows |
|---|---|---|---|---|---|---|---|
| Ph27 baseline (3-stage) | 55.3% | 12.9% | 42.4% | 21.6% | 5.0% | 18.2% | 9,127 |
| **0.5** | **57.9%** | 28.8% | 29.1% | 25.3% | 5.7% | 11.1% | 15,100 |
| 0.6 | 57.6% | 30.7% | 26.9% | 24.7% | 5.7% | 12.0% | 14,111 |
| 0.7 | 57.2% | 32.7% | 24.5% | 24.3% | 5.6% | 12.9% | 13,185 |
| 0.8 | 56.6% | 35.2% | 21.4% | 23.8% | 5.5% | 14.1% | 12,105 |
| 0.9 | 55.9% | **38.3%** | 17.6% | **22.9%** | **5.4%** | 15.8% | 10,923 |

**All thresholds beat Phase 27 on G+S. No threshold simultaneously beats Phase 27 on both G+S and CO.**

**Why CO increases:** Lower thresholds emit multiple labels per group (all scoring ≥ threshold become
separate top-k rows). When the GroupClassifier assigns the wrong group, each extra row is CO.
Raising the threshold narrows the pool, reducing multi-row CO inflation at the cost of recall (↑FN).

**Two operating points:**
- **lp-t=0.5** — best G+S (57.9%), lowest FN (11.1%). Best for completeness/recall.
- **lp-t=0.9** — minimum CO regression (+1.3pp over baseline), G+S=55.9% (+0.6pp). Best for coding precision.

**Recommended: `--label-presence-threshold 0.5`** (best overall G+S). Switch to 0.9 if exact-term coding accuracy matters more than completeness.

**Notable per-group results at lp-t=0.5:**
- Mast cell: 90% Good — near-perfect term selection
- Squamous: 68%, Blood vessel: 64%, Paragangliomas: 64%, Neoplasms of histiocytes: 63%
- Weakest: Ductal/lobular (1%), Mature B-cell (1%), Acinar cell (2%) — uncommon groups with limited training signal

Full training and threshold sweep details: [training-log-label-presence.md](training-log/training-log-label-presence.md)

---

### Phase 26 Evaluation Results (2026-05-04)

**3-stage pipeline (CasePresenceClassifier rw=0.85 + GroupClassifier Phase 26 + KW + argmax fallback + subtype KW).**

> **Evaluation methodology change (2026-05-02):** FN now counts each uncovered expected label
> separately (per-label FN), not just "Uncategorized on a cancer case". This inflates FN% for the
> 3-stage pipeline (which makes fewer predictions per case) vs earlier per-case-FN numbers. The
> Phase 25 per-label baseline was 51.8% G+S.

| Config | G+S | CO | FP | FN | Total | Notes |
|--------|-----|----|----|-----|-------|-------|
| Ph25 3-stage gate=0.5 group-t=0.90 (per-label baseline) | 51.8% | 19.3% | 4.7% | 24.1% | 8,744 | Starting point |
| + group-t=0.85 + argmax fallback + subtype KW | 53.6% | 23.4% | 4.9% | 18.1% | 9,255 | Tier 1+2 |
| **+ GroupCLF Phase 26 (F1=0.4335, lr=5e-5, epoch 219)** | **54.6%** | **22.3%** | **5.0%** | **18.2%** | **9,127** | **Tier 3** |

Phase 25 per-case-FN results (old methodology, not directly comparable):

| Config | G+S | CO | FP | FN | Total |
|--------|-----|----|----|-----|-------|
| Ph24 3-stage rw=0.7 gate=0.5 | 49.1% | 22.1% | 3.7% | 25.0% | 6,748 |
| Ph25 3-stage rw=0.85 gate=0.5 | 62.6% | 26.2% | 6.7% | 4.5% | 7,084 |

Best config for the Phase 25/26 3-stage setup above: `--case-presence-threshold 0.5 --group-classifier-threshold 0.85`. Superseded by the 4-stage concat-3 production stack (`--case-presence-threshold 0.85`, see the README Quick Start for the current canonical command).

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
- [x] `ml/production/petbert_pipeline/stages/keyword_correction.py` — KW correction within predicted groups
- [x] `ml/production/petbert_pipeline/pipeline.py` — orchestrates `stages/case_presence_classifier.py` → `stages/group_classifier.py` → `stages/label_presence_classifier.py` → `stages/keyword_correction.py`
- [x] `ml/production/petbert_pipeline/cli.py` — `--case-presence-classifier`, `--case-presence-threshold` flags

### Behavior-Keyword Term Selection — Implemented ✅ (2026-03-28)

ICD-O behavior code keyword matching is applied in **Stage 4 (KW correction)** of
the 4-stage pipeline by default — there is no `--categorization-mode` flag. It runs
on every prediction with no opt-out.

**Design:**

After Stage 3a (per-group LabelPresenceClassifier) selects within-group candidates,
Stage 4 narrows further by ICD-O behavior digit (the digit after `/`):
`/0`=benign, `/1`=borderline, `/2`=in situ, `/3`=malignant, `/6`=metastatic.
This is matched to weighted clinical vocabulary in the report text (e.g. `malignant`
→ `/3`, `metastatic` → `/6`). When a behavior signal is present the candidate pool is
filtered to that digit; when no signal is found, all in-group candidates remain.
Subtype-keyword discriminators (Meningiomas, Osseous, Gliomas, +3 others as of
Phase 27) further refine the pool topographically/histologically.

**Keyword module:** `ml/ICD_labels/behavior_keywords.py` (behavior code) +
`ml/ICD_labels/subtype_keywords.py` (topographic/histologic). Pure Python, no model
dependencies.

**Phase 18 result (legacy single-classifier, 5,788 cases):** 86.5% G+S — 56.5% Good /
30.0% Slight with KW correction vs 30.6% / 55.9% without. KW correction lifted Good%
+25.9pp with no change to G+S, CO, FP, or FN. See `training-log-finetune.md` for the
full Phase 18 evaluation.

This was generalised from Approach 1 to Approach 2 (GroupClassifier) and now runs as
Stage 4 of the production 4-stage pipeline against any in-group candidate set.

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

- **Result:** 86.5% Good+Slight (Phase 18, c16, legacy single-classifier on 5,788 cases) — 56.5% Good / 30.0% Slight with KW correction; 30.6% Good / 55.9% Slight without. Note: Phase 18 (keyword annotation, 5,788 cases) and Phase 28 (LLM annotation, 46,652 cases, 4-stage) are different datasets and architectures — not directly comparable.
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

## Approach 4 — End-to-end Fine-tuned PetBERT (attempted, reverted)

End-to-end fine-tuning of PetBERT as a Stage 2 group classifier was attempted in 2026-05 and reverted. The val macro F1 ceiling lifted to 0.5774 (vs Phase 27 GroupCLF's 0.4475), but end-to-end test G+S landed at 56.4% — a 1.5pp loss vs Phase 28's 57.9%. The F1 win was absorbed by LP coupling (LPs trained against Phase 27 GroupCLF outputs), and closing the gap would have required regenerating the embedding cache and retraining Stages 1 and 3a — a substantial commitment for an uncertain ~+1–3pp ceiling, against ~30× slower inference.

> **Bar today is higher:** the experiment was measured against Phase 28's 57.9% G+S. The current concat-3 + per-section contrastive + 4-stage pipeline reaches **G+S 62.1%** on eval-half (`ml/` post-2026-05-13 promotion). Resurrecting Approach B would need to beat 62.1%, not 57.9%.

See `training-log/training-log-finetune.md` Approach B for the full Phase A/B/sweep/8-epoch results, the cost analysis, and the resurrection path.

---

## Comparison

| | Binary PresenceClassifier | GroupClassifier | Contrastive fine-tuning |
|---|---|---|---|
| **Status** | Superseded by Approach 3 | **Competitive (Phase 23)** | **Production best** |
| **Best result** | 41.9% G+S (Phase 16) | **50.1% G+S at threshold=0.90 (Phase 23, Round 2 backbone)** | **86.5% G+S, 56.5% Good / 30.0% Slight (Phase 18, c16, group-keyword mode)** |
| **PetBERT** | Frozen | Contrastive fine-tuned | Fine-tuned (InfoNCE) |
| **Training style** | Iterative (CO feedback) | One-shot | One-shot fine-tune + iterative PresenceClassifier |
| **Data requirement** | Works from ~1,273 cases | Competitive at ~21,853 LLM cases | Works at ~5,788 cases |
| **Training speed** | Fast (MLP on cached embeddings) | Fast (MLP on cached embeddings) | Slow once (full transformer) + fast iterative |
| **Inference speed** | Slow (~857 pair scores/report) | Fast (~42 group scores + cosine) | Slow (~857 pair scores/report) |
| **CO floor** | ~28.6% (Phase 23) | ~25.5% @ t=0.90 | **~7%** — dramatically reduced |
| **FP rate** | 24.2% (Phase 23) | **8.9% @ t=0.90** | ~1.9% (Phase 18) |
| **Main constraint** | CO floor; FP from implicit argmax | FN trade-off at high threshold | LLM annotation ceiling |

### When to Use Each

- **Four-stage pipeline (current production — CasePresenceCLF → GroupCLF → per-group LabelPresenceCLF → KW correction):**
  ```bash
  ml/.venv/Scripts/python.exe ml/scripts/run_production.py \
    --csv ml/data/report.csv \
    --model ml/output/checkpoints/contrastive --local-only \
    --embedding-cache ml/output/training/embedding_cache.npz \
    --case-presence-classifier ml/output/checkpoints/case_presence/case_presence_classifier.pt \
    --case-presence-threshold 0.85 \
    --group-classifier ml/output/checkpoints/group/group_classifier_best.pt \
    --group-classifier-threshold 0.85 \
    --label-presence-classifier-dir ml/output/checkpoints/label_presence \
    --label-presence-thresholds-json ml/output/checkpoints/label_presence/lp_thresholds.json \
    --tail-max-predictions 2 --tail-max-group-prob-gap 0.08 \
    --out-dir ml/output/production --device cuda
  ```
  Stage 1 (2304-dim gate) filters non-cancer; Stage 2 assigns the ICD group with argmax fallback and tail-gate trimming; Stage 3a (per-group LP head, `n_cols=3, col_pair_mode=True, col_combine="learned"`) picks the within-group term using per-LP calibrated thresholds; Stage 3b (KW correction) applies behavior-code and subtype discriminators.

- **Legacy TF-IDF 4-stage (`ml-tfidf/`):** preserved baseline at G+S 56.6% on eval-half. Separate pipeline directory, not driven by `ml/scripts/run_production.py`.
- **Phase 23 GroupClassifier alone (Run 8 baseline):** `--group-classifier` only, no gate — 50.1% G+S, 8.9% FP, 15.5% FN at threshold=0.90.
- **Phase 23 binary:** legacy iterative LabelPresenceClassifier (since deleted) — 47.2% G+S, higher FP, lower FN.
- **Phase 18 contrastive (keyword annotation):** highest single-classifier G+S (86.5%) but evaluated on a smaller dataset — not directly comparable to LLM-annotation results.
- **End-to-end fine-tuning:** revisit only after the 4-stage pipeline plateaus; current bar is G+S 62.1%.

---

## Training History

Phase-by-phase results, fix descriptions, and cycle-by-cycle tables are in
[training-log-binary.md](training-log/training-log-binary.md).


