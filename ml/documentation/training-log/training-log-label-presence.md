# Per-Group LabelPresenceClassifier Training Log

Stage 3a of the 4-stage pipeline. One binary classifier per ICD group: given a report
embedding and a label embedding, predict whether that label is present.

Input: `[tfidf_selected_emb (768) | label_emb (768)]` → 1536-dim → 512 hidden → 1 logit.
Uses `n_cols=1, col_pair_mode=False` (single concat architecture, not per-column pairs).

Training data built by `training/label_presence/build_training_pairs.py`:
- Positives: annotation-confirmed (case, label) pairs for the group
- Negatives: other labels in the same group (`--label-presence-negs-per-pos 5`)
- Filtered to train cases only

---

## Phase 28 — Initial training (2026-05-07)

**GroupClassifier:** Phase 27 (F1=0.4475, epoch 192, dropout=0.1)
**Backbone:** Round 2 contrastive fine-tuned PetBERT
**Annotation:** `ml/output/annotation/llm/llm_annotation.csv`
**Train split:** `ml/output/splits/train_cases.txt` (46,652 cases)
**Recall weight:** 0.5 (F1-optimised)

```bash
ml/.venv/Scripts/python.exe ml/scripts/run_training.py \
  --mode train-label-presence \
  --model ml/output/checkpoints/contrastive \
  --annotation-csv ml/output/annotation/llm/llm_annotation.csv \
  --train-cases ml/output/splits/train_cases.txt \
  --group-classifier-path ml/output/checkpoints/group/group_classifier_best.pt \
  --label-presence-out-dir ml/output/checkpoints/label_presence \
  --label-presence-epochs 25 \
  --label-presence-recall-weight 0.5 \
  --label-presence-negs-per-pos 5 \
  --device xpu --local-only
```

**Result: 25 trained, 0 skipped.**

### Per-group best validation scores

| Group | Positives | Best Score | Epoch |
|-------|-----------|------------|-------|
| Adenomas and adenocarcinomas | 3,989 | 0.906 | 22 |
| Adnexal and skin appendage neoplasms | 1,384 | 0.841 | 18 |
| Basal cell neoplasms | 413 | 0.810 | 17 |
| Blood vessel tumors | 2,586 | 0.922 | 22 |
| Complex mixed and stromal neoplasms | 678 | 0.849 | 9 |
| Epithelial neoplasms, NOS | 2,260 | 0.856 | 20 |
| Fibromatous neoplasms | 1,143 | 0.916 | 9 |
| Gliomas | 464 | 0.820 | 24 |
| Lipomatous neoplasms | 1,930 | 0.930 | 16 |
| Malignant lymphomas, NOS or diffuse | 1,555 | 0.857 | 14 |
| Mast cell neoplasms | 2,256 | 0.962 | 14 |
| Melanocytoma and Melanomas | 1,443 | 0.890 | 17 |
| Meningiomas | 510 | 0.756 | 12 |
| Myomatous neoplasms | 420 | 0.884 | 16 |
| Neoplasms of histiocytes and accessory lymphoid cells | 618 | 0.912 | 12 |
| Nerve sheath tumors | 323 | 0.829 | 15 |
| Odontogenic tumors | 324 | 0.791 | 14 |
| Osseous and chondromatous neoplasms | 1,710 | 0.845 | 22 |
| Paragangliomas and glomus tumors | 409 | 0.943 | 19 |
| Plasma cell neoplasms | 276 | 0.823 | 19 |
| Soft tissue tumors and sarcomas, NOS | 1,275 | 0.921 | 5 |
| Specialized gonadal neoplasms | 306 | 0.827 | 11 |
| Squamous cell neoplasms | 923 | 0.954 | 10 |
| Transitional cell papillomas and carcinomas | 388 | 0.948 | 14 |
| Uncommon (25 merged groups, 2,216 pos) | 2,216 | 0.896 | 11 |

**Score range:** 0.756 (Meningiomas) – 0.962 (Mast cell). Most groups 0.82–0.95.

**Checkpoints:** `ml/output/checkpoints/label_presence/` (25 `.pt` files, `{safe_name}.pt`)

### Notes

- Mast cell, Squamous, and Transitional cell achieved high scores (>0.94) — these groups
  have clear distinctive vocabulary that the classifier learns quickly.
- Meningiomas is the weakest group (0.756). The group has high within-group label
  overlap (topographic subtypes); the binary task is harder here.
- Gliomas also weak (0.820) — small group (464 pos), few within-group negatives.

### Phase 28 evaluation results (2026-05-07)

**Command run:**
```bash
ml/.venv/Scripts/python.exe ml/scripts/run_production.py \
  --label-presence-classifier-dir ml/output/checkpoints/label_presence \
  --group-classifier-threshold 0.85 --device xpu --local-only

ml/.venv/Scripts/python.exe ml/scripts/run_evaluation.py \
  --test-cases ml/output/splits/test_cases.txt \
  --out-dir ml/output/evaluation/contrastive_test \
  --annotation-csv ml/output/annotation/llm/llm_annotation.csv \
  --label "phase28-label-presence-stage3a"
```

| Metric | Phase 27 baseline | Phase 28 Stage 3a | Delta |
|--------|-------------------|-------------------|-------|
| Good% | 12.9% | **28.8%** | +15.9pp ✅ |
| Slightly off% | 42.4% | 29.1% | -13.3pp |
| **G+S** | **55.3%** | **57.9%** | **+2.6pp** ✅ |
| CO% | 21.6% | 25.3% | +3.7pp ❌ |
| FP% | 5.0% | 5.7% | +0.7pp |
| FN% | 18.2% | **11.1%** | **-7.1pp** ✅ |
| Total rows | 9,127 | 15,100 | +65% |

**Phase 27 baseline = row 15 in evaluation_history.csv (12.9% Good + 42.4% Slight = 55.3% G+S, group-t=0.85).**

**Key findings:**
- Stage 3a successfully converts Slight → Good (+15.9pp Good, -13.3pp Slight) — exactly the intended function.
- G+S beats baseline by +2.6pp and FN drops 7.1pp.
- CO% and total row count both increased significantly. Total rows jumped from 9,127 to 15,100 (+65%).
  This is caused by `--label-presence-threshold 0.5` selecting multiple labels per group (all labels
  scoring ≥0.5 become separate top-k rows). Higher threshold should reduce this.
- The absolute Good count more than tripled (1,177 → 4,353).

**Notable per-group highlights:**
- Mast cell: 90% Good (393/436) — near-perfect term selection
- Squamous: 68% Good, Blood vessel: 64%, Paragangliomas: 64%, Neoplasms of histiocytes: 63%
- Weakest: Ductal/lobular (1% Good), Mature B-cell (1%), Acinar cell (2%) — uncommon groups

### Threshold sweep — `--label-presence-threshold` (2026-05-07)

All runs: `--group-classifier-threshold 0.85`, `--device xpu`, `--local-only`.

| lp-t | G+S | Good% | Slight% | CO% | FP% | FN% | Total rows |
|------|-----|-------|---------|-----|-----|-----|------------|
| Ph27 baseline | 55.3% | 12.9% | 42.4% | 21.6% | 5.0% | 18.2% | 9,127 |
| **0.5** | **57.9%** | 28.8% | 29.1% | 25.3% | 5.7% | 11.1% | 15,100 |
| 0.6 | 57.6% | 30.7% | 26.9% | 24.7% | 5.7% | 12.0% | 14,111 |
| 0.7 | 57.2% | 32.7% | 24.5% | 24.3% | 5.6% | 12.9% | 13,185 |
| 0.8 | 56.6% | 35.2% | 21.4% | 23.8% | 5.5% | 14.1% | 12,105 |
| 0.9 | 55.9% | **38.3%** | 17.6% | **22.9%** | **5.4%** | 15.8% | 10,923 |

**All thresholds beat Phase 27 on G+S. No threshold simultaneously beats Phase 27 on both G+S and CO.**

**Why CO increases:** Lower thresholds select multiple labels per group (all scoring ≥ threshold become
separate top-k rows). When the GroupClassifier assigns the wrong group, every extra row is CO.
Raising the threshold narrows the pool, reducing multi-row CO inflation but missing more cancer cases (↑FN).

**Good% rises monotonically with threshold** — stricter selection keeps only the highest-confidence
labels, improving precision at the cost of recall.

**Two operating points:**
- **lp-t=0.5** — best G+S (57.9%, +2.6pp), lowest FN (11.1%). Best for completeness.
- **lp-t=0.9** — least CO regression (22.9%, +1.3pp over baseline), G+S=55.9% (+0.6pp). Best if coding accuracy > completeness.

**Recommended: lp-t=0.5** (best overall G+S). Use `--label-presence-threshold 0.9` if exact-term
coding precision is the priority.

---

## Phase 28d — Thymic/Myxomatous LP on 27-group structure (2026-05-07)

**Goal:** Properly land Thymic and Myxomatous by retraining only the 3 affected LP classifiers
(`thymic_epithelial_neoplasms.pt`, `myxomatous_neoplasms.pt`, `uncommon.pt`) on the 27-group
GroupClassifier (`group_classifier_current.pt`, F1=0.4330, epoch 273).

**New feature:** `--label-presence-groups` filter added to `run_training.py` so that only
specified groups are retrained, leaving the other 24 LP files untouched.

```bash
ml/.venv/Scripts/python.exe ml/scripts/run_training.py \
  --mode train-label-presence \
  --group-classifier-path ml/output/checkpoints/group/group_classifier_current.pt \
  --label-presence-groups "Thymic epithelial neoplasms|Myxomatous neoplasms|Uncommon" \
  --label-presence-epochs 25 \
  --label-presence-recall-weight 0.5 \
  --train-cases ml/output/splits/train_cases.txt \
  --annotation-csv ml/output/annotation/llm/llm_annotation.csv \
  --model ml/output/checkpoints/contrastive \
  --device xpu --local-only
```

**Results (trained: 2, skipped: 1):**

| Group | Positives | Best Score | Notes |
|-------|-----------|------------|-------|
| Myxomatous neoplasms | 129 | 0.817 | P=0.739 R=0.895 at epoch 4 |
| Thymic epithelial neoplasms | — | skipped | Only 1 label in ICD-O taxonomy — no within-group negatives possible. Falls back to cosine similarity in production. |
| Uncommon (23 groups) | 1,966 | 0.904 | Improved from 0.896 — more coherent bucket after removing Thymic/Myxomatous |

**Test-set evaluation (group-t=0.85, lp-t=0.5):**

| Metric | Phase 28 (25-group, best) | Phase 28c (27-group) | Phase 28d (27-group + aligned LP) |
|--------|--------------------------|----------------------|-----------------------------------|
| Good% | 28.8% | 28.1% | 28.2% |
| Slight% | 29.1% | 28.2% | 28.0% |
| **G+S** | **57.9%** | **56.3%** | **56.2%** |
| CO% | 25.3% | 27.7% | 27.7% |
| FP% | 5.7% | 5.7% | 5.6% |
| FN% | 11.1% | 10.4% | 10.5% |
| Total rows | 15,100 | 15,637 | 15,602 |

**Per-group highlights for new groups:**
- Thymic epithelial neoplasms: 83% Good (20/24) — excellent (only 1 label, no real ambiguity)
- Myxomatous neoplasms: 54% Good (14/26), 35% Slight — solid

**Conclusion:** Phase 28d ≈ Phase 28c (flat). Both trail Phase 28 by ~1.7pp G+S.
Root cause: the 27-group GroupClassifier (F1=0.4330) is weaker than Phase 27/28's
GroupClassifier (F1=0.4475). The LP alignment fix is correct but cannot overcome the
GroupClassifier regression caused by splitting Thymic/Myxomatous out of Uncommon.

**To land Thymic + Myxomatous properly:** the 27-group GroupClassifier would need to match
or exceed F1=0.4475. Until then, **Phase 28 (57.9%) remains the production best.**

> **[Update 2026-05-10]** Superseded. After the Phase 29 backbone cold-start +
> 17-group LP retrain, the operational baseline shifted to **G+S = 59.5%**. See
> the "Current Baseline (2026-05-10) — VERIFIED" section at the end of this
> document. The 27-group Thymic/Myxomatous promotion remains untried on the
> Phase 29 cold-start backbone.

---

## Phase 29 — Full LP retrain on 17-group GroupClassifier (2026-05-07)

**GroupClassifier:** `group_classifier_best.pt` — 16 common groups + Uncommon (33 merged groups)
**Backbone:** Phase 29 backbone — Round 1 InfoNCE on TF-IDF-selected text (`model.safetensors` 2026-05-07 05:18, 3 epochs in-batch negatives, no hard-neg loss). This is the embedding space these LPs were trained on; the prior Round-2/hard-neg backbone (Phase 23 Run 8) was overwritten by the 05/07 retrain.
**Annotation:** `ml/output/annotation/llm/llm_annotation.csv`
**Train split:** `ml/output/splits/train_cases.txt` (46,652 cases)
**Recall weight:** 0.5 (F1-optimised)

```bash
ml/.venv/Scripts/python.exe ml/scripts/run_training.py \
  --mode train-label-presence \
  --model ml/output/checkpoints/contrastive \
  --annotation-csv ml/output/annotation/llm/llm_annotation.csv \
  --train-cases ml/output/splits/train_cases.txt \
  --group-classifier-path ml/output/checkpoints/group/group_classifier_best.pt \
  --label-presence-out-dir ml/output/checkpoints/label_presence \
  --label-presence-epochs 25 \
  --label-presence-recall-weight 0.5 \
  --label-presence-negs-per-pos 5 \
  --device xpu --local-only
```

**Result: 17 trained, 0 skipped.**

### Per-group best validation scores

| Group | Positives | Best Score | Epoch |
|-------|-----------|------------|-------|
| Adenomas and adenocarcinomas | 3,989 | 0.897 | 11 |
| Adnexal and skin appendage neoplasms | 1,384 | 0.838 | 22 |
| Blood vessel tumors | 2,586 | 0.921 | 7 |
| Complex mixed and stromal neoplasms | 678 | 0.849 | 15 |
| Epithelial neoplasms, NOS | 2,260 | 0.849 | 6 |
| Fibromatous neoplasms | 1,143 | 0.926 | 21 |
| Gliomas | 464 | 0.810 | 21 |
| Lipomatous neoplasms | 1,930 | 0.928 | 13 |
| Malignant lymphomas, NOS or diffuse | 1,555 | 0.820 | 19 |
| Mast cell neoplasms | 2,256 | 0.966 | 14 |
| Melanocytoma and Melanomas | 1,443 | 0.854 | 6 |
| Meningiomas | 510 | 0.768 | 12 |
| Neoplasms of histiocytes and accessory lymphoid cells | 618 | 0.919 | 18 |
| Osseous and chondromatous neoplasms | 1,710 | 0.836 | 20 |
| Soft tissue tumors and sarcomas, NOS | 1,275 | 0.918 | 11 |
| Squamous cell neoplasms | 923 | 0.941 | 21 |
| Uncommon (33 merged groups, 5,075 pos) | 5,075 | 0.903 | 18 |

**Score range:** 0.768 (Meningiomas) – 0.966 (Mast cell). Consistent with Phase 28 pattern.

**Checkpoints:** `ml/output/checkpoints/label_presence/` (17 `.pt` files)

---

## Phase 30 — QW1 hard-negative mining (2026-05-10) — NET-NEGATIVE

**Goal:** Replace random within-group negatives in LP training with cosine-similar
hard negatives (`build_training_pairs.py`) at default fraction=0.7
(int(round(5×0.7))=4 hard + 1 random per positive). Expected per ideas-to-try QW1:
+2.5 to +4.0 pp G+S, target macro precision 0.5–0.75.

**Code:** `ml/training/label_presence/build_training_pairs.py` now accepts
`hard_neg_fraction` and a `label_embeddings` dict. Per-group similarity matrix
computed once over in-group label embeddings from `embedding_cache.load_cache()`;
hard rankings cached per positive label. `--label-presence-hard-neg-fraction`
CLI flag added to `run_training.py`.

```bash
ml/.venv/Scripts/python.exe ml/scripts/run_training.py \
  --mode train-label-presence --label-presence-epochs 25 \
  --label-presence-recall-weight 0.5 --label-presence-negs-per-pos 5 \
  --label-presence-hard-neg-fraction 0.7 \
  --group-classifier-path ml/output/checkpoints/group/group_classifier_best.pt \
  --model ml/output/checkpoints/contrastive --device xpu --local-only \
  --train-cases ml/output/splits/train_cases.txt \
  --annotation-csv ml/output/annotation/llm/llm_annotation.csv
```

### Per-group validation scores (Phase 29 → QW1)

| Group | Ph29 | QW1 | Δ |
|---|---|---|---|
| Adenomas and adenocarcinomas | 0.897 | 0.886 | -0.011 |
| Adnexal and skin appendage neoplasms | 0.838 | 0.655 | **-0.183** |
| Blood vessel tumors | 0.921 | 0.900 | -0.021 |
| Complex mixed and stromal neoplasms | 0.849 | 0.637 | **-0.212** |
| Epithelial neoplasms, NOS | 0.849 | 0.825 | -0.024 |
| Fibromatous neoplasms | 0.926 | 0.938 | +0.012 |
| Gliomas | 0.810 | 0.671 | **-0.139** |
| Lipomatous neoplasms | 0.928 | 0.930 | +0.002 |
| Malignant lymphomas, NOS or diffuse | 0.820 | 0.934 | +0.114 |
| Mast cell neoplasms | 0.966 | 0.978 | +0.012 |
| Melanocytoma and Melanomas | 0.854 | 0.882 | +0.028 |
| Meningiomas | 0.768 | 0.821 | +0.053 |
| Histiocytes | 0.919 | 0.914 | -0.005 |
| Osseous and chondromatous neoplasms | 0.836 | 0.830 | -0.006 |
| Soft tissue tumors and sarcomas, NOS | 0.918 | 0.860 | -0.058 |
| Squamous cell neoplasms | 0.941 | 0.904 | -0.037 |
| Uncommon | 0.903 | 0.781 | -0.122 |

Note: Phase 29 val used random negatives (easy); QW1 val uses hard+random,
so direct comparison is not apples-to-apples. Production is the real test.

### LP-only on test set (per-stage isolation, 11,661 cases)

| Threshold | Macro P | Macro R | Macro F1 | Micro P | Micro F1 |
|-----------|---------|---------|----------|---------|----------|
| Ph28 LP @ t=0.5 (baseline) | 0.5368 | 0.9175 | **0.6421** | 0.2856 | 0.4369 |
| QW1 @ t=0.5 | 0.4437 | 0.9164 | 0.5552 | 0.1898 | 0.3145 |
| QW1 @ t=0.7 | 0.5197 | 0.8667 | 0.6141 | 0.2521 | 0.3914 |
| QW1 @ t=0.8 | 0.5650 | 0.8202 | 0.6370 | 0.2942 | 0.4352 |
| QW1 @ t=0.9 | 0.6368 | 0.7409 | **0.6581** | 0.3674 | 0.4964 |

QW1 macro F1 catches up to Phase 28 only at t≈0.8 and exceeds at t=0.9 — but
recall collapses from 0.92 → 0.74 as threshold rises, which hurts end-to-end
G+S (uncovered gold labels become FN, not Slightly).

### Pipeline G+S on test set (16,902–20,231 prediction rows)

| Config | n_rows | Good | Slightly | CO | FP | FN | **G+S** |
|---|---|---|---|---|---|---|---|
| Phase 28 baseline (LP-t=0.5) | 16,902 | 25.7 | 33.8 | 23.4 | 8.0 | 9.2 | **59.5** |
| QW1 hard-neg=0.7, LP-t=0.5 | 20,231 | 19.3 | 39.1 | 24.0 | 9.5 | 8.1 | 58.4 |
| QW1 hard-neg=0.7, LP-t=0.6 | 18,220 | 21.6 | 36.4 | 23.9 | 9.2 | 9.0 | 58.0 |
| QW1 hard-neg=0.7, LP-t=0.7 | 16,251 | 24.1 | 33.5 | 23.3 | 9.0 | 10.2 | 57.6 |

**Conclusion:** QW1 at fraction=0.7 is **net-negative at every tested LP threshold**.
Best operating point is the default LP-t=0.5, but G+S is still 1.1pp below Phase 28.

### Why hard-neg hurt

QW1 preserved recall (0.917 ≈ 0.918) but lost ~9pp macro precision at t=0.5.
Mechanism: training against cosine-similar in-group negatives squeezes
sigmoid scores toward the decision boundary (calibration shift). At t=0.5
the LP fires on far more borderline (case, label) pairs — predicted-label
count rose ~20% (16,902 → 20,231 rows), inflating Slightly-Off at the
expense of Good and dragging precision down.

Per-group precision changes vs the plan's documented Phase 28 numbers:
- Uncommon: 0.11 → 0.07 (worse) — 33 merged groups, label semantics noisy
- Osseous: 0.30 → 0.20 (worse) — sub-types co-occur in reports
- Gliomas: 0.18 → 0.12 (worse) — same pattern
- Adenomas: not in plan's "worst" list, now 0.145 — also degraded

The groups with already-poor precision regressed the most. Groups already
strong (Mast cell, Malignant lymphomas, Histiocytes ≥0.82 P) held.

### Next-step options

1. **Revert + skip QW1** — restore Phase 28 LPs from archive
   (`ml/output/archive/2026-05-10_pre-QW1-hardneg/label_presence`).
2. **Try fraction=0.5** — plan's documented fallback (less aggressive).
3. **Bundle QW1+QW2** — case-disjoint split + dropout/weight-decay tuning +
   early stopping may rescue precision. Plan's original Week-1 recipe.
4. **Stack QW3 (per-group threshold calibration) on QW1** — different
   per-group thresholds may compensate for the calibration shift.

**Archives:**
- `ml/output/archive/2026-05-10_pre-QW1-hardneg/label_presence/` — Phase 28 LPs (restored 2026-05-10)
- `ml/output/archive/2026-05-10_QW1-fraction-0.7/label_presence/` — QW1-only LPs
- `ml/output/archive/2026-05-10_QW1+QW2-bundle/label_presence/` — Phase 30b bundle LPs

---

## Phase 30b — QW1 + QW2 bundle (2026-05-10) — NET-NEGATIVE; REVERTED

**Goal:** Combine QW1 (hard_neg_fraction=0.7) with QW2 (case-disjoint
`GroupShuffleSplit` + `weight_decay=1e-2` + `dropout=0.2` + `patience=5` +
`recall_weight=0.35`) to rescue the calibration shift QW1 introduced.

**Code added:**
- `ml/training/label_presence/train.py` — `GroupShuffleSplit` replaces
  `train_test_split(stratify=…)`; new `weight_decay` and `patience` params;
  early-stopping loop. Defaults reverted to Phase 28 values after this run
  (dropout=0.3, recall_weight=0.5, weight_decay=1e-4, patience=0=disabled).
- `ml/scripts/run_training.py` — `--label-presence-{dropout,weight-decay,patience}`
  CLI flags. Defaults match Phase 28.

```bash
ml/.venv/Scripts/python.exe ml/scripts/run_training.py \
  --mode train-label-presence --label-presence-epochs 25 \
  --label-presence-hard-neg-fraction 0.7 --label-presence-negs-per-pos 5 \
  --label-presence-recall-weight 0.35 \
  --label-presence-dropout 0.2 --label-presence-weight-decay 1e-2 \
  --label-presence-patience 5 \
  --group-classifier-path ml/output/checkpoints/group/group_classifier_best.pt \
  --model ml/output/checkpoints/contrastive --device xpu --local-only \
  --train-cases ml/output/splits/train_cases.txt \
  --annotation-csv ml/output/annotation/llm/llm_annotation.csv
```

Early stopping fired aggressively on most groups (best epoch typically 2–16,
stop within ~5 more epochs), confirming Phase 29's val scores were inflated
by case leakage.

### Pipeline G+S on test set

| Config | n_rows | Good | Slightly | CO | FP | FN | **G+S** |
|---|---|---|---|---|---|---|---|
| Phase 28 baseline (LP-t=0.5) | 16,902 | 25.7 | 33.8 | 23.4 | 8.0 | 9.2 | **59.5** |
| QW1 alone (LP-t=0.5) | 20,231 | 19.3 | 39.1 | 24.0 | 9.5 | 8.1 | 58.4 |
| Bundle (LP-t=0.5) | 20,017 | 19.2 | 39.4 | 23.6 | 9.7 | 8.2 | **58.6** |
| Bundle (LP-t=0.6) | 17,675 | 21.6 | 36.5 | 23.2 | 9.4 | 9.3 | 58.1 |
| Bundle (LP-t=0.7) | 15,643 | 24.2 | 33.2 | 22.8 | 9.2 | 10.6 | 57.4 |
| Bundle (LP-t=0.8) | 13,611 | **27.4** | 28.9 | **22.8** | **8.7** | 12.3 | 56.3 |

**Bundle peak G+S = 58.6%** (LP-t=0.5), only +0.2pp over QW1-alone, still **−0.9pp vs Phase 28**.

### Why QW2 didn't rescue QW1

- Case-disjoint split confirmed the val/prod gap (val scores dropped on the
  same model, exposing the leakage that previously inflated them) but didn't
  improve production precision.
- `recall_weight=0.35` did not move production macro precision
  (bundle 0.437 vs QW1 0.444 — basically equal).
- The calibration shift introduced by training against cosine-similar
  in-group negatives is *fundamental* to hard-neg mining. Regularization
  tightens the model but doesn't undo the sigmoid-score compression.
- At LP-t=0.8, the bundle gets the best Good% and CO% of any config
  tried (27.4% Good, 22.8% CO — both better than Phase 28!), but FN climbs
  to 12.3% pulling G+S below baseline. The bundle makes more confident,
  more-accurate-when-right predictions but misses more borderline labels.

### Action taken

Reverted: bundle LPs → `archive/2026-05-10_QW1+QW2-bundle/`,
Phase 28 LPs restored to live `checkpoints/label_presence/` (byte-identical to
`archive/2026-05-10_pre-QW1-hardneg/label_presence/`). Re-ran production + eval
as sanity check: **G+S = 59.5%** (25.7 Good + 33.8 Slight, 16,902 rows) —
identical to the prior Phase 28 measurement and to `evaluation_history.csv`
row #32 (2026-05-08 stage-all post-prod). CLI defaults flipped back to Phase 28
values so accidental retrains don't reproduce the regression; the new flags
(`--label-presence-hard-neg-fraction`, `--label-presence-dropout`,
`--label-presence-weight-decay`, `--label-presence-patience`) and the
underlying mechanisms remain in the code as opt-in infrastructure.

### Code that stayed in place (reusable infrastructure)

- Hard-negative mining (`build_training_pairs.py`) — cosine-ranked top-k
  negatives, mixed with random at a configurable fraction, with random-only
  fallback when embeddings aren't supplied.
- Case-disjoint `GroupShuffleSplit` in `train.py` — strictly better than
  the previous target-stratified split regardless of negative-mining choice;
  preserved as the new default split strategy.
- Optional early stopping (`patience > 0`).
- Per-LP regularization knobs.

---

## Current Baseline (2026-05-10) — VERIFIED

**Status:** This is the operational baseline that all future LP experiments must beat.

**Setup on disk (live in `ml/output/checkpoints/`):**
- Backbone: `contrastive/` — Round 1 InfoNCE only on TF-IDF text (`model.safetensors` 2026-05-07 05:18; no hard-neg/Round-2 pass since the TF-IDF refit).
- CasePresenceClassifier: `case_presence/case_presence_classifier.pt` (Phase 25 gate, rw=0.85, val=0.939).
- GroupClassifier: `group/group_classifier_best.pt` — 17-group, post-Phase-29 cold-start, best F1=0.494 @ epoch 246.
- LabelPresenceClassifier: 17 `.pt` files in `label_presence/` (dated 2026-05-10 13:19, byte-identical to `archive/2026-05-10_pre-QW1-hardneg/label_presence/`). These are the Phase 29 cold-start LPs restored after the QW1/QW1+QW2 revert; the QW1 narrative refers to them as "Phase 28 LPs."

**Run command (must use these exact thresholds):**
```bash
ml/.venv/Scripts/python.exe ml/scripts/run_production.py \
  --group-classifier-threshold 0.85 \
  --label-presence-threshold 0.5 \
  --device xpu --local-only

ml/.venv/Scripts/python.exe ml/scripts/run_evaluation.py \
  --stage all \
  --test-cases ml/output/splits/test_cases.txt \
  --out-dir ml/output/evaluation/contrastive_test \
  --label "<your description>"
```

**End-to-end (test set, 11,661 cases, 16,902 prediction rows; `evaluation_history.csv` row #33):**

| Good | Slightly | **G+S** | CO | FP | FN |
|---|---|---|---|---|---|
| 25.7% | 33.8% | **59.5%** | 23.4% | 8.0% | 9.2% |

**Per-stage isolation on the same test set:**

| Stage | Metric | Value |
|---|---|---|
| Stage 1 — CasePresence (t=0.50) | P / R / F1 / Acc / AUC | 0.9072 / 0.9511 / **0.9287** / 0.9316 / 0.9807 |
| Stage 2 — Group (t=0.85, 17 groups, cancer-only n=5,461) | Macro F1 / Micro F1 / Top-1 / Top-3 / Exact | **0.7794** / 0.7531 / 0.8270 / 0.9751 / 0.6047 |
| Stage 3 — LabelPresence (t=0.50, 17/17 LPs) | Macro P / R / F1; Micro P / R / F1 | 0.5368 / 0.9175 / **0.6421**; 0.2856 / 0.9297 / 0.4369 |

(Identical to row #32, 2026-05-08 — no drift since the QW1 revert.)

**Where the historical 57.9% came from:** the original Phase 28 measurement (2026-05-07, row #16 of `evaluation_history.csv`) used 25 LP files trained on the *pre-cold-start* backbone with the Phase 27 GroupCLF (F1=0.4475). That table appears in `classifiers.md` as a historical lp-threshold sweep. After the Phase 29 backbone cold-start + 17-group LP retrain, the embedding space and group structure changed; the operational baseline shifted to 59.5%. Direct apples-to-apples comparison between the two numbers is not meaningful — they're different setups.

---

## Per-LP threshold calibration (2026-05-10)

Stage 3 baseline had macro precision 0.537 / recall 0.918 — heavily recall-tilted. 13 of 17 LPs produced overconfident probability distributions, so a single global `lp-t=0.5` was severely miscalibrated. Calibrated per-LP thresholds via grid sweep on a held-out half of the test set.

**Methodology** (`ml/scripts/sweep_lp_thresholds.py`):
1. Test cases split 50/50 by MD5 hash of `case_id` (deterministic, case-disjoint).
2. Sweep half → grid search `t ∈ [0.05, 0.95]` step 0.01 per LP; pick argmax F1.
3. Eval half → measure unbiased P/R/F1 at baseline (t=0.5) vs tuned threshold.

**Per-LP thresholds** (`ml/output/checkpoints/label_presence/lp_thresholds.json`, consumed automatically by `run_production.py`):

| LP | t* | F1 baseline | F1 tuned | ΔF1 |
|---|---:|---:|---:|---:|
| Gliomas | 0.94 | 0.279 | 0.626 | **+0.347** |
| Adenomas and adenocarcinomas | 0.95 | 0.352 | 0.592 | **+0.240** |
| Osseous and chondromatous neoplasms | 0.94 | 0.443 | 0.620 | +0.177 |
| Uncommon | 0.95 | 0.198 | 0.359 | +0.160 |
| Epithelial neoplasms, NOS | 0.92 | 0.601 | 0.750 | +0.149 |
| Blood vessel tumors | 0.93 | 0.747 | 0.844 | +0.097 |
| Complex mixed and stromal neoplasms | 0.88 | 0.596 | 0.690 | +0.094 |
| Adnexal and skin appendage neoplasms | 0.94 | 0.449 | 0.541 | +0.092 |
| Melanocytoma and Melanomas | 0.89 | 0.698 | 0.782 | +0.084 |
| Lipomatous neoplasms | 0.86 | 0.790 | 0.868 | +0.078 |
| Fibromatous neoplasms | 0.91 | 0.804 | 0.860 | +0.056 |
| Squamous cell neoplasms | 0.94 | 0.850 | 0.901 | +0.051 |
| Meningiomas | 0.85 | 0.465 | 0.483 | +0.019 |
| Mast cell neoplasms | 0.69 | 0.948 | 0.954 | +0.006 |
| Neoplasms of histiocytes and accessory lymphoid cells | 0.49 | 0.841 | 0.841 | 0.000 |
| Soft tissue tumors and sarcomas, NOS | 0.89 | 0.923 | 0.921 | -0.002 |
| Malignant lymphomas, NOS or diffuse | 0.84 | 0.783 | 0.760 | -0.023 |

**Stage 3 aggregate on eval half:**

| Metric | Baseline (t=0.5) | Tuned per-LP | Δ |
|---|---:|---:|---:|
| Macro F1 | 0.633 | **0.729** | +0.096 |
| Micro F1 | 0.428 | **0.621** | +0.193 |
| Micro P | 0.278 | **0.507** | +0.229 |
| Micro R | 0.923 | 0.801 | −0.122 |

**End-to-end per-case impact on eval half** (2,683 cancer cases, history rows #36 baseline / #37 tuned):

| Metric | Baseline | Tuned | Δ |
|---|---:|---:|---:|
| **Top-1 Good (exact term)** | 1252 (46.7%) | **1486 (55.4%)** | **+234 (+8.7pp)** |
| Top-1 Slight | 668 (24.9%) | 466 (17.4%) | −202 |
| Mean labels predicted / case | 2.74 | 1.68 | −38.7% |
| Spurious labels (non-annotated) | 4985 | 2331 | **−53.2%** |
| Any-K Good | 1873 (69.8%) | 1841 (68.6%) | −32 (−1.2pp) |
| Abstained on cancer | 131 (4.9%) | 131 (4.9%) | 0 |

**Trade-off:** large precision win (top-1 exact-term accuracy +8.7pp, half as many spurious labels), tiny recall cost (32 cases lost an any-K Good hit).

**Wired:** `LABEL_PRESENCE_THRESHOLDS_JSON` lives in `config.py`; `run_production.py` sets it as a default. If the JSON is missing the pipeline warns and falls back to `--label-presence-threshold`. Recompute on every LP retrain: `ml/.venv/Scripts/python.exe ml/scripts/sweep_lp_thresholds.py` (writes the JSON in place by default).

**Caveat:** thresholds were picked on a held-out half of the *test* set rather than a true validation split carved from `train_cases.txt`. Both halves are case-disjoint and neither was used for LP training, so threshold curves are realistic — but on the next LP retrain, do the sweep on a fresh val split carved from train *before* retraining LPs, so the val cases are out-of-sample for the LP too.

---

## 2026-05-13 — Concat-3 + per-section contrastive + 2304-dim LP promoted to ml/

The `ml-4-stage/` prototype was promoted to be the canonical `ml/` on 2026-05-13. The prior
TF-IDF 768-dim baseline is preserved at `../ml-tfidf/` (renamed from the old `ml/`).

### What changed

- **Text representation**: TF-IDF-selected single 512-token string → **concat-3** (per-section
  embed of HIST / FC+C / ANCILLARY → concatenate per-row to 2304-dim, stored under cache key
  `tfidf_selected`).
- **Backbone**: TF-IDF-trained contrastive → **per-section contrastive** (each section is its
  own positive against the case's label; ~2.7× pairs vs the TF-IDF builder).
- **CasePresenceClassifier**: 768-dim → **2304-dim** (`emb_dim=2304`). Val F1=0.942 at
  `recall_weight=0.7`; operating threshold 0.85.
- **GroupClassifier**: 768→25 → **2304→25**, macro F1 **0.4475 → 0.5712** (+0.124 absolute,
  +28% relative). Phase 27 best config (dropout=0.1, lr=5e-5, max_class_weight=50,
  weight_decay=1e-3, 300 epochs; best at epoch 258).
- **LabelPresenceClassifier**: per-group, `n_cols=1, col_pair_mode=False` →
  **`n_cols=3, col_pair_mode=True, col_combine="learned"`**. Each section's 768-dim view
  forms a `[section_emb | label_emb] → 1536` pair through a shared 1536→512→1 MLP; a learned
  `Linear(3 → 1)` combines per-section logits. Mean val score ≈ 0.88 across 25 LPs (min 0.69
  T-cell lymphomas, max 0.99 Paragangliomas).
- **Per-LP thresholds**: re-calibrated on the sweep-half. Unbiased eval-half Macro F1 0.6522
  → 0.7342 (+0.082); Micro F1 0.5309 → 0.7224 (+0.19). Weakest LPs gain most (Gliomas +0.36,
  Uncommon +0.27).
- **CLI**: new `--concat-3` shortcut on `run_production.py` (sets section_text_cols +
  concat_columns); new `--embed-only` for cache-build runs; new `--label-presence-n-cols /
  --label-presence-col-pair-mode / --label-presence-col-combine` flags on `run_training.py`.
- **`categorize_per_case`** now takes a new `lp_embeddings` parameter so the LP head (2304-dim)
  and cosine-similarity fallbacks (768-dim, must match label embedding dim) each get the
  correct view.

### End-to-end result

| Pipeline | G+S | Good | Slight | CO | FP | FN | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| `../ml-tfidf/` (preserved baseline) | 56.6 | 37.3 | 19.3 | 16.8 | 7.8 | 18.7 | 10,334 |
| **ml/ (current, eval-half unbiased)** | **62.1** | **46.1** | 16.0 | 14.7 | 2.3 | 20.8 | 4,414 |
| ml/ (full test) | 62.3 | 46.6 | 15.7 | 14.4 | 2.3 | 21.0 | 8,835 |

**+5.5 pp G+S, +8.8 pp Good, FP −5.5 pp**. The shift is away from Slight toward Good — the
per-section LP head + concat-3 features pick the exact term within the right group more
often than the 768-dim single-col LP could. Small FN uptick (+2.1 pp) from a stricter gate
threshold (0.85 vs the old 0.5) — sweep at 0.75/0.80 to recover recall if needed.

### Reproducibility command (canonical)

```bash
ml/.venv/Scripts/python.exe ml/scripts/run_production.py \
  --csv ml/data/report.csv \
  --concat-3 \
  --model ml/output/checkpoints/contrastive --local-only \
  --embedding-cache ml/output/training/embedding_cache.npz \
  --case-presence-classifier ml/output/checkpoints/case_presence/case_presence_classifier.pt \
  --case-presence-threshold 0.85 \
  --group-classifier ml/output/checkpoints/group/group_classifier_best.pt \
  --group-classifier-threshold 0.85 \
  --label-presence-classifier-dir ml/output/checkpoints/label_presence \
  --label-presence-thresholds-json ml/output/checkpoints/label_presence/lp_thresholds.json \
  --tail-max-predictions 2 --tail-max-group-prob-gap 0.08 \
  --out-dir ml/output/production --device xpu
```

### Caveat on memory citations

Memory entries written before 2026-05-13 that reference `ml/output/archive/<date>/` paths
now live at `ml-tfidf/output/archive/<date>/`. The transformation is mechanical; update
lazily when next encountered.
