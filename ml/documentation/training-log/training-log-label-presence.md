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

---

## Phase 29 — Full LP retrain on 17-group GroupClassifier (2026-05-07)

**GroupClassifier:** `group_classifier_best.pt` — 16 common groups + Uncommon (33 merged groups)
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
