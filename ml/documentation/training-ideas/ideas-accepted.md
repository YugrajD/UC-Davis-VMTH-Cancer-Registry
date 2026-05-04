# Ideas Accepted

Ideas that were implemented and improved pipeline performance. Preserved as design records and patterns to build on.

---

## TF-IDF Multi-Column Text Selection

**Status:** Implemented — Phase 24 (2026-04-29). Live in production.

**Problem:** The fallback chain picked the first non-empty column from a priority list
(HISTOPATHOLOGICAL SUMMARY → FINAL COMMENT → COMMENT) and discarded all secondary columns.
Diagnostic signal in secondary columns was lost.

**Sanity-check findings (58,313 cases):**

| Column | Fill rate | Median tokens | Value |
|---|---|---|---|
| HISTOPATHOLOGICAL SUMMARY | 97.1% | 229 | High — the diagnosis |
| COMMENT | 66.4% | 100 | Medium-high — pathologist notes |
| FINAL COMMENT | 31.5% | 137 | High — final conclusions |
| GROSS DESCRIPTION | 98.9% | 128 | Low — excluded (macroscopic appearance) |
| CLINICAL ABSTRACT | 99.4% | 96 | Low — excluded (patient history / signalment) |

- 73.3% of cases: HIST + FINAL COMMENT + COMMENT fit within 512 tokens (return as-is)
- 26.7% of cases: combined overflows → TF-IDF sentence scoring to fit budget

**Implementation:** `TextSelector` in `ml/production/petbert_pipeline/text_selector.py`.
Concatenates HIST + FINAL COMMENT + COMMENT with section markers; compresses via TF-IDF sentence
scoring when combined > 512 tokens. Vectorizer saved at `ml/output/training/tfidf_selector.joblib`.
Applied identically in training and inference to keep backbone embedding space aligned.

**Result:** PresenceClassifier binary G+S improved from ~47% (Phase 23 plateau) to 63.8% at cycle 10.
GroupClassifier macro F1 improved from 0.192 to 0.3136. Overall best: Phase 25 3-stage G+S = 62.6% test.

**Architectural note:** Text selection must be applied identically at training time
(`build_contrastive_dataset.py`) and inference time (`pipeline.py`). Misalignment causes
embedding space drift and silent regression.

---

## Three-Stage Pipeline (CasePresenceClassifier → GroupClassifier → KW Correction)

**Status:** Implemented — Phase 23 Run 10. Current production architecture.

**Problem:** Binary PresenceClassifier generated too many FP predictions (comparing every report
against every label). No mechanism to reject non-cancer cases early.

**Implementation:**
1. **CasePresenceClassifier** (`case_presence_classifier.py`): Binary gate — rejects non-cancer
   cases before group prediction. Trained with `recall_weight=0.85` (Phase 25) to minimize FN.
   Threshold `--case-presence-threshold 0.5` for best G+S vs FP trade-off.
2. **GroupClassifier** (`group_classifier.py`): Multi-label sigmoid over 49 groups.
   Threshold `--group-classifier-threshold 0.90` — critical, CLI default is 0.3 which causes
   prediction explosion.
3. **KW correction** (`behavior_keywords.py`): Narrows candidates within the predicted group
   by ICD-O behavior digit. Rule-based, no retraining needed.

**Result (Phase 25, test set, per-label evaluation, 2026-05-02):**
G+S = 51.8% | CO = 19.3% | FP = 4.7% | FN = 24.1% | Total = 8,744 rows

**Key finding (Phase 25):** CasePresenceClassifier recall_weight matters. Training with
`--case-presence-recall-weight 0.7` caused FN = 25% (gate too aggressive). Retraining with
`--case-presence-recall-weight 0.85` fixed this (FN = 4.5% at gate=0.5, G+S = 62.6%).

---

## GroupClassifier — More Training Epochs (150 → 300)

**Status:** Implemented — Phase 26 (2026-05-04). Live in production.

**Problem:** Phase 24 GroupClassifier best checkpoint was at epoch 120/150 with macro F1=0.3136. Loss was still trending when training stopped — headroom for further convergence.

**Result:** Best macro F1 = **0.4335** at epoch 219/300 (vs 0.3136 — +0.120). Model plateaued after epoch 219; last 80 epochs produced no new best. End-to-end evaluation (gate=0.5, group-t=0.85, all Tier 2 changes):

| Stage | G+S | CO | FP | FN | Total |
|-------|-----|----|----|-----|-------|
| Before (Tier 2b best) | 53.6% | 23.4% | 4.9% | 18.1% | 9,255 |
| **After (new GroupCLF)** | **54.6%** | 22.3% | 5.0% | 18.2% | 9,127 |

+1.0pp G+S, −1.1pp CO. Checkpoint saved to `group_classifier_best.pt`.

**Training config:** `--epochs 300 --lr 5e-5 --max-class-weight 50 --weight-decay 1e-3`. Both `--max-class-weight 50` and `--weight-decay 1e-3` are required — without them BCE pos_weights reach 3,587× and the model predicts all groups for every case.

---

## Argmax Fallback for "Unidentified Cancer" Predictions

**Status:** Implemented — Phase 26 (2026-05-04). Live in production.

**Problem:** Every "Unidentified Cancer" output is a 0% G+S outcome. Many gate-passed cases have a top-group probability in the 0.82–0.89 range — just below the threshold — where GroupClassifier is often correct. Those cases were being discarded instead of predicted.

**Implementation:** Added `fallback_to_argmax: bool = True` to `run_categorization_group` in `categorization.py`. Wired through `ScanConfig.group_classifier_fallback_to_argmax`, `pipeline.py`, and `--no-group-classifier-fallback-to-argmax` CLI flag in `cli.py`.

**Result (on top of group-t=0.85):**

| Config | G+S | CO | FP | FN | Total |
|--------|-----|----|----|-----|-------|
| group-t=0.85, no fallback | 52.3% | 24.0% | 4.9% | 18.8% | 9,335 |
| **+ argmax fallback** | **53.6%** | 23.4% | 4.9% | 18.1% | 9,256 |

+1.3pp G+S, −0.6pp CO, FP flat. "Unidentified Group" predictions eliminated.

---

## Subtype Keyword Discriminators (Meningiomas / Osseous / Gliomas)

**Status:** Implemented — Phase 26 (2026-05-04). Live in production.

**Problem:** GroupClassifier lumps histologic subtypes into broad groups; some groups (Meningiomas, Osseous, Gliomas) have meaningfully different ICD codes per subtype that keyword rules can resolve post-hoc.

**Implementation:** New module `ml/ICD_labels/subtype_keywords.py` with `filter_by_subtype()`. Exported via `ICD_labels/__init__.py`. Applied in `categorization.py` after the behavior filter. Covers Meningiomas (histologic subtype), Osseous (osteo vs chondro), and Gliomas (glioblastoma, astrocytoma, oligodendroglioma, ependymoma, etc.).

**Result (on top of argmax fallback):**

| Config | G+S | CO | FP | FN | Total |
|--------|-----|----|----|-----|-------|
| argmax fallback only | 53.6% | 23.4% | 4.9% | 18.1% | 9,256 |
| **+ subtype keywords** | **53.6%** | 23.4% | 4.9% | 18.1% | 9,255 |

Negligible global change (+0.1pp), but Meningiomas Good% improved from 3% → 10%. Worth keeping for group-level quality.

---

## Lower Group Classifier Threshold (0.90 → 0.85)

**Status:** Implemented — Phase 26 (2026-05-04). Live in production.

**Problem:** At threshold=0.90, many gate-passed cases with top-group probability in the 0.82–0.89 range were emitted as "Unidentified Cancer" (0% G+S outcome) even when GroupClassifier was correct.

**Result:**

| Threshold | G+S | CO | FP | FN | Total |
|-----------|-----|----|----|-----|-------|
| 0.90 (baseline) | 51.8% | 19.3% | 4.7% | 24.1% | 8,744 |
| **0.85** | **52.3%** | 24.0% | 4.9% | 18.8% | 9,335 |
| 0.80 | 50.4% | 29.2% | 5.0% | 15.4% | 10,096 |

**Winner: group-t=0.85.** At 0.80, G+S drops as low-confidence predictions flood in as CO. **Always pass `--group-classifier-threshold 0.85`** — the CLI default is 0.3.

---

## Contrastive Backbone Adaptation (InfoNCE Fine-Tuning)

**Status:** Implemented — Phase 17 (2026-03-24). Foundation of all subsequent phases.

**Problem:** Pre-trained PetBERT embeds report text and label text in different regions of
embedding space — the cosine similarity between a report and its correct label was not reliably
higher than against wrong labels.

**Implementation:** `train_contrastive.py` adapts PetBERT backbone using InfoNCE contrastive loss
on `(report_text, label_text)` positive pairs from LLM annotation. Backbone checkpoint saved to
`ml/output/checkpoints/contrastive/`. All downstream classifiers use this adapted backbone.

**Result:** Binary G+S jumped from ~22% (frozen backbone, Phases 1–16) to ~70% (Phase 17–18).
The adapted backbone is the primary factor enabling all subsequent performance gains.
