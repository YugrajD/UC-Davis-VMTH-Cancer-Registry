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

**Implementation:** `TextSelector` in `ml/text_selection/text_selector.py`.
Concatenates HIST + FINAL COMMENT + COMMENT with section markers; compresses via TF-IDF sentence
scoring when combined > 512 tokens. Vectorizer saved at `ml/output/training/tfidf_selector.joblib`.
Applied identically in training and inference to keep backbone embedding space aligned.

**Result:** PresenceClassifier binary G+S improved from ~47% (Phase 23 plateau) to 63.8% at cycle 10.
GroupClassifier macro F1 improved from 0.192 to 0.3136. Overall best: Phase 25 3-stage G+S = 62.6% test.

**Architectural note:** Text selection must be applied identically at training time
(`build_contrastive_dataset.py`) and inference time (`pipeline.py`). Misalignment causes
embedding space drift and silent regression.

---

## Four-Stage Pipeline (CasePresenceClassifier → GroupClassifier → per-group LabelPresenceClassifier → KW Correction)

**Status:** Implemented — Phase 23 Run 10 (initial 3-stage) → Phase 28 (Stage 3a per-group LabelPresenceClassifier added). Current production architecture.

**Problem:** Binary PresenceClassifier generated too many FP predictions (comparing every report
against every label). No mechanism to reject non-cancer cases early. And the single-classifier
argmax over ~857 labels created a hard CO floor.

**Implementation:**
1. **CasePresenceClassifier** (`case_presence_classifier.py`): Binary gate — rejects non-cancer
   cases before group prediction. Trained with `recall_weight=0.85` to minimize FN.
   Production threshold `--case-presence-threshold 0.85` (Phase 25 used 0.5; the concat-3
   stack tightened this to 0.85 alongside the 2304-dim gate retrain — see "2304-dim case-presence gate" below).
2. **GroupClassifier** (`group_classifier.py`): Multi-label sigmoid over 25 post-uncommon groups.
   Production threshold `--group-classifier-threshold 0.85` (CLI default is 0.3 — must override).
3. **Per-group LabelPresenceClassifier** (`label_presence_classifier.py`, Phase 28+): one
   per-group MLP scores labels within the active group. Default `n_cols=3, col_pair_mode=True,
   col_combine="learned"` matches concat-3 inference.
4. **KW correction** (`behavior_keywords.py` + `subtype_keywords.py`): Narrows candidates by
   ICD-O behavior digit and (for 6 groups) subtype keywords. Rule-based, no retraining needed.

**Result (Phase 25, test set, per-label evaluation, 2026-05-02 — historical 3-stage baseline):**
G+S = 51.8% | CO = 19.3% | FP = 4.7% | FN = 24.1% | Total = 8,744 rows.
After Phase 28 + concat-3 + per-section contrastive (2026-05-13), the same 4-stage pipeline reaches **G+S 62.1% on eval-half** — see the concat-3 entries below.

**Key finding (Phase 25):** CasePresenceClassifier recall_weight matters. Training with
`--case-presence-recall-weight 0.7` caused FN = 25% (gate too aggressive). Retraining with
`--case-presence-recall-weight 0.85` fixed this.

---

## GroupClassifier — More Training Epochs (150 → 300)

**Status:** Implemented — Phase 26 (2026-05-04). Live in production.

**Problem:** Phase 24 GroupClassifier best checkpoint was at epoch 120/150 with macro F1=0.3136. Loss was still trending when training stopped — headroom for further convergence.

**Result:** Best macro F1 = **0.4335** at epoch 219/300 (vs 0.3136 — +0.120) on the Phase 26 TF-IDF backbone. Superseded by the concat-3 + per-section contrastive backbone — see the per-section contrastive entry below for the current F1=0.5712 at epoch 258. The hyperparameter recipe (`--epochs 300 --lr 5e-5 --dropout 0.1 --max-class-weight 50 --weight-decay 1e-3`) is unchanged across backbones; only the F1 ceiling moved. Model plateaued after epoch 219 on the old backbone; last 80 epochs produced no new best. End-to-end evaluation (gate=0.5, group-t=0.85, all Tier 2 changes):

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

---

## Per-LP Threshold Calibration

**Status:** Implemented — 2026-05-10. Production default.

**Problem:** A single `label_presence_threshold=0.5` across all per-group LP heads (precision spans 0.18–0.96) is wrong by construction. The optimal sigmoid threshold is group-specific.

**Implementation:** New script `ml/scripts/sweep_lp_thresholds.py` reads the per-(case, label) score CSV from `run_evaluation.py --stage label-presence` and sweeps thresholds 0.05–0.95 in 0.05 steps per LP on a sweep-half (eval-half held out unbiased). Writes the chosen thresholds to `ml/output/checkpoints/label_presence/lp_thresholds.json` (single consolidated map). Production loads it via `--label-presence-thresholds-json` (auto-detected when the file exists); any LP missing from the map falls back to `--label-presence-threshold` (default 0.5).

**Result:** Top-1 exact-term Good +8.7 pp on test eval-half; +0.082 macro F1 / +0.19 micro F1 over the flat-0.5 baseline. 13 of 17 LPs landed at t≥0.85 in the live calibration.

---

## Stage-2 Tail-Gate (`--tail-max-predictions` + `--tail-max-group-prob-gap`)

**Status:** Implemented — 2026-05-11. Production default.

**Problem:** After the argmax-fallback fix, Stage 2 could still emit several below-threshold groups per case (each one a row in the predictions CSV). Tail groups far below the top score were inflating CO because Stage 3a / KW still ran on them.

**Implementation:** New params `tail_max_predictions` (cap predictions per case, default 2) and `tail_max_group_prob_gap` (drop tail groups whose probability is more than this far below the top, default 0.08), wired through `ScanConfig` (`types.py`) and the CLI (`cli.py`). Sweep tool at `ml/scripts/sweep_tail_gate.py` (no retrain).

**Result:** +0.9 pp G+S (CO −4.6 pp, FN +3.3 pp) at K=2, gap=0.08 on test set, n=10,334. Pre-tail-gate baseline (`evaluation_history.csv` row 40, no gate) archived at `ml/output/archive/2026-05-11_pre-tailgate-default/`.

---

## concat-3 Text Representation

**Status:** Implemented — 2026-05-12 in `ml-3-stage/`, promoted to `ml/` 2026-05-13. Production default.

**Problem:** The TF-IDF text selector concatenated HIST + FINAL COMMENT + COMMENT then truncated to 512 tokens — 26.7% of cases overflowed and lost sentence-level signal. Single 768-dim mean-pooled report embedding also flattened section structure that Stage 2 / Stage 3a could exploit.

**Implementation:** Three synthetic columns (`__sec_0__` = HIST, `__sec_1__` = FC+C, `__sec_2__` = ANCILLARY) are built per row; each section is embedded independently against the full 512-token budget; the three 768-dim views are concatenated per row into a 2304-dim `tfidf_selected` cache key. New `--concat-3` flag on `run_production.py` selects this path. Downstream classifiers (Stage 1 gate, GroupClassifier, LabelPresenceClassifier) consume the 2304-dim view; 768-dim cosine fallbacks use the masked-mean (`mean_embeddings`).

**Result:** +1.5 pp G+S over TF-IDF on default PetBERT; +10.9 pp more when stacked with the per-section contrastive backbone (see next entry). Embedding cache keys updated: `col___sec_{0,1,2}__ (N, 768)`, `col_tfidf_selected (N, 2304)`, `mean_embeddings (N, 768)`, plus 768-dim `label_embeddings`.

---

## Per-Section Contrastive Backbone

**Status:** Implemented — 2026-05-12. Production default.

**Problem:** The Round-1 InfoNCE backbone was trained on TF-IDF-selected text — its embedding geometry aligned reports-as-one-blob with labels. Under concat-3 inference, each section is embedded *independently*, so the report-side anchor seen at inference time no longer matches what the backbone was trained against.

**Implementation:** Backbone retrained on per-section `(section_text, label_text)` pairs — each row's HIST, FC+C, and ANC sections each generate a pair with the row's gold label. Pair builder lives in `ml-3-stage/training/contrastive/build_contrastive_dataset.py::build_per_section_contrastive_pairs` (not yet ported to `ml/training/contrastive/`). Pre-retrain backbone archived at `ml-3-stage/output/archive/2026-05-12_pre-concat3-contrastive/`. `run_embed_compare.py` extended to accept `--model`.

**Result:** **G+S 13.1% → 24.0% (+10.9 pp)** on top of concat-3 alone. CO −10.5 pp, FN −3.1 pp. The per-section alignment is what unlocks the bulk of the concat-3 gains.

---

## 2304-dim Case-Presence Gate

**Status:** Implemented — 2026-05-12. Production default.

**Problem:** The legacy 768-dim CasePresenceClassifier was trained on mean-pooled report embeddings. Under concat-3, the natural Stage-1 input is the 2304-dim per-row concat — the gate should see the same view the downstream classifiers see.

**Implementation:** `CasePresenceClassifier(emb_dim=2304)` retrained on the concat-3 cache; `emb_dim` is now saved in the checkpoint and auto-detected at load time. Threshold sweep (0.30–0.95) picked **t=0.85** as the operating point. New training package at `ml/training/case_presence/`, exposed via `run_training.py --mode train-case-presence`. Production CLI: `--case-presence-classifier`, `--case-presence-threshold` (default updated to 0.85 to match the gate's operating point). No-retrain sweep tool at `ml/scripts/sweep_case_gate_threshold.py`.

**Result:** F1=0.942 val. Stacked with per-section contrastive + concat_3 + CO-bank cycle: **G+S 51.8%** on test split (vs 13.1% default-PetBERT baseline, +38.7 pp total). Subsequent integration into the 4-stage pipeline (with per-LP thresholds and tail-gate) yields the current production baseline of G+S 62.1% on eval-half.
