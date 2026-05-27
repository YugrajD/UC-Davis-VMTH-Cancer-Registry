# Classifiers

Architecture reference for the three trainable heads plus the keyword-correction stage. The pipeline runs them in order: Stage 1 gate → Stage 2 group → Stage 3a per-group label → Stage 3b keyword filter.

All three trained heads consume the 2304-dim per-row concat-3 embedding (three 768-dim section views stacked) produced by the per-section contrastive PetBERT backbone at `ml/output/checkpoints/contrastive/`. Label texts are embedded once with the same backbone (768-dim each) and reused.

---

## Stage 1 — CasePresenceClassifier

**Role.** Cancer / non-cancer gate. Cases below threshold short-circuit to `Uncategorized` without reaching Stages 2–4. Reduces false positives from non-cancer reports that share vocabulary with cancer reports.

**Files.**
- Module: `ml/model/case_presence_classifier.py`
- Training: `ml/training/binary/build_case_presence_dataset.py` + `ml/training/binary/train_case_presence.py`
- Stage caller: `ml/production/petbert_pipeline/stages/case_presence_classifier.py`
- Checkpoint: `ml/output/checkpoints/case_presence/case_presence_classifier.pt`

**I/O.** Input: 2304-dim concat-3 vector. Output: scalar cancer probability in `[0, 1]`. `emb_dim` is stored in the checkpoint.

**Architecture.**
```
Linear(2304 → 512) → ReLU → Dropout(0.3) → Linear(512 → 1) → sigmoid
```

**Training data.** Built by `build_case_presence_dataset.py`: every train case from `train_cases.txt` becomes a row. Target = 1 if the case has at least one annotation row with a non-empty matched term in `llm_annotation.csv`, else 0.

**Loss.** `BCEWithLogitsLoss` with `pos_weight=1.0`. WeightedRandomSampler balances positives and negatives per batch. Cosine LR over `--epochs` (default 20). Best checkpoint selected by `(1 − recall_weight) × P + recall_weight × R` with `recall_weight=0.7`: prefer missing fewer cancer cases over fewer false positives.

**Threshold.** `--case-presence-threshold` in `run_production.py`. Default 0.5; production uses 0.85.

**Current performance.** Val F1 = 0.942 (P=0.937, R=0.947) at `recall_weight=0.7`.

---

## Stage 2 — GroupClassifier

**Role.** Multi-label group prediction. Decides which of the 25 ICD groups (24 common + the "Uncommon" union) a cancer case belongs to. Groups compete explicitly in the loss, which eliminates the wrong-group floor that a global per-label model suffers from.

**Files.**
- Module: `ml/model/group_classifier.py`
- Training: `ml/training/group/build_training_data.py` + `ml/training/group/train.py`
- Stage caller: `ml/production/petbert_pipeline/stages/group_classifier.py`
- Checkpoint: `ml/output/checkpoints/group/group_classifier_best.pt`

**I/O.** Input: 2304-dim concat-3 vector (`emb_dim` auto-detected from the training NPZ). Output: per-group sigmoid probability (25 outputs). Sigmoid (not softmax) because a case can belong to multiple groups.

**Architecture.**
```
Linear(2304 → 512) → ReLU → Dropout(0.1) → Linear(512 → 25) → sigmoid
```

**Training data.** Built by `build_training_data.py`: every train case from `train_cases.txt` becomes a row whose multi-hot target marks every group that case's annotations cover. Cases with no annotation are all-zeros. Groups with fewer than `--uncommon-threshold` cases (default 200) are merged into a single `Uncommon` output class; the merged group list is written to `ml/output/training/group/uncommon_groups.txt`.

**Loss.** `BCEWithLogitsLoss` with per-group inverse-frequency `pos_weight`. Two guards are required and not optional:
- `--max-class-weight 50` caps per-group `pos_weight` (rare-group weights would otherwise reach >3000×).
- `--weight-decay 1e-3` (Adam) prevents the degenerate "predict every group on every case" solution.

Best checkpoint selected by validation macro F1 across all groups with positives in val. Stratified split keeps per-group case distribution constant.

**Threshold.** `--group-classifier-threshold`. Default 0.3; production uses 0.85. Argmax fallback keeps a concrete prediction for gate-passed cases when no group clears the threshold (disable with `--no-group-classifier-fallback-to-argmax`).

**Tail gate.** After thresholding, `--tail-max-predictions` (default 2) caps the number of group predictions per case, and `--tail-max-group-prob-gap` (default 0.08) drops tail groups whose probability is more than 0.08 below the top group. Calibrate after retraining with `ml/scripts/sweep_tail_gate.py`.

**Current performance.** Val macro F1 = 0.5712 (epoch 258/300, `lr=5e-5`, `dropout=0.1`).

---

## Stage 3a — LabelPresenceClassifier (per group)

**Role.** Within a predicted group, pick the right specific term(s). One head per ICD group plus one shared `uncommon.pt` covering all merged groups. Converts "Slightly off" (right group, wrong term) into "Good" (exact term).

**Files.**
- Module: `ml/model/label_presence_classifier.py`
- Training: `ml/training/label_presence/build_training_pairs.py` + `ml/training/label_presence/train.py`
- Stage caller: `ml/production/petbert_pipeline/stages/label_presence_classifier.py`
- Checkpoints: `ml/output/checkpoints/label_presence/{safe_group_name}.pt`
- Per-LP thresholds: `ml/output/checkpoints/label_presence/lp_thresholds.json`

**I/O.** Input: a 2304-dim case embedding + a 768-dim label embedding. Output: scalar logit; sigmoid → probability.

**Architecture.** Configured with `n_cols=3, col_pair_mode=True, col_combine="learned"`. The 2304-dim case embedding is split into three 768-dim section views. Each section forms a `[section_emb | label_emb]` pair (1536-dim), goes through a shared MLP, and the three per-section logits are combined by a learned `Linear(3 → 1)`:

```
For each of 3 sections:
  pair = concat(section_emb, label_emb)            # (B, 1536)
  per_col_logit = Linear(1536 → 512) → ReLU → Dropout(0.3) → Linear(512 → 1)
Combine: Linear(3 → 1) over the per-section logits → final logit
```

`n_cols`, `col_pair_mode`, `col_combine`, `emb_dim`, and `hidden_dim` are serialized in the checkpoint so the loader rebuilds the right architecture.

**Training data.** One CSV per group, built by `build_training_pairs.py`. For each annotated case in the group:
- 1 positive `(case, matched_label, 1.0)` per annotation row.
- N within-group random-negative rows `(case, other_label_in_same_group, 0.0)` with `--label-presence-negs-per-pos 5`.

The Uncommon group draws labels from the union of all merged groups in `uncommon_groups.txt`.

**Loss.** `BCEWithLogitsLoss` with `pos_weight=1.0`; AdamW with `weight_decay=1e-4`; cosine LR over `--label-presence-epochs` (default 25). `GroupShuffleSplit` produces a case-disjoint train/val split so the same case never appears in both. WeightedRandomSampler balances pos/neg in each batch. Best checkpoint selected by `(1 − recall_weight) × P + recall_weight × R` with `recall_weight=0.5` (i.e., F1).

**Threshold.** Per-LP lookup against `lp_thresholds.json`, with `--label-presence-threshold` (default 0.5) as fallback. The JSON is produced by `ml/scripts/sweep_lp_thresholds.py`, which splits cases 50/50 by case-ID hash, picks the F-beta-maximising threshold for each LP on the sweep half, and reports unbiased metrics on the eval half. Pass `--beta 0.5` to weight precision more (reduces CO), or `--beta 2.0` to weight recall more (reduces FN).

**Current performance.** Per-LP val F1 varies (~0.85–0.95 for common groups; lower for rare groups). End-to-end contribution: per-LP threshold calibration adds roughly +0.08 macro F1 / +0.19 micro F1 over the global default of 0.5.

---

## Stage 3b / Stage 4 — keyword correction

**Role.** Constrain Stage 3a's pool by behavior code, then by group-specific subtype keywords. When Stage 3a is absent (no `.pt` for a group), keyword correction runs against the full in-group label pool and cosine similarity breaks ties.

**Files.**
- Stage caller: `ml/production/petbert_pipeline/stages/keyword_correction.py`
- Behavior-code keywords: `ml/ICD_labels/behavior_keywords.py`
- Subtype keywords: `ml/ICD_labels/subtype_keywords.py`

**Behavior code filter.** ICD-O codes embed a behavior digit after `/`: `/0`=benign, `/1`=borderline, `/2`=in situ, `/3`=malignant, `/6`=metastatic. The report text is scored against weighted vocabulary (e.g. `malignant` → `/3`, `metastatic` → `/6`); the highest-ranked behavior digit narrows the candidate pool to labels carrying that code. When no signal is detected, the pool passes through unchanged.

**Subtype keyword filter.** For 6 groups (Mast cell neoplasms, Blood vessel tumors, Melanocytoma and Melanomas, Meningiomas, Osseous and chondromatous neoplasms, Gliomas), each has an ordered list of `(regex, label_substring)` rules. Each rule's regex is tried against the report text; the first rule that matches AND produces a non-empty pool subset is applied. Pure Python, no model dependencies.

This stage runs unconditionally — no flag to disable it.

---

## Pipeline-level dispatcher

`ml/production/petbert_pipeline/stages/__init__.py::categorize_per_case` walks each case:

1. If Stage 1 rejects (or text is empty) → emit `Uncategorized` / empty.
2. Take the predicted group list (after threshold + argmax-fallback + tail-gate).
3. For each group up to `tail_max_predictions`:
   - If an LP head is loaded for the group: Stage 3a scores → Stage 3b narrows → emit top label.
   - Else: Stage 3b runs on the full in-group pool → cosine vs label embeddings picks the term.
4. Deduplicate winners across groups; optional rerank via `--rerank-stage3`.
5. If nothing was selected after the loop → `Unidentified Cancer` (gate passed) or `Uncategorized` (gate failed).

The dispatcher uses two embedding views: `lp_embeddings` (2304-dim) feeds the LP head, `mean_embeddings` (768-dim) feeds cosine fallback comparisons against 768-dim label embeddings.

---

## End-to-end performance

Verdict definitions (`ml/evaluation/evaluate.py`):

| Verdict | Meaning |
|---|---|
| `good` | Predicted term exactly matches an annotated term for the case |
| `slightly_off` | Predicted group matches an annotated group; term differs |
| `completely_off` | Neither term nor group matches any annotation for this case |
| `false_positive` | Case has no annotation labels; pipeline emitted a cancer prediction |
| `false_negative` | Annotated cancer case with no good or slightly-off prediction |
| `true_negative` | Pipeline emitted `Uncategorized` on a non-cancer case (not counted in totals) |

Current production baseline on the held-out eval-half (4,414 rows):

| Good | Slight | CO | FP | FN | **G+S** |
|---|---|---|---|---|---|
| 46.1% | 16.0% | 14.7% | 2.3% | 20.8% | **62.1%** |

Run end-to-end + per-stage evaluation with `ml/scripts/run_evaluation.py --stage all --test-cases ml/output/splits/test_cases.txt`.
