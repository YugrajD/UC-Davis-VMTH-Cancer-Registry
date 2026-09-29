# PetBERT Training Architecture

## Current State

The inference pipeline is complete and produces measurable output:

| Stage | Script / File | Output |
|---|---|---|
| Embed + classify | `ml/petbert_scan/` | `ml/output/report/petbert_scan_predictions.csv` |
| Ground-truth labels | `ml/scripts/keyword_scan.py` | `ml/output/diagnoses/keyword_predictions.csv` |
| Evaluate | `ml/scripts/evaluate_predictions.py` | `ml/output/evaluation/evaluation.csv` + `evaluation_summary.csv` |

Each prediction is scored as one of four verdicts:

| Verdict | Meaning |
|---|---|
| `good` | Predicted term exactly matches a keyword label |
| `slightly_off` | Predicted group matches, but not the exact term |
| `completely_off` | Neither term nor group matches any keyword label |
| `false_positive` | Case had no keyword labels (should be Uncategorized) |

The per-group breakdown in `evaluation_summary.csv` identifies which taxonomy groups the model struggles with most. This is the error signal that drives training.

---

## Training Goal

Teach the pipeline to distinguish between "this report is linguistically similar to a label" and "this report actually has this diagnosis". The target metric is **Good %** from `evaluate_predictions.py`. Secondary metrics are **Slightly off %** (partial credit) and **False positive rate**.

---

## Labeled Dataset

- **Source:** `keyword_predictions.csv` rows where `matched_term` is non-blank
- **Size:** ~1 709 labeled rows across ~1 273 cases (out of ~2 783 total)
- **Label:** `(case_id, matched_term)` pairs — one case can have multiple correct terms
- **Negative examples:** cases where all rows have blank `matched_term` → ground-truth Uncategorized

Class imbalance is severe. Most cases are Uncategorized. Training must not simply optimize for predicting everything as Uncategorized.

---

## Root Cause: Threshold Cannot Fix False Positives

Evaluation shows that false positives score **0.84–0.94** (mean 0.90) — higher than many correct predictions. This makes threshold tuning ineffective: there is no cut-off that separates true matches from false positives because the raw embedding space is not calibrated for presence/absence.

**Why does this happen?**
PetBERT was pre-trained on veterinary clinical notes. Pathology reports discussing an organ, ruling out a diagnosis, or describing benign findings use very similar vocabulary to reports that confirm a cancer diagnosis. The cosine similarity captures linguistic similarity, not diagnostic presence. A report that says "no evidence of lymphoma" may score 0.90 against the lymphoma label embedding for exactly the same reason a true lymphoma case does.

**Implication:** The fix is not a better threshold — it is teaching the model what "belongs to this category" means, not just "is similar in language to this label."

---

## Training Strategy

### Approach: Binary Presence Classifier on Top of Frozen PetBERT

Train a lightweight **classification head** that takes the concatenated embeddings of a report and a taxonomy label as input and predicts a binary label: **present** (this case has this diagnosis) vs **absent** (it does not, including Uncategorized).

This directly addresses the root cause: the model learns the decision boundary between linguistic similarity and diagnostic presence.

**Why not a projection head / contrastive loss?**
- Contrastive loss still operates in similarity space — it would compress the embeddings but not solve the presence/absence distinction
- A binary classifier can explicitly learn the "mentioned but absent" signal from the false positive cases
- ~7 495 false positive examples (confidence 0.84–0.94) are high-quality hard negatives for training

**Why not full fine-tuning?**
- Only ~1 273 labeled cases — too few to safely update all PetBERT weights
- Full fine-tuning risks catastrophic forgetting of PetBERT's veterinary domain knowledge
- A classifier head is fast to train, easy to swap out, and leaves the base model intact

**Architecture:**
`[report_emb (768) | label_emb (768)] → Linear(1536 → 256) → ReLU → Dropout(0.3) → Linear(256 → 1)`
Output is a raw logit; sigmoid gives presence probability. Replaces cosine similarity at inference time.

**Training signal:**
- **Positives:** `(case, label)` pairs where `matched_term` is non-blank (keyword-confirmed diagnoses)
- **Hard negatives:** rows in `evaluation.csv` with `verdict = "false_positive"` — high-similarity predictions for Uncategorized cases; the most important training signal
- **Easy negatives:** for each labeled case, randomly sampled wrong taxonomy terms (`--easy-neg-per-pos`, default 3)

---

## Implementation

### `ml/scripts/build_training_pairs.py`
Assembles the training dataset from three sources and writes `ml/data/training_pairs.csv`.

| Column | Description |
|---|---|
| `case_id` | Patient identifier |
| `merged_text` | All report columns concatenated with `[COLUMN NAME]` prefixes |
| `label_term` | Taxonomy term being scored |
| `label_group` | Taxonomy group for that term |
| `target` | `1` = confirmed diagnosis, `0` = absent |
| `source` | `positive` / `hard_negative` / `easy_negative` |

### `ml/model/presence_classifier.py`
PyTorch `nn.Module` with two linear layers.

- `forward(report_emb, label_emb)` → raw logits (use with `BCEWithLogitsLoss`)
- `score_matrix(report_embeddings, label_embeddings)` → `(N, M)` presence probability matrix, computed efficiently in row-batches to avoid materialising the full N×M×1536 tensor at once

### `ml/scripts/train_classifier.py`
Full training loop:
1. Load `training_pairs.csv`
2. Embed all unique report texts with frozen PetBERT (one pass, results cached in NumPy arrays)
3. Embed all unique label strings with frozen PetBERT
4. Free PetBERT from GPU — all subsequent work is on small cached arrays
5. Stratified train/val split (default 85/15)
6. `WeightedRandomSampler` + `BCEWithLogitsLoss(pos_weight=...)` to handle class imbalance
7. `AdamW` + `CosineAnnealingLR`, 20 epochs by default
8. Print precision / recall / F1 / accuracy per epoch; save best checkpoint by validation F1

Best checkpoint written to `ml/model/checkpoints/presence_classifier_best.pt`.

### `ml/petbert_scan/` (integration)
`pipeline.py` accepts `--presence-classifier <path>`. When set:
- Loads the checkpoint after PetBERT embedding is complete
- Calls `classifier.score_matrix(mean_case_embeddings, label_embeddings)` to produce an `(N, M)` presence probability matrix
- Passes the matrix to `run_categorization()` via the new `score_matrix` parameter, bypassing cosine similarity entirely
- `--embedding-min-sim` continues to serve as the acceptance threshold (0.5 is natural for probabilities)

`categorization.py` accepts the optional `score_matrix: np.ndarray | None` parameter — fully backwards-compatible; existing behaviour is unchanged when `score_matrix` is `None`.

---

## Training Loop (One Iteration)

```
report.csv
    │
    ▼
petbert_scan (inference)
    │
    ▼
petbert_scan_predictions.csv ──► evaluate_predictions.py ◄── keyword_predictions.csv
                                          │
                                          ▼
                                   evaluation.csv
                                   evaluation_summary.csv
                                          │
                                          ▼
                              build_training_pairs.py
                                          │
                                          ▼
                              train_classifier.py
                                          │
                                          ▼
                         presence_classifier_best.pt
                                          │
                                          ▼
               petbert_scan --presence-classifier presence_classifier_best.pt
                                          │
                                    (repeat loop)
```

---

## Success Criteria

| Metric | Baseline | Classifier v1 | Target |
|---|---|---|---|
| Good % | 0.1% (13 / 13 855) | **1.0% (139 / 13 855)** | ≥ 30% |
| Slightly off % | 3.2% (438 / 13 855) | 1.6% (224 / 13 855) | ≥ 20% |
| Completely off % | 42.6% (5 909 / 13 855) | 43.3% (5 997 / 13 855) | ≤ 30% |
| False positive rate | 54.1% (7 495 / 13 855) | 54.1% (7 495 / 13 855) | ≤ 20% |

**Classifier v1 notes:** Taxonomy labels were refined between runs. Good increased 10× (13 → 139) — more label terms now align exactly with keyword ground truth. Slightly off halved, with the reduction shifting into Completely off rather than Good, suggesting the refined terms are more specific and some previous group-level matches no longer hold. False positive count is unchanged (threshold not adjusted; classifier not yet trained).

---

## Next Steps

- [x] Run `evaluate_predictions.py` on the current output to establish baselines
- [x] Write `ml/scripts/build_training_pairs.py`
- [x] Write `ml/model/presence_classifier.py`
- [x] Write `ml/scripts/train_classifier.py`
- [x] Integrate presence classifier into `petbert_scan/pipeline.py` behind `--presence-classifier`
- [ ] Run `build_training_pairs.py` to generate `ml/data/training_pairs.csv`
- [ ] Run `train_classifier.py` to train and save `presence_classifier_best.pt`
- [ ] Re-run pipeline with `--presence-classifier` and re-evaluate
- [ ] Compare `evaluation_summary.csv` before and after to measure improvement
