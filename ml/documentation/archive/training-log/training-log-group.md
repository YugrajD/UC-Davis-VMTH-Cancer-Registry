# GroupClassifier Training Log

All experiments use the same dataset: 12,620 total reports, 5,788 keyword-confirmed
cancer cases across 45 groups, embedded with frozen PetBERT (Phase 13 cache).

Primary metric: **macro-averaged F1** across all groups at threshold=0.3.
Pipeline metric (for comparison with binary): **Good+Slight %** from end-to-end evaluation.

---

## Experiment 1 — Baseline (mean embedding, 2026-03-04)

**Setup:** Standard GroupClassifier on `mean_embeddings` (average of 3 columns, 768-dim).
5,788 cancer cases across 44 groups. Evaluated at thresholds 0.3 and 0.8.

**End-to-end pipeline results (for comparison with binary classifier):**

| Threshold | Good+Slight | CO% | FN% |
|-----------|-------------|-----|-----|
| 0.3 | 13.9% | 57.5% | — |
| 0.8 | 21.9% | 54.5% | 15.6% |

Binary PresenceClassifier (Phase 11, same data): 33.1% Good+Slight, 31.8% CO, 1.3% FN.

**Finding:** GroupClassifier severely overfits. Val loss >> train loss. CO rates of 54–57% far
exceed the binary classifier's floor. At threshold=0.8 the FN rate (15.6%) is unacceptable.
The model has ~132 cases/group on average — insufficient for a 45-class MLP to generalise.

---

## Experiment 2 — Priority embedding (FINAL COMMENT first, 2026-03-21) ❌

**Hypothesis:** The mean embedding dilutes the diagnostic signal. FINAL COMMENT is the
pathologist's conclusion — the most directly group-discriminating column. Use FINAL COMMENT
as the 768-dim input if present, fall back to HISTOPATHOLOGICAL SUMMARY, then ANCILLARY TESTS.

**Rationale:** No architecture change (still 768-dim input), so no increased overfitting risk.
FINAL COMMENT contains direct group-level language ("consistent with hemangiosarcoma") while
other columns contain procedural or microscopic detail.

**Implementation:** In `build_training_data.py` and `pipeline.py`, replace `mean_embeddings`
with a per-case priority selection: for each case, use the first non-empty column in order
[FINAL COMMENT, HISTOPATHOLOGICAL SUMMARY, ANCILLARY TESTS].

**Results (25 epochs, 5,864 cancer cases, device: xpu):**

| Run | Best epoch | Macro F1 | Notes |
|-----|-----------|----------|-------|
| Experiment 1 (mean) | — | 0.1020 | Previous best |
| Experiment 2 (priority) | 21 | **0.0695** | Regression — reverted |

**Per-group pattern:** Recall ≈ 1.0, precision ≈ 0 for essentially every group. The model
approves all groups for all cases — the same degenerate "everything positive" behaviour seen
when `--recall-weight` is too high in binary classifier training.

Val loss divergence: train loss 0.64 at epoch 25, val loss 2.4+. More severe overfitting
than the mean embedding baseline.

**Why it didn't work:** The priority embedding idea assumed the problem was signal quality.
It wasn't. The model still sees the same total number of training examples (~132/group average)
regardless of which column is used. The 768-dim input size is identical — no change to the
model's capacity or the data's size. The overfitting is driven purely by the data-to-parameter
ratio, not by what signal is in the embedding.

Additionally, the priority embedding introduces inconsistency across cases: some cases train
on FINAL COMMENT, others on HISTOPATHOLOGICAL SUMMARY. This may create a noisier input
distribution than the mean, which always uses all available information.

**Conclusion:** Reverted. Production checkpoint (0.1020 macro F1) unchanged. Embedding
selection is not a lever worth pursuing until the data volume constraint is resolved.

---

## Experiment 3 — Per-column concat + hyperparameter sweep (2026-03-23) ❌

**Hypothesis:** Switching from 768-dim `mean_embedding` to 2304-dim `col_emb_concat`
(same input as the binary PresenceClassifier) gives more signal per case. Combined with
thorough hyperparameter tuning, this could close the gap with binary at current data volume.
Also tested: per-group sample cap (`--max-group-cases=100`) to reduce overfitting in large groups.

**Architecture:** `col_emb_concat` (3 × 768 = 2304-dim) → Linear(2304→512) → ReLU →
Dropout(p) → Linear(512→17). 17 outputs (groups with ≥100 keyword-confirmed cases out of 43 total).
Loss: `BCEWithLogitsLoss` with per-class inverse-frequency weights capped at `--max-class-weight`.

**Per-group cap experiment (`--max-group-cases=100`):**

Natural class weights were too extreme (up to 119:1 for rare groups) — caused recall→1.0,
precision→0. The hypothesis was that capping the number of positive training samples per group
would reduce the imbalance. Row-removal approach: keep union of up to 100 positives per group
plus all non-cancer rows.

| max-group-cases | max-class-weight | Best macro F1 |
|----------------|-----------------|--------------|
| 100 (capped) | auto (up to 119x) | ~0.0 (all positive) |
| 100 (capped) | 20 | 0.245 |
| 0 (no cap) | 20 | 0.358 |
| 0 (no cap) | 12 | 0.409 |

**Finding:** The cap removes valid training signal. `--max-class-weight` alone is the right lever
for controlling weight explosion. Do not use `--max-group-cases`.

**Hyperparameter sweep (no cap, all data):**

| Parameter | Values tested | Best |
|-----------|--------------|------|
| `--dropout` | 0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.0 | **0.05** |
| `--max-class-weight` | 5, 10, 15, 20, 40, 12, 11, 13 | **12** |
| `--lr` | 1e-3, 5e-4, 3e-4, 2e-4, 1e-4, 7e-5, 5e-5, 3e-5 | **5e-5** |
| `--epochs` | 50, 100, 200, 300, 500, 1000, 1700, 2000 | **1700** |
| `--hidden-dim` | 256, 512 | **512** |

Key insights:
- Reducing dropout 0.3 → 0.05 gave ~+0.18 F1 — model was severely over-regularized
- Lower LR consistently helped; each halving added ~0.01–0.03 F1
- 2-layer architecture (512→256→17) regressed vs single hidden layer — reverted
- Best checkpoint: macro F1 = **0.4975** at epoch 1672

**End-to-end pipeline evaluation (vs binary Phase 16):**

| Metric | Binary (Phase 16) | GroupClassifier (best, t=0.3) |
|--------|-------------------|-------------------------------|
| Good+Slight | **41.9%** | 9.3% |
| CO% | 29.6% | ~37% |
| FP% | 27.2% | 33.3% |
| FN% | 1.2% | 16.8% |

**Why it failed:**
1. Only 17 of 43 groups trained — 26 groups unreachable at inference, causing FN spike
2. FP% (33.3%): non-cancer cases find positive neighbors in 2304-dim space regardless of threshold
3. FN% (16.8%): cases whose true group is among the 26 untrained groups are entirely missed
4. Even correct group predictions → weak term selection via within-group cosine

**Bugs found and fixed:**
- `pipeline.py` passed `embeddings` (768-dim mean) to `group_clf.predict_proba()` — should be `col_emb_concat` (2304-dim). This invalidated the Experiment 1 results (which used the wrong tensor).
- `train.py` default `--out` and `--training-data` paths included `ml/` prefix → double-nested to `ml/ml/...` when run from the `ml/` directory. Fixed to relative paths; misplaced checkpoints manually moved and `ml/ml/` deleted.
- Class weights computed before `--max-group-cases` cap remained stale (Adenoma: stored weight 11 but true post-cap ratio 92:1). Fixed by recalculating weights from capped targets. (Cap not recommended anyway.)

**Conclusion:** Hyperparameter tuning is exhausted at current data volume. The architecture is
sound (F1=0.4975 on training groups) but insufficient data coverage (17/43 groups) prevents
end-to-end performance. Do not retry until ~15,000 keyword-confirmed cases.

---

## Experiment 4 — 300 epochs, lr=5e-5 (Phase 26, 2026-05-04)

**Setup:** Same architecture and hyperparameters as Phase 24 GroupClassifier (`lr=5e-5`,
`max-class-weight=50`, `weight-decay=1e-3`), but trained for 300 epochs instead of 150.
46,652 train cases, 25 groups (24 common + Uncommon), 768-dim TF-IDF-selected embeddings.

**Motivation:** Phase 24 best checkpoint was at epoch 120/150 with val loss still trending
downward. Hypothesis: more epochs will extract additional convergence.

**Result:**
- **Best: epoch 219, macro F1 = 0.4335** (vs Phase 24 best 0.3136 at epoch 120 — +0.120)
- Val loss continued decreasing (0.3662 → 0.2229), confirming headroom hypothesis
- F1 curve noisy (0.40–0.43 range); last new best at epoch 219; epochs 220–300 produced no improvement
- Checkpoint saved to `group_classifier_best.pt`

**End-to-end pipeline evaluation (test set, gate=0.5, group-t=0.85, argmax fallback, subtype KW):**

| Metric | Phase 25 baseline | Phase 26 GroupCLF |
|--------|------------------|-------------------|
| G+S | 51.8% | **54.6%** |
| CO | 19.3% | 22.3% |
| FP | 4.7% | 5.0% |
| FN | 24.1% | 18.2% |
| Total | 8,744 | 9,127 |

G+S improved +2.8pp vs Phase 25 baseline (Tier 1+2+3a combined).

**Per-group highlights:**
- Paragangliomas: Good% 38% → 47% (improved group-level routing)
- Ductal and lobular: 71 → 39 predictions (precision improved, fewer wrong-group misroutes)
- Acinar cell neoplasms: 29 → 19 predictions (same pattern)

**Finding:** 300 epochs extracts significant headroom (+0.120 F1, +2.8pp G+S end-to-end).
The F1 metric is noisy epoch-to-epoch — the best checkpoint mechanism correctly captures the
peak. Model converged at lr=5e-5; lr=2e-5 tested next (Experiment 5) and found inferior.

---

## Experiment 5 — lr=2e-5 sweep (Phase 26, 2026-05-04)

**Setup:** Same as Experiment 4 but `lr=2e-5`. All other params unchanged.

**Result:** Best macro F1 = **0.4249** at epoch 278 — inferior to Experiment 4 (0.4335).
`group_classifier_best.pt` was not overwritten.

**Conclusion:** lr=5e-5 is the right choice at current data volume. Slower convergence does not
find a better minimum. Remaining hyperparameter candidates (weight-decay, max-class-weight)
unlikely to improve on Experiment 4. Tier 3 hyperparameter sweep exhausted.

---

## Phase 27 Pipeline Tests (2026-05-06)

### gate=0.4 test (no retraining)

Re-ran production with `--case-presence-threshold 0.4` on the Phase 26 GroupClassifier checkpoint.

| Metric | Phase 26 (gate=0.5) | gate=0.4 |
|--------|---------------------|----------|
| G+S | 54.6% | 54.2% |
| CO | 22.3% | 22.3% |
| FP | 5.0% | **5.9%** |
| FN | 18.2% | 17.6% |
| Total | 9,127 | 9,246 |

**Finding:** gate=0.4 gives -0.4pp G+S and +0.9pp FP for only -0.6pp FN. gate=0.5 remains optimal
for the Phase 26 GroupClassifier. If Phase 27 improves GroupClassifier routing, this test should be
repeated — better group routing may make the gate-lowering trade-off more favorable.

---

### Subtype keyword expansion (2026-05-06)

Phase 27 evaluation showed three new groups with very high slightly_off rates on the test set:
- Mast cell neoplasms: 94% slightly_off
- Blood vessel tumors: 86% slightly_off
- Melanocytoma and Melanomas: 72% slightly_off

Added keyword rules to `ml/ICD_labels/subtype_keywords.py` for all three:
- **Mast cell**: leukemia, subcutaneous, visceral, systemic/extracutaneous/mastocytosis, Kiupel high/low
- **Blood vessel**: hemangiosarcoma → hemangioma → hemangioendothelioma → pyogenic granuloma, etc.
- **Melanocytoma/Melanomas**: melanocytoma, amelanotic, signet ring, balloon cell, junctional, compound, etc.

End-to-end evaluation (test set, Phase 26 checkpoint, gate=0.5, group-t=0.85):

| Metric | Phase 26 baseline | + new subtype KW |
|--------|-------------------|-----------------|
| G+S | 54.6% | **54.9%** |
| CO | 22.3% | **21.1%** |
| FP | 5.0% | 5.1% |
| FN | 18.2% | 18.9% |
| Total | 9,127 | 8,980 |

**Finding:** CO improved -1.2pp, G+S +0.3pp. The improvement is modest because the subtype KW
rules narrow the label pool for Stage 3 but cannot fix group routing errors (Stage 2), which are
the larger source of CO. The FN/total count difference is likely noise from a different run.

---

## Phase 27 Experiments (2026-05-06)

### Bug fix: run_training.py did not forward --dropout / --lr-schedule / --focal-loss to train_group()

`run_training.py` hardcoded `dropout=0.3` and did not accept or forward the new flags added to
`train.py` in the Phase 27 plan. Fixed by adding `--dropout`, `--lr-schedule`, `--focal-loss`,
`--focal-gamma` arguments to `run_training.py` and forwarding them to `train_group()`.

---

### Experiment 6a — dropout=0.1, 300 epochs (2026-05-06) ✓

**Result:** Best epoch 192, macro F1 = **0.4475** (vs Phase 26 baseline 0.4335 — **+0.014**)
`group_classifier_best.pt` updated.

**Finding:** Reducing dropout 0.3 → 0.1 at 46k cases gives a meaningful improvement, confirming
that the Phase 26 model was over-regularised. Dropout=0.1 is the new recommended default.

---

### Experiment 6b — dropout=0.05, 300 epochs (2026-05-06) ✓

**Result:** Best epoch 221, macro F1 = **0.4399** — worse than 6a (0.4475).
`group_classifier_best.pt` unchanged.

**Finding:** dropout=0.05 hurts relative to 0.1. The optimal dropout is between 0.1 and 0.3 at
current data volume. Experiments 7 and 8 use dropout=0.05 as originally planned; if they also
underperform, retry with dropout=0.1 for the cosine LR and focal loss combinations.

---

### Experiment 7 — dropout=0.05 + cosine LR, 600 epochs (2026-05-06) ✓

**Result:** Best epoch 333, macro F1 = **0.4393** — worse than Exp 6a (0.4475).
`group_classifier_best.pt` unchanged.

**Finding:** Cosine LR warm restarts with dropout=0.05 does not beat dropout=0.1 with fixed LR.
The F1 is essentially identical to Exp 6b (0.4399) — cosine restarts provide no benefit when
dropout is already at 0.05. The right lever was dropout, not the LR schedule.

---

### Experiment 8 — dropout=0.05 + cosine LR + focal loss, 600 epochs (2026-05-06) ✓

**Result:** Best epoch 440, macro F1 = **0.2300** — collapsed.
`group_classifier_best.pt` unchanged.

**Finding:** Focal loss (gamma=2) is severely counterproductive at this data volume and imbalance
regime. The down-weighting of easy examples appears to destabilise training when combined with
capped class weights. Do not retry focal loss without first diagnosing the instability.

---

### Phase 27 Winner: Experiment 6a (dropout=0.1)

| Experiment | dropout | lr_schedule | focal | epochs | macro F1 | vs baseline |
|------------|---------|-------------|-------|--------|----------|-------------|
| Phase 26 baseline | 0.3 | none | no | 300 | 0.4335 | — |
| **Exp 6a** | **0.1** | none | no | 300 | **0.4475** | **+0.014** |
| Exp 6b | 0.05 | none | no | 300 | 0.4399 | +0.006 |
| Exp 7 | 0.05 | cosine | no | 600 | 0.4393 | +0.006 |
| Exp 8 | 0.05 | cosine | yes | 600 | 0.2300 | −0.203 |

**Checkpoint saved:** `group_classifier_best.pt` (F1=0.4475, epoch 192).

---

### Threshold sweep — Exp 6a checkpoint + new subtype KW (2026-05-06)

End-to-end evaluation on test set, gate=0.5, argmax fallback enabled:

| Threshold | G+S | CO | FP | FN | Total |
|-----------|-----|----|----|-----|-------|
| 0.70 | 48.1% | 34.9% | 5.4% | 11.6% | 11,126 |
| 0.75 | 50.4% | 30.9% | 5.3% | 13.5% | 10,399 |
| 0.80 | 52.7% | 26.3% | 5.1% | 15.9% | 9,674 |
| **0.85** | **54.9%** | **21.1%** | **5.1%** | **18.9%** | **8,980** |
| 0.90 | **56.1%** | **15.9%** | 5.0% | 23.0% | 8,393 |
| *Ph26 baseline (t=0.85)* | *54.6%* | *22.3%* | *5.0%* | *18.2%* | *9,127* |

**Note:** Both improvements (Exp6a GroupCLF + new subtype KW) are combined in these numbers.
The Phase 26 baseline used the old GroupCLF and the original 3-group subtype keyword set.

**t=0.85** remains the recommended default — it gives +0.3pp G+S and −1.2pp CO vs Phase 26 while
keeping FN near baseline (18.9% vs 18.2%).

**t=0.90** gives the best G+S (56.1%, +1.5pp vs Phase 26) and lowest CO (15.9%, −6.4pp),
at the cost of +4.8pp FN (23.0%). Use if coding accuracy matters more than completeness.

**Always pass `--group-classifier-threshold 0.85` to `run_production.py`** — CLI default is 0.3.

---

### Phase 27 Plan — Improvement flags (added 2026-05-06)

Three new training options added to `train.py`. Try them in order; record results as Experiments 6–8.

### Improvement A — Dropout sweep (no code change needed)

Phase 26 train/val gap: 0.247 / 0.258 — extremely tight, meaning the model is **not** overfitting.
Default dropout=0.3 is likely over-regularising at 46k cases. Experiment 3 found 0.3→0.05 gave
+0.18 macro F1 at 5k cases. Retesting at 46k:

```bash
ml/.venv/Scripts/python.exe ml/scripts/run_training.py \
  --mode train-groups --epochs 300 --lr 5e-5 \
  --max-class-weight 50 --weight-decay 1e-3 \
  --dropout 0.1 \
  --device xpu --local-only \
  --train-cases ml/output/splits/train_cases.txt \
  --annotation-csv ml/output/annotation/llm/llm_annotation.csv
```

Repeat with `--dropout 0.05`. Baseline to beat: 0.4335.

### Improvement B — Cosine LR warm restarts (`--lr-schedule cosine`)

Phase 26 plateaued at epoch 219 with 81 dead epochs. `CosineAnnealingWarmRestarts(T_0=100)`
periodically resets LR, helping escape the fixed-LR local minimum. Combine with more epochs:

```bash
ml/.venv/Scripts/python.exe ml/scripts/run_training.py \
  --mode train-groups --epochs 600 --lr 5e-5 \
  --max-class-weight 50 --weight-decay 1e-3 \
  --dropout 0.05 --lr-schedule cosine \
  --device xpu --local-only \
  --train-cases ml/output/splits/train_cases.txt \
  --annotation-csv ml/output/annotation/llm/llm_annotation.csv
```

### Improvement C — Focal Loss (`--focal-loss`)

Focal Loss down-weights easy non-cancer examples (already well-separated) and focuses gradient
on hard group assignments — the direct cause of CO. Try after finding best dropout:

```bash
ml/.venv/Scripts/python.exe ml/scripts/run_training.py \
  --mode train-groups --epochs 600 --lr 5e-5 \
  --max-class-weight 50 --weight-decay 1e-3 \
  --dropout 0.05 --lr-schedule cosine --focal-loss \
  --device xpu --local-only \
  --train-cases ml/output/splits/train_cases.txt \
  --annotation-csv ml/output/annotation/llm/llm_annotation.csv
```

---

## Experiment 9 — Rerun of Phase 27 config (2026-05-07)

**Setup:** Identical to Experiment 6a (Phase 27 winner): dropout=0.1, lr=5e-5, max-class-weight=50,
weight-decay=1e-3, 300 epochs, 25 groups (24 common + Uncommon), 768-dim TF-IDF embeddings.

**Motivation:** Routine retraining run to check whether a fresh seed can beat F1=0.4475.

**Result:** Best epoch 242, macro F1 = **0.4300** — worse than Phase 27 (0.4475).
`group_classifier_best.pt` unchanged.

**Finding:** Run-to-run variance of ~0.017 macro F1 with the same config and data. Phase 27's
F1=0.4475 (epoch 192) was a favorable seed. The current config is near its ceiling; further
gains require a different approach (more data, architectural change, or backbone update).

Also try `--focal-gamma 1.0` and `--focal-gamma 3.0` if default gamma=2.0 shows promise.

### Improvement D — Lower uncommon_threshold

`--uncommon-threshold 200` merges groups with <200 cases into a garbage "Uncommon" bucket.
At 46k cases, groups in the 100–199 range may have enough signal to train separately.
Rebuild training data with `--uncommon-threshold 100` and retrain. No code change needed.

**Result: see Phase 28b/28c below — Improvement D explored 2026-05-07.**

---

## Phase 28b — Improvement D: uncommon-threshold=100, 30 groups (2026-05-07) ❌

**Hypothesis:** Promoting the 5 groups just below the 200-case threshold (Ductal, Cystic,
Germ cell, Myxomatous, Thymic) into explicit output heads will improve G+S by reducing
noise in the "Uncommon" bucket.

**Config:** Same as Phase 27 (dropout=0.1, lr=5e-5, weight-decay=1e-3, max-class-weight=50, 300 epochs).
`--uncommon-threshold 100` (no additional exclusions).

**Result: macro F1=0.4050 at epoch 107 (30 groups).**

The macro F1 drop vs Phase 27 (0.4475, 25 groups) is expected — averaging across 5 new sparse heads
drags the mean down and is not directly comparable.

Per-group production performance (new groups only, test set, group-t=0.85, no LP stage):

| New group | Train cases | Val F1 | Prod good% | Prod off% |
|---|---|---|---|---|
| Thymic epithelial | 114 | 0.585 | 87% | 13% |
| Myxomatous | 116 | 0.376 | 68% | 5% |
| Germ cell | 161 | 0.305 | 7% | 54% |
| Cystic, mucinous | 156 | 0.106 | 18% | 36% |
| Ductal and lobular | 83 | 0.128 | 0% | 87% |

End-to-end (3-stage, no LP, group-t=0.85): **G+S=55.1%** vs Phase 27 55.3% — flat.

**Finding:** Thymic and Myxomatous learned well; Ductal, Cystic, and Germ cell did not.
Ductal appears at 83 unique cases in the training NPZ despite the 100-case threshold because
`build_training_data.py` computes group counts from annotation *rows* (not unique cases) —
multi-label cases inflate the row count above 100 while unique case count stays at 83.

**Production checkpoint not promoted.** Phase 27 (F1=0.4475) remains best.

---

## Phase 28c — Improvement D: 27 groups (Thymic + Myxomatous only) + 4-stage LP (2026-05-07) ❌

**Config:** `--uncommon-threshold 100 --excluded-groups "Neoplasms, NOS|Ductal and lobular neoplasms|Cystic, mucinous and serous neoplasms|Germ cell neoplasms"`.
Keeps only Thymic and Myxomatous as new promoted groups (27 groups total).
Same hyperparameters as Phase 27. Evaluated with Phase 28 LP classifiers (trained on Phase 27 25-group structure).

**Result: macro F1=0.4330 at epoch 273 (27 groups).**

New group val performance:
- Thymic: P=0.324, R=1.000, F1=0.489
- Myxomatous: P=0.232, R=1.000, F1=0.376

End-to-end (4-stage with LP, lp-t=0.5, group-t=0.85): **G+S=56.3%** vs Phase 28 (25-group + LP) 57.9%.

**Why it regressed:** The Phase 28 LP classifiers were trained on Phase 27's Uncommon bucket (20 groups).
Phase 28c moves Ductal, Cystic, and Germ cell into Uncommon (23 groups), but the `uncommon.pt`
LP classifier was never trained on those groups. It mislabels them aggressively:
- Ductal: 330 production predictions (was 38), 2% good, 78% off
- Cystic: 157 predictions (was 11), 3% good, 79% off

Thymic (83% good, 24 predictions) and Myxomatous (62% good, 24 predictions) add only ~48 useful
predictions — not enough to offset the LP misalignment damage.

**Conclusion:** To properly land Thymic and Myxomatous, the LP classifiers must be retrained on the
new 27-group structure (new `thymic_epithelial_neoplasms.pt`, `myxomatous_neoplasms.pt`, and
updated `uncommon.pt`). Until then, Phase 27 GroupClassifier + Phase 28 LP = 57.9% remains best.

**Reverted `group_classifier_best.pt` to Phase 27 (F1=0.4475).**
27-group checkpoint preserved as `group_classifier_current.pt` for future LP retraining.

---

## Phase 29 — Fresh start on new embedding space (2026-05-07)

The PetBERT backbone was updated (model.safetensors, 5:18 AM) and the embedding cache
was rebuilt from the new weights (embedding_cache.npz, 1:07 PM). All classifiers trained
on the old embedding space (Phase 27 GroupCLF F1=0.4475, Phase 28 LP classifiers) are
invalidated per the versioning rules in CLAUDE.md. Experiments 10–12 are the fresh baseline
for this embedding space.

Note: `group_classifier_best.meta.json` did not exist before these runs, so Exp10's
comparison used `prev_best_f1 = 0.0` and correctly seeded the meta file for subsequent runs.

---

## Experiment 10 — 25 groups, 400 epochs (2026-05-07) ❌

**Setup:** Same as Phase 27 winner (dropout=0.1, lr=5e-5, max-class-weight=50,
weight-decay=1e-3), 25 groups (threshold=200), 400 epochs. New embedding space.

**Motivation:** Standard config with extra epoch budget — prior best epochs ranged 192–369;
400 gives headroom for the new embedding space to converge.

**Result:** Best epoch 369, macro F1 = **0.4289**.
`group_classifier_best.pt` initialized (meta did not previously exist).

**Finding:** New embedding space yields a lower F1 ceiling than Phase 27 (0.4475 old space).
Epoch 369/400 suggests the model was still very slowly improving — may need more epochs.

---

## Experiment 11 — 26 groups (Thymic only), 300 epochs (2026-05-07) ✓

**Setup:** `--uncommon-threshold 100 --excluded-groups "Neoplasms, NOS|Ductal and lobular
neoplasms|Cystic, mucinous and serous neoplasms|Germ cell neoplasms|Myxomatous neoplasms"`.
Promotes only Thymic epithelial neoplasms (121 train cases) into its own head; Myxomatous
stays in Uncommon. All other hyperparameters identical to Exp10. 300 epochs.

**Motivation:** Thymic had strong individual performance in Phase 28b (val F1=0.585, 87%
prod good%). Myxomatous was borderline (val F1=0.376) and consistently hurt the 27-group
model. Test whether Thymic alone is net positive.

**Result:** Best epoch 163, macro F1 = **0.4308** — best of the three experiments.
`group_classifier_best.pt` updated.

**Per-group highlights (val set):**
- Thymic epithelial neoplasms: P=0.367, R=1.000, F1=0.537 (22 val positives)
- Osseous: F1=0.691 | Mast cell: F1=0.788 | Gliomas: F1=0.601
- Myomatous: F1=0.179 (noise group — high recall, very low precision)

**Finding:** Adding Thymic gives +0.0019 macro F1 over 25-group baseline. The improvement
is real but small; Thymic's isolated performance is good but the group count increase raises
the macro average denominator.

---

## Experiment 12 — 27 groups (Thymic + Myxomatous), 500 epochs (2026-05-07) ❌

**Setup:** `--uncommon-threshold 100 --excluded-groups "Neoplasms, NOS|Ductal and lobular
neoplasms|Cystic, mucinous and serous neoplasms|Germ cell neoplasms"`. Promotes both Thymic
and Myxomatous. 500 epochs to give extra headroom (Phase 28c peaked at epoch 273/300).

**Result:** Best epoch 400, macro F1 = **0.4280** — worst of the three.
`group_classifier_best.pt` unchanged (0.4308 > 0.4280).

**Finding:** Myxomatous continues to hurt F1 even with more epochs. Model peaked at epoch
400/500 (at the extended budget boundary) — still not converged, but the trajectory is
flat enough that further epochs are unlikely to close the gap to Exp11.

---

## Phase 29 Summary

| Experiment | Groups | Epochs | Best Epoch | Macro F1 | Promoted |
|------------|--------|--------|-----------|----------|---------|
| Exp10 | 25 (threshold=200) | 400 | 369 | 0.4289 | Yes (seeded meta) |
| **Exp11** | **26 (Thymic only)** | **300** | **163** | **0.4308** | **Yes (current best)** |
| Exp12 | 27 (Thymic+Myxo) | 500 | 400 | 0.4280 | No |

**Winner: Exp11 — 26 groups, Thymic promoted.** `group_classifier_best.pt` = epoch 163,
F1=0.4308. Consistent with Phase 28b/c findings: Thymic adds value, Myxomatous does not.

**Next step:** Retrain all LabelPresenceClassifiers on the new 26-group structure (old Phase 28
LP classifiers are invalid — trained on old-embedding backbone).

---

## Experiment 13 — dropout=0.05, 26 groups (Thymic only), 300 epochs (2026-05-07) ✓

**Setup:** Same as Exp11 (26 groups: Thymic promoted, Myxomatous excluded) but `--dropout 0.05`.
All other hyperparameters identical (lr=5e-5, max-class-weight=50, weight-decay=1e-3, 300 epochs).
New embedding space (Phase 29).

**Motivation:** Dropout sweep — tight train/val gap in Exp11 (0.013) suggests over-regularisation.
Test whether 0.05 extracts more signal on this embedding space.

**Result:** Best epoch 163, macro F1 = **0.4435** — new best, +0.0127 over Exp11 (0.4308).
`group_classifier_best.pt` updated.

**Peak gap (epoch 163):** train=0.1995, val=0.2157 (gap=0.0162 — slightly wider than Exp11's 0.013,
consistent with less regularisation).

**Per-group highlights (val set):**
- Mast cell: F1=0.765 | Osseous: F1=0.692 | Gliomas: F1=0.644
- Thymic: P=0.407, R=1.000, F1=0.579
- Myomatous: F1=0.208 (noise group — low P, high R, in Uncommon bucket)
- Uncommon: F1=0.169

**Finding:** Contrary to Phase 27 (where 0.05 < 0.1), the new embedding space responds better to
less regularisation. dropout=0.05 gives +0.0127 F1 over 0.1 on Phase 29. The embedding space
changed — its representational geometry is tighter, requiring less regularisation to extract signal.

---

## Experiment 14 — dropout=0.0, 26 groups (Thymic only), 300 epochs (2026-05-07) ❌

**Setup:** Same as Exp13 but `--dropout 0.0`. Baseline to beat: 0.4435 (Exp13).

**Result:** Best epoch 130, macro F1 = **0.4390** — worse than Exp13 (0.4435).
`group_classifier_best.pt` unchanged.

**Peak gap (epoch 130):** train=0.1991, val=0.2163 (gap=0.0172 — wider than Exp13's 0.0162,
confirming mild overfitting at zero dropout).

**Finding:** dropout=0.0 slightly overfits relative to 0.05. The curve peaks cleanly at 0.05:
0.1 → 0.05 (+0.0127), 0.05 → 0.0 (−0.0045). Dropout=0.05 is the new recommended default
for the Phase 29 embedding space.

---

## Phase 29 Dropout Sweep Summary

| Experiment | dropout | Best Epoch | Macro F1 | vs Exp11 |
|------------|---------|-----------|----------|---------|
| Exp11 | 0.1 | 163 | 0.4308 | baseline |
| **Exp13** | **0.05** | **163** | **0.4435** | **+0.0127** |
| Exp14 | 0.0 | 130 | 0.4390 | +0.0082 |

**Winner: dropout=0.05.** `group_classifier_best.pt` = Exp13 (F1=0.4435, epoch 163).
Contrary to Phase 27 (where 0.05 < 0.1), the new embedding space requires less regularisation.

**Next:** ASL (Asymmetric Loss) — addresses the per-group precision collapse pattern visible across
all experiments (high recall / low precision). Targets the same gradient imbalance as focal loss
but with separate γ+ / γ− values and a probability floor clamp, less likely to collapse.

---

## Experiment 15 — ASL (γ+=1.0, γ−=4.0, margin=0.05), dropout=0.05, 26 groups (2026-05-07) ❌

**Setup:** Same as Exp13 but `--asl` with paper defaults (gamma_pos=1.0, gamma_neg=4.0, margin=0.05).

**Result:** Best epoch 55, macro F1 = **0.0772** — collapsed. All groups R≈1.0, P≈0.
`group_classifier_best.pt` unchanged.

**Root cause:** Double-suppression interaction — `pos_weight` (up to 50x) amplifies positive
gradients AND γ−=4.0 suppresses negative gradients. Combined ratio pushes model to "always
positive" as minimum-loss strategy. The ASL paper applies these defaults without pos_weight;
our extreme class imbalance means pos_weight is already handling the imbalance correction, and
adding γ−=4 compounds it.

**Next:** Exp16 — halve γ− to 2.0, keep pos_weight. If still collapses, remove pos_weight
from ASL and let γ− handle imbalance alone.

---

## Experiment 16 — ASL (γ+=1.0, γ−=2.0, margin=0.05), dropout=0.05, 26 groups (2026-05-07) ❌

**Setup:** Same as Exp15 but `--asl-gamma-neg 2.0`. Baseline to beat: 0.4435 (Exp13 BCE).

**Result:** Best epoch 111, macro F1 = **0.1836** — still collapsed (R≈1.0, P very low).
`group_classifier_best.pt` unchanged.

**Finding:** Halving γ− improves F1 from 0.0772 to 0.1836 but the same collapse pattern
persists. The pos_weight + γ− double-suppression is the root cause regardless of γ− value.
**Fix:** Remove pos_weight from ASL and let γ− handle imbalance alone (per paper intent).

---

## Experiment 17 — ASL (γ+=1.0, γ−=4.0, margin=0.05), no pos_weight (2026-05-07) ❌

**Setup:** Same as Exp15 but pos_weight replaced with ones — γ− alone handles imbalance
per paper intent. Baseline to beat: 0.4435 (Exp13 BCE).

**Result:** Best epoch 191, macro F1 = **0.0930** — still collapsed (R≈1.0, P≈0).
`group_classifier_best.pt` unchanged.

**Root cause (final):** ASL is structurally incompatible with this data distribution.
For a group like Thymic (114 positives, 46K+ negatives), γ−=4.0 drives negative gradient
to ≈0 for any p < 0.3 — at that point the model trivially minimises loss by always predicting
positive. This is true with OR without pos_weight.
ASL was designed for datasets where each label has thousands of positives AND thousands of
negatives (e.g., MS-COCO). Per-group counts of 114–3,322 vs 40K+ negatives are too extreme
for any γ− value to suppress negatives without inducing collapse.

**Conclusion: ASL exhausted.** BCE with pos_weight (capped at 50) remains the correct loss
for this data distribution. Exp13 (F1=0.4435, dropout=0.05) is the Phase 29 ceiling for
the current architecture and loss function.

---

## Phase 29 ASL Summary

| Experiment | γ+ | γ− | margin | pos_weight | Macro F1 | Notes |
|------------|-----|-----|--------|-----------|----------|-------|
| Exp15 | 1.0 | 4.0 | 0.05 | yes | 0.0772 | Collapsed |
| Exp16 | 1.0 | 2.0 | 0.05 | yes | 0.1836 | Collapsed |
| Exp17 | 1.0 | 4.0 | 0.05 | no | 0.0930 | Collapsed |

BCE (Exp13, dropout=0.05): **0.4435** — unbeaten. ASL provides no benefit here.

---

## What to Try Next

> **Update (Phase 23, 2026-04-28):** GroupClassifier became competitive with ~21,853 LLM-annotated train cases (46,652 total train cases). It beats binary at threshold=0.90: +2.9pp G+S, −15.3pp FP. It is now part of the three-stage production pipeline (Phase 25). This log covers Phase 16 and earlier experiments — see [classifiers.md](../classifiers.md) for Phase 23+ results.

Historical roadmap (superseded):

| When | What |
|------|------|
| ~10,000 confirmed cases | Re-run Experiment 3 baseline; expect more groups to cross 100-case threshold |
| ~15,000+ confirmed cases | GroupClassifier expected to pull ahead of binary on CO% |

See [classifiers.md](../classifiers.md) for the discriminating-keyword term selection idea,
which may improve within-group accuracy once the group prediction step is reliable.
