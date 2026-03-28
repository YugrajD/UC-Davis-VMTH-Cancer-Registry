# Fine-tuned PetBERT Training Log

Two distinct fine-tuning approaches exist in this codebase:

| Approach | Scripts | Status | Data requirement |
|----------|---------|--------|-----------------|
| **Contrastive (InfoNCE)** | `build_contrastive_dataset.py`, `train_contrastive.py` | Ready to run | Works at current ~5,788 cases |
| **End-to-end classification** | `build_dataset.py`, `train.py` | Known bugs — do not run yet | Needs ~10,000+ cases |

---

## Approach A — Contrastive Fine-tuning (InfoNCE)

### Motivation

The binary PresenceClassifier plateau (~42% Good+Slight, ~30% CO floor) is driven by
labels competing via argmax over an embedding space that was never optimised for this
task. PetBERT was pre-trained on UK veterinary EHRs with masked-language-modelling —
it has no signal pushing report embeddings toward their correct label embeddings.

Contrastive fine-tuning directly optimises this: for each (report, matched_label) pair,
pull the report embedding toward the correct label embedding and push it away from all
other labels in the batch. The fine-tuned backbone then produces better per-column
embeddings, which the PresenceClassifier (retrained from scratch) uses as input.

### Architecture

```
Training:
    for each batch of N (report_text, label_text) pairs:
        report_emb = PetBERT.base_model(report_text) → mean pool → 768-dim → L2-norm
        label_emb  = PetBERT.base_model(label_text)  → mean pool → 768-dim → L2-norm
        sim_matrix = report_emb @ label_emb.T / temperature    # (N, N)
        loss = symmetric cross-entropy (diagonal = positives)   # InfoNCE
        backprop through PetBERT base transformer only

    save full AutoModelForMaskedLM checkpoint

Inference (after fine-tuning + cold start + PresenceClassifier retraining):
    unchanged — pipeline uses --model <checkpoint> --local-only
```

Label text format: `"{term} {group}"` — matches the pipeline's `build_taxonomy_label_texts()`.
Report text: non-empty columns concatenated as `"[COL NAME] text"`.

### Scripts

| Script | Role |
|--------|------|
| `ml/training/contrastive/build_contrastive_dataset.py` | Build `(report_text, label_text)` CSV from keyword annotations |
| `ml/training/contrastive/train_contrastive.py` | InfoNCE training loop, saves HF checkpoint |
| `ml/scripts/run_training.py --mode adapt-backbone` | Orchestrates both steps |

### Standard Run Command

```bash
ml/.venv/Scripts/python.exe ml/scripts/run_training.py \
  --mode adapt-backbone \
  --epochs 3 \
  --batch-size 32 \
  --lr 2e-5 \
  --temperature 0.07 \
  --device xpu \
  --local-only
```

> **Note:** Annotation is skipped automatically if `keyword_annotation.csv` already exists — no flag needed.

### After Fine-tuning: Cold Start + Retrain

The embedding space changes after fine-tuning. Old cached embeddings and the CO bank
are anchored to the old space and will add noise — delete them:

```bash
rm -f ml/data/embedding_cache.npz
rm -f ml/output/training/contrastive/evaluation_co_bank.csv
rm -f ml/output/checkpoints/contrastive/presence_classifier_current.pt
```

Then retrain the label classifier with the adapted backbone:

```bash
ml/.venv/Scripts/python.exe ml/scripts/run_training.py \
  --mode train-classifier \
  --label "adapted backbone c1" \
  --co-neg-per-case 5 \
  --fp-neg-per-case 10 \
  --embedding-min-sim 0.05 \
  --epochs 25 \
  --recall-weight 0.25 \
  --hidden-dim 512 \
  --model ml/output/checkpoints/contrastive \
  --device xpu \
  --local-only
```

### Key Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--epochs` | 3 | Keep low — 110M params, ~5,788 pairs |
| `--batch-size` | 32 | Larger = more in-batch negatives; try 64 if memory allows |
| `--lr` | 2e-5 | Standard BERT fine-tuning rate |
| `--temperature` | 0.07 | InfoNCE temperature; lower = harder negatives |
| `--max-length` | 256 | Token budget per text (matches pipeline per-column budget) |

### Design Decisions

**Why symmetric InfoNCE?** Both directions of the loss matter: each report should
identify its label, and each label should identify its report. The symmetric form
(averaging row-loss and column-loss) was used in CLIP and gives more stable gradients.

**Why in-batch negatives?** Simple to implement and effective at batch_size=32 with
~857 unique labels — collision probability (same label appearing twice in a batch) is
~4%, acceptable noise.

**Why `model.base_model` not the full model?** The MLM head is not called during the
contrastive forward pass, so it receives no gradients and its weights are unchanged.
The saved checkpoint is still a valid `AutoModelForMaskedLM` and loads in the pipeline
without any code changes.

**Why not hard negatives?** In-batch negatives for a first run. Hard negatives (same
group, wrong term) can be added later if improvement plateaus.

### Prerequisite Checklist

- [x] `ml/training/contrastive/build_contrastive_dataset.py` — reads annotation + report CSVs, writes pairs CSV
- [x] `ml/training/contrastive/train_contrastive.py` — InfoNCE loop, saves checkpoint
- [x] `ml/scripts/run_training.py --mode adapt-backbone` — orchestration runner
- [x] Run it and record results here

---

## Training Runs

### Run 1 — 2026-03-23 (Phase 17, production best)

**Fine-tuning config:** epochs=3, batch=32, lr=2e-5, temperature=0.07, device=xpu, pairs=7,398

| Epoch | Avg InfoNCE Loss |
|-------|-----------------|
| 1 | 1.9044 |
| 2 | 1.3581 |
| 3 | 1.2222 |

Loss decreased steadily — fine-tuning converged normally.
Checkpoint saved to `ml/output/checkpoints/contrastive/`.

**PresenceClassifier retraining (cold start, hd=512, co=5, fp=10, epochs=25):**

| Cycle | Good% | Slight% | Good+Slight | CO% | FP% | FN% | Notes |
|-------|-------|---------|-------------|-----|-----|-----|-------|
| c1 | 11.9 | 37.7 | **49.6%** | 22.3 | 27.4 | 0.7 | Cold start — cache rebuilt |
| c2 | 17.0 | 37.5 | **54.5%** | 22.1 | 23.1 | 0.3 | |
| c3 | 19.2 | 45.3 | **64.5%** | 8.8 | 26.5 | 0.2 | Large jump — CO bank kicking in |
| c4 | 20.4 | 47.6 | **68.0%** | 7.9 | 23.7 | 0.4 | |
| c5 | 19.9 | 46.4 | 66.3% | 6.9 | 26.5 | 0.3 | Minor dip |
| c6 | 20.7 | 48.0 | **68.7%** | 7.3 | 23.7 | 0.3 | New best |
| c7 | 19.6 | 47.0 | 66.6% | 6.7 | 26.4 | 0.3 | |
| c8 | 20.4 | 48.6 | **69.0%** | 6.9 | 23.7 | 0.3 | **Best checkpoint** |
| c9 | 19.5 | 47.8 | 67.3% | 6.4 | 26.1 | 0.3 | Plateau oscillation |
| c10 | 21.0 | 47.8 | 68.8% | 7.0 | 23.9 | 0.3 | Confirmed plateau |

**Best: c8 — 69.0% Good+Slight, CO=6.9%, FP=23.7%**

**vs Phase 16 (frozen PetBERT, best=41.9%):**
- Good+Slight: +27.1pp (41.9% → 69.0%)
- CO%: −22.7pp (29.6% → 6.9%) — contrastive training eliminated most wrong-group predictions
- FP%: −3.5pp (27.2% → 23.7%)

**Bugs fixed during this run:**
- `ml/ICD-labels/` (hyphen — invalid Python package name) renamed to `ml/ICD_labels/`
- All `from labels.*` imports updated to `from ICD_labels.*` across 6 files
- All `ml/labels/labels.csv` and `ml/ICD-labels/labels.csv` data paths updated to `ml/ICD_labels/labels.csv`
- `train.py` had `model_name` hardcoded to `"SAVSNET/PetBERT"` in `load_cache()` call — caused cache invalidation when using contrastive backbone; fixed by threading `model_name` through `train()` and `run_cycle.py`

**Best checkpoint:** `ml/output/checkpoints/contrastive/presence_classifier_best.pt` (69.0%, c8)
**Phase 17 backup:** `ml/output/checkpoints/contrastive/presence_classifier_best_phase17_contrastive.pt`

---

### Run 2 — 2026-03-27 (Phase 18 — continued cycling after refactor)

**Context:** Continued cycling from Phase 17's contrastive backbone after a codebase
refactor. No new fine-tuning — same backbone, same embedding cache. Key refactor changes:
- CLI mode renamed: `--mode binary` → `--mode train-classifier`, `--mode contrastive-fine-tuning` → `--mode adapt-backbone`
- Checkpoint/output dirs moved: `ml/model/checkpoints/` → `ml/output/checkpoints/`
- `--skip-keyword-scan` flag removed (annotation file existence is checked automatically)
- `--co-neg-bank-csv` arg added to `run_training.py` (was only in `run_cycle.py`)
- `build_training_pairs.py` fixed to handle missing `evaluation.csv` gracefully (first cycle)
- New evaluation history written to `ml/output/evaluation/contrastive/` (previous history was in `binary/`)

**Config:** hd=512, co=5, fp=10, epochs=25, recall-weight=0.25, embedding-min-sim=0.05
**CO bank:** carried forward from Phase 17 via `--co-neg-bank-csv ml/output/training/binary/evaluation_co_bank.csv`

| Cycle | Good% | Slight% | Good+Slight | CO% | FP% | FN% | Notes |
|-------|-------|---------|-------------|-----|-----|-----|-------|
| c11 | 13.0 | 35.2 | 48.2% | 7.2 | 44.5 | 0.2 | First cycle — no eval.csv yet, FP hard negatives = 0 |
| c12 | 21.5 | 48.0 | **69.5%** | 6.8 | 23.3 | 0.4 | FP hard negatives restored — immediate recovery |
| c13 | 18.9 | 46.1 | 65.0% | 6.5 | 28.2 | 0.2 | Low cycle (fewer FP hard neg from c12) |
| c14 | 21.3 | 48.3 | **69.6%** | 7.0 | 23.1 | 0.4 | New best |
| c15 | 18.5 | 46.0 | 64.5% | 7.0 | 28.3 | 0.2 | Low cycle |
| c16 | 20.3 | 50.1 | **70.4%** | 6.5 | 22.7 | 0.4 | **Best checkpoint — first time >70%** |
| c17 | 18.9 | 46.7 | 65.6% | 6.3 | 27.8 | 0.2 | Low cycle |
| c18 | 20.2 | 49.6 | 69.8% | 6.8 | 23.0 | 0.4 | High cycle, below c16 |
| c19 | 19.2 | 46.3 | 65.5% | 6.6 | 27.7 | 0.3 | Plateau confirmed |

**Best: c16 — 70.4% Good+Slight, CO=6.5%, FP=22.7%**

**vs Phase 17 best (69.0%):** +1.4pp Good+Slight, −0.4pp CO, −1.0pp FP

**Key observations:**
- c11 resets to ~48% because there is no evaluation.csv to supply FP hard negatives —
  unavoidable whenever the output subdir changes (e.g. after a refactor or cold start)
- After c12, the model enters a stable alternating pattern: "high" cycles (~69–70%, FP~23%)
  follow "low" cycles (~65%, FP~28%), driven by FP negative count swinging each cycle
- High-cycle peaks crept up slowly: 69.5 → 69.6 → 70.4 → 69.8 — genuine but slow improvement
- CO bank additions slowed to ~167 new pairs/cycle by c17–c19, indicating diminishing new signal
- Plateau reached at ~70.4%; further gains likely require more labelled data or a second
  round of contrastive fine-tuning

**Best checkpoint:** `ml/output/checkpoints/contrastive/presence_classifier_best.pt` (70.4%, c16)
**Phase 17 backup still valid:** `ml/output/checkpoints/contrastive/presence_classifier_best_phase17_contrastive.pt`

---

### Run 3 — 2026-03-27 (Phase 19 — Round 2 warm-start fine-tuning)

**Context:** Second InfoNCE fine-tuning pass, warm-starting from the Phase 18 backbone.
Same 7,398 pairs, lower LR (1e-5 vs 2e-5), 2 epochs, `--skip-pair-build`.
The goal was to push further with the same data before committing to hard-negative Round 3.

**Fine-tuning config:** epochs=2, batch=32, lr=1e-5, temperature=0.07, device=xpu, pairs=7,398

| Epoch | Avg InfoNCE Loss |
|-------|-----------------|
| 1 | 1.2089 |
| 2 | 1.1474 |

Round 1 had ended at 1.2222 (epoch 3). Round 2 picked up from there and pushed to **1.1474**,
confirming meaningful additional learning on the same pairs at lower LR.

**PresenceClassifier retraining (cold start, hd=512, co=5, fp=10, epochs=25, CO bank carried forward):**

| Cycle | Good% | Slight% | Good+Slight | CO% | FP% | FN% | Notes |
|-------|-------|---------|-------------|-----|-----|-----|-------|
| c1 | 21.3 | 47.7 | **69.0%** | 7.0 | 23.6 | 0.4 | Cold start — no reset penalty (CO bank carried forward) |
| c2 | 20.0 | 46.8 | 66.8% | **6.0** | 26.8 | 0.3 | Low cycle — new CO low |
| c3 | 21.3 | 48.4 | **69.7%** | 6.3 | 23.7 | 0.4 | High cycle, climbing |
| c4 | 20.3 | 45.9 | 66.2% | **5.9** | 27.6 | 0.3 | Low cycle — new CO floor |
| c5 | 21.0 | 48.9 | **69.9%** | 6.2 | 23.6 | 0.4 | High cycle |
| c6 | 20.1 | 46.2 | 66.3% | 6.1 | 27.3 | 0.3 | Low cycle |
| c7 | 21.0 | 48.9 | 69.9% | 6.1 | 23.6 | 0.4 | High cycle — plateau confirmed |

**Best: c5/c7 — 69.9% Good+Slight, CO≈6.1%, FP≈23.6%**

**vs Phase 18 best (70.4%):** −0.5pp Good+Slight — did NOT surpass Phase 18 best checkpoint.
The Phase 18 `presence_classifier_best.pt` (70.4%, c16) remains the production best.

**Key observations:**
- No cold-start penalty because CO bank was carried forward — c1 landed at 69.0% immediately
- CO floor dropped further: low cycles hit 5.9% (vs Phase 18 low of 6.3%) — backbone did improve
- High cycles plateaued at 69.9%, just below Phase 18's 70.4% ceiling
- Two consecutive high cycles at 69.9% confirm plateau; more cycling unlikely to help
- **Conclusion:** Same training data cannot push past ~70% regardless of InfoNCE pass count.
  Hard-negative signal (Round 3) is the next lever.

**Best checkpoint for Round 2:** not a new best — Phase 18 `presence_classifier_best.pt` (70.4%) retained.
**Round 2 backbone saved as:** `ml/output/checkpoints/contrastive/` (overwrites Phase 18 backbone)
**Phase 18 backbone backup:** `ml/output/checkpoints/contrastive/model_phase18_backup.safetensors`

---

### Run 4 — 2026-03-27 (Phase 20 — Round 3 hard-negative fine-tuning)

**Context:** Third InfoNCE fine-tuning pass, warm-starting from the Phase 19 (Round 2) backbone.
Same positive pairs (7,398), plus 33,196 hard-negative triplets from the CO bank, combined with
a margin loss (weight=0.5, margin=0.3). Goal: push reports away from their known wrong-group labels
during backbone training itself.

**Hard-neg triplet build:**
- Source: `ml/output/training/binary/evaluation_co_bank.csv` (~24.3k wrong-group pairs)
- 4,664 cases with both a verified correct label and at least one wrong-group prediction
- 33,196 (report, correct_label, wrong_label) triplets written to `ml/data/hard_neg_pairs.csv`

**Fine-tuning config:** epochs=2, batch=32, lr=1e-5, temp=0.07, hard-neg-weight=0.5, margin=0.3

| Epoch | Avg InfoNCE Loss | Avg Hard-Neg Loss |
|-------|-----------------|-------------------|
| 1 | 1.3818 | 0.4712 |
| 2 | 1.2739 | 0.3246 |

Both losses decreased — model learned from hard negatives. InfoNCE slightly higher than Round 2
end (1.2739 vs 1.1474), expected given the harder combined objective.

**PresenceClassifier retraining (cold start, hd=512, co=5, fp=10, epochs=25, CO bank carried forward):**

| Cycle | Good% | Slight% | Good+Slight | CO% | FP% | FN% | Notes |
|-------|-------|---------|-------------|-----|-----|-----|-------|
| c1 | 20.3 | 45.5 | 65.8% | 6.5 | 27.4 | 0.3 | Low first cycle — CO bank partially stale for new embedding space |
| c2 | **21.9** | 46.6 | **68.5%** | 6.9 | 24.2 | 0.3 | High cycle — new Good% record |
| c3 | 20.6 | 45.7 | 66.3% | 6.5 | 26.9 | 0.3 | Low cycle |
| c4 | 21.7 | 46.8 | 68.5% | 6.9 | 24.2 | 0.4 | High cycle — plateau confirmed |

**Best: c2/c4 — 68.5% Good+Slight, CO=6.9%, FP=24.2%**

**vs Round 2 best (69.9%):** −1.4pp Good+Slight — **REGRESSION**
**vs Phase 18 best (70.4%):** −1.9pp Good+Slight — **REGRESSION**

**Phase 18 `presence_classifier_best.pt` (70.4%) remains the production best.**

**Key findings:**
- Hard-neg margin loss had a measurable and consistent effect: Good% rose (~+0.7pp) but Slight% fell (~−2.1pp)
- Net result: the model became more conservative — it pushes away from wrong groups but also
  over-shoots and drops some borderline Slight matches
- CO% on high cycles (6.9%) is not better than Phase 18 (6.5–7.0%) — CO floor didn't improve
- The hard-neg loss may need tuning: lower weight (0.25?) or lower margin (0.15?) to reduce
  the over-shooting effect on Slight predictions
- Alternatively, the CO bank entries may be too noisy as hard negatives (some "completely_off"
  predictions may have been close calls, not genuine wrong-group errors)
- Round 3 backbone saved at `ml/output/checkpoints/contrastive/model_phase20_round3_backup.safetensors`
- Round 2 backbone backup: `ml/output/checkpoints/contrastive/model_phase19_round2_backup.safetensors`
- Phase 18 backbone backup: `ml/output/checkpoints/contrastive/model_phase18_backup.safetensors`

---

### Run 5 — 2026-03-27 (Phase 21 — Round 3b, softer hard-negative weight=0.25)

**Context:** Retry of Round 3 hard-negative fine-tuning with halved weight (0.25 vs 0.5), warm-starting
from the Round 2 (Phase 19) backbone — not the Round 3 backbone, which had already regressed.
Same 33,196 hard-neg triplets (reused from Round 3). Goal: soften the push-away signal to preserve
borderline Slight matches while still correcting residual CO cases.

**Fine-tuning config:** epochs=2, batch=32, lr=1e-5, temp=0.07, hard-neg-weight=0.25, margin=0.3
**Warm-start from:** `model_phase19_round2_backup.safetensors` (Round 2 backbone)

| Epoch | Avg InfoNCE Loss | Avg Hard-Neg Loss |
|-------|-----------------|-------------------|
| 1 | 1.2823 | 0.5249 |
| 2 | 1.2022 | 0.4150 |

InfoNCE lower than Round 3's endpoint (1.2022 vs 1.2739) — softer hard-neg signal less disruptive
to positive-pair alignment. Hard-neg loss higher in absolute terms (weight=0.25 means gradient
contribution is halved, but the raw penalty can be larger).

**PresenceClassifier retraining (cold start, hd=512, co=5, fp=10, epochs=25, CO bank carried forward):**

| Cycle | Good% | Slight% | Good+Slight | CO% | FP% | FN% | Notes |
|-------|-------|---------|-------------|-----|-----|-----|-------|
| c1 | 21.7 | 44.5 | 66.2% | 6.2 | 27.3 | 0.3 | Low first cycle |
| c2 | 22.9 | 45.9 | **68.8%** | 6.3 | 24.6 | 0.4 | High cycle |
| c3 | 21.4 | 45.0 | 66.4% | 6.3 | 27.0 | 0.3 | Low cycle |
| c4 | 22.1 | 47.3 | **69.4%** | 6.1 | 24.1 | 0.4 | **Best checkpoint** — new Round 3b peak |
| c5 | 20.9 | 45.5 | 66.4% | 6.1 | 27.2 | 0.3 | Low cycle |
| c6 | 22.1 | 46.4 | 68.5% | 6.5 | 24.6 | 0.4 | High cycle — plateau |
| c7 | 21.3 | 45.0 | 66.3% | 6.3 | 27.1 | 0.3 | Low cycle — plateau confirmed |

**Best: c4 — 69.4% Good+Slight, CO=6.1%, FP=24.1%**

**vs Phase 18 best (70.4%):** −1.0pp Good+Slight — still below Phase 18
**vs Round 3 best (68.5%):** +0.9pp — improvement over weight=0.5
**vs Round 2 best (69.9%):** −0.5pp — just below Round 2

**Key findings:**
- Softer weight (0.25) meaningfully helped vs Round 3 (weight=0.5): +0.9pp Good+Slight, CO floor 6.1% (new low)
- High cycles oscillate 68.5–69.4% — plateau, not continuing to climb
- Slight% still suppressed vs Phase 18 (46–47% vs 50.1%) — hard-neg signal, even at 0.25, costs Slight
- Good% is consistently higher (~22%) vs Phase 18 (~20%) — the trade-off is real and tunable but not eliminable
- CO floor (6.1%) is marginally better than Phase 18 (6.5%) — hard-neg signal does push the right direction
- **Conclusion:** Hard-negative fine-tuning at any weight cannot break the ~70% ceiling with current data.
  The ~70% ceiling is a data ceiling. Slight% is the bottleneck — it represents borderline cases that
  require more labelled examples to resolve correctly.
- Round 3b backbone saved at `ml/output/checkpoints/contrastive/` (current `model.safetensors`)
- Round 3 backbone backup: `ml/output/checkpoints/contrastive/model_phase20_round3_backup.safetensors`
- **Phase 18 `presence_classifier_best.pt` (70.4%) remains the production best**

---

### Calibration Experiment — 2026-03-28 (does not help — do not use)

**Context:** Per-label score calibration (`--mode calibrate`) was implemented to correct
systematic score-variance differences between labels after mean-centering. The idea: labels
with low score variance lose argmax to higher-variance labels even when they are correct.
A scalar offset per label, grid-searched on the annotation set, should fix this.

Three objective functions were tried, all on top of the Phase 18 best checkpoint (70.4%):

**V1 — Exact-match objective** (maximize argmax exact-term hits for label L's own GT cases):
- In-sample: 58.7% → 70.5% exact match (+11.89pp) — looks good
- Production: Good+Slight 69.4% → 59.7% (−9.7pp) — **regression**
- 100 labels calibrated, offsets 0.01–0.30 (mean 0.16)
- Problem: offsets large enough to steal wins from other groups; Slight% collapsed −13.1pp

**V2 — Group-level objective** (maximize group wins for label L's own GT cases):
- In-sample: 93.7% → 89.2% Good+Slight (−4.50pp) — immediately shows as harmful
- Discarded without full production run
- Problem: groups already win 93.7% of annotation cases; the greedy per-label search
  still doesn't see cross-label interference, producing net-harmful offsets

**V3 — Net-gain objective** (optimize net Good+Slight gain across ALL annotated cases):
- In-sample: 93.7% → 94.0% (+0.31pp)
- Production: Good+Slight 69.4% → 63.9% (−5.5pp) — **regression**
- 26 labels calibrated, offsets 0.01–0.04 (mean 0.017) — much more conservative
- Off% improved (6.1% → 6.9%, vs v1's 11.9%), but Slight% still fell (47.3% → 41.3%)
- Total predictions dropped ~5k (~29k → ~24k); calibration changes argmax scores in a way
  that interacts with the embedding_min_sim threshold, converting Slight predictions to FN

**Summary:**

| | Pre-cal | v1 (exact match) | v3 (net-gain) |
|---|---|---|---|
| Good% | 22.1% | 25.5% | 22.6% |
| Slight% | 47.3% | 34.2% | 41.3% |
| **Good+Slight** | **69.4%** | 59.7% | 63.9% |
| Off% | 6.1% | 11.9% | 6.9% |

**Conclusion:** Score calibration does not help this model. The score-variance bias it was
designed to fix is not the binding constraint. The real ceiling is data — calibration cannot
resolve borderline Slight cases without more labelled examples. **Do not use `--calibration-offsets`
in production.** Phase 18 `presence_classifier_best.pt` (70.4%) without calibration remains the
production best.

**Code:** `ml/training/binary/calibrate.py` — kept for reference (uses net-gain objective, v3).
Offsets file `ml/output/calibration/label_offsets.json` is empty (`{}`).

---

## Group-Keyword Categorization Mode (2026-03-28)

### Motivation

At Phase 18 (70.4% Good+Slight), the remaining Slightly-off cases (42% of rank-1 predictions in
the default mode) are cases where the PresenceClassifier picks the right ICD group but the wrong
specific term within that group. Within a group, the main disambiguator is the ICD-O behavior digit
(the character after `/` in the code: `/0`=benign, `/1`=borderline, `/2`=in situ, `/3`=malignant,
`/6`=metastatic). This signal maps directly to plain clinical vocabulary and requires no model training.

### Implementation

New categorization mode: `--categorization-mode group-keyword` in `run_production.py`.

**Stage 1** (identical to default): Compute the default top-k — all labels with centered
PresenceClassifier score ≥ threshold, sorted descending, up to `max_predictions` (default 5).
CO, FP, and FN are provably unchanged vs default mode.

**Stage 2** (applied per-row): For each of the K top-k rows from Stage 1, identify that row's
predicted ICD group, apply `best_behavior(text)` from `ml/ICD_labels/behavior_keywords.py` to
match behavior vocabulary in the report text, filter group candidates to the matching digit, and
pick by highest raw PresenceClassifier score. Falls back to all group candidates if no keyword
signal. Rows where the group has no candidates fall back to the Stage 1 term.

This produces the same K rows per case as default mode — Stage 2 cannot change which cases
get predictions (the Uncategorized decision is Stage 1), only which specific term each row shows.

### evaluate.py Bug Fix (2026-03-28)

`score_prediction()` was returning `"false_positive"` for any case with no verified cancer
labels — including cases where the model correctly predicted `"Uncategorized"`. A correct
abstention on a non-cancer case is a **true negative**, not a false positive.

**Fix:** Added `"true_negative"` verdict when `predicted_term == "Uncategorized"` and
`matched_terms` is empty. True negatives are excluded from `evaluation.csv` and from all
metric totals. `build_training_pairs.py` requires no changes — it already ignores any
verdict that isn't `"false_positive"`.

**Impact on reported metrics:** All previous phase percentages (Phases 17–21) were computed
with true negatives incorrectly counted as false positives in the denominator, making G+S%
lower than the true rate against cancer cases. The relative ordering of phases is still valid
since all cycles used the same methodology. The corrected absolute numbers are below.

### Results (Phase 18 checkpoint, corrected evaluation, 2026-03-28)

Both modes write up to 5 top-k rows per case. Percentages are computed over prediction rows.

| Metric | Default mode (top-k) | **Group-keyword mode (top-k) — NEW PRODUCTION** |
|---|---|---|
| **Good%** | 30.6% | **56.5%** |
| **Slightly off%** | 55.9% | **30.0%** |
| **Good + Slight** | **86.5%** | **86.5%** |
| Completely off% | 9.4% | 9.3% |
| **False positive%** | **1.9%** | **1.9%** |
| False negative% | 2.2% | 2.2% |
| True negatives (excluded from CSV) | 6,329 | 6,330 |
| Rows in denominator | 17,860 | 17,838 |

Good+Slight is identical to default (86.5%) — Stage 2 only redistributes Slight → Good, it
does not change which cases pass or fail. Good% rose +25.9pp (30.6% → 56.5%) and Slightly off%
fell −25.9pp. FP, CO, and FN are within rounding of default mode.

Selected per-group Good% comparison:

| Group | Default top-k | Group-keyword top-k |
|---|---|---|
| Blood vessel tumors | 21% | **70%** |
| Adenomas and adenocarcinomas | 32% | **71%** |
| Squamous cell neoplasms | 24% | **79%** |
| Malignant lymphomas | 79% | **86%** |
| Osseous and chondromatous | 18% | **52%** |

### Remaining Slight

Groups where terms differ by more than behavior code (topography, histologic subtype):
- Meningiomas: 80% Slight — terms differ by topography (spinal vs intracranial), not behavior
- Odontogenic tumors: 82% Slight — subtype vocabulary
- Osseous neoplasms: 46% Slight — complex subtype vocabulary
- Gliomas: 46% Slight — complex subtype vocabulary

Future improvement: add topography/grade keyword vocabularies for these groups.

### Usage (production command)

```bash
ml/.venv/Scripts/python.exe ml/scripts/run_production.py \
  --categorization-mode group-keyword \
  --out-dir ml/output/production/contrastive_kw \
  --device xpu --local-only
```

Evaluate:
```bash
ml/.venv/Scripts/python.exe ml/evaluation/evaluate.py \
  --prediction-csv ml/output/production/contrastive_kw/petbert_predictions.csv \
  --out-dir ml/output/evaluation/contrastive_kw
```

---

## Approach B — End-to-end Group Classification (WIP, blocked)

Fine-tunes PetBERT as a sequence classifier directly predicting Vet-ICD-O groups.
Architecturally identical to the GroupClassifier but replaces the frozen-embedding
MLP with the full transformer.

**Status:** Not benchmarked. Known code bugs must be fixed, and the GroupClassifier
needs to prove competitive (~10,000 cases) before this is worth the compute cost.

### Known Bugs (must fix before running)

- [ ] `WeightedTrainer.__init__` argument order is fragile — `class_weights` should be keyword-only
- [ ] Class weights moved to device in `__init__` before device resolves — move to `compute_loss`
- [ ] No stratified val split in `build_dataset.py`
- [ ] `--finetuned-model-path` and `--presence-classifier` not mutually exclusive
- [ ] `evaluation_strategy` deprecated — replace with `eval_strategy`
- [ ] No `local_files_only=True` in `build_dataset.py` tokenizer call

### When to Run

1. Resolve bugs above
2. Wait until GroupClassifier proves competitive with binary (~10,000 confirmed cases)
3. Benchmark against GroupClassifier
