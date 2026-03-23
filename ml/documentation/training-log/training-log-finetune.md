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
| `ml/training/finetune/build_contrastive_dataset.py` | Build `(report_text, label_text)` CSV from keyword annotations |
| `ml/training/finetune/train_contrastive.py` | InfoNCE training loop, saves HF checkpoint |
| `ml/scripts/run_training.py --mode contrastive-finetuning` | Orchestrates both steps |

### Standard Run Command

```bash
ml/.venv/Scripts/python.exe ml/scripts/run_training.py \
  --mode contrastive-finetuning \
  --skip-keyword-scan \
  --epochs 3 \
  --batch-size 32 \
  --lr 2e-5 \
  --temperature 0.07 \
  --device xpu \
  --local-only
```

> **Note:** Always pass `--skip-keyword-scan` — the keyword scan step is broken due to the `ICD_labels` package rename. Annotation already exists at `ml/output/annotation/keyword/keyword_annotation.csv`.

### After Fine-tuning: Cold Start + Retrain

The embedding space changes after fine-tuning. Old cached embeddings and the CO bank
are anchored to the old space and will add noise — delete them:

```bash
rm -f ml/data/embedding_cache.npz
rm -f ml/output/training/contrastive/evaluation_co_bank.csv
rm -f ml/model/checkpoints/contrastive/presence_classifier_current.pt
```

Then retrain the PresenceClassifier with the fine-tuned backbone:

```bash
ml/.venv/Scripts/python.exe ml/scripts/run_training.py \
  --mode binary \
  --skip-keyword-scan \
  --label "contrastive cold-start c1" \
  --co-neg-per-case 5 \
  --fp-neg-per-case 10 \
  --embedding-min-sim 0.05 \
  --epochs 25 \
  --recall-weight 0.25 \
  --hidden-dim 512 \
  --model ml/model/checkpoints/contrastive \
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

- [x] `build_contrastive_dataset.py` — reads annotation + report CSVs, writes pairs CSV
- [x] `train_contrastive.py` — InfoNCE loop, saves checkpoint
- [x] `run_finetune_contrastive.py` — orchestration runner
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
Checkpoint saved to `ml/model/checkpoints/contrastive/`.

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

**Best checkpoint:** `ml/model/checkpoints/binary/presence_classifier_best.pt` (69.0%, c8)
**Phase 17 backup:** `ml/model/checkpoints/contrastive/presence_classifier_best_phase17_contrastive.pt`

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
