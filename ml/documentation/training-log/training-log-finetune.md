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
| `ml/scripts/run_finetune_contrastive.py` | Orchestrates both steps |

### Standard Run Command

```bash
ml/.venv/Scripts/python.exe ml/scripts/run_finetune_contrastive.py \
  --epochs 3 \
  --batch-size 32 \
  --lr 2e-5 \
  --temperature 0.07 \
  --device xpu \
  --local-only
```

### After Fine-tuning: Cold Start + Retrain

The embedding space changes after fine-tuning. Old cached embeddings and the CO bank
are anchored to the old space and will add noise — delete them:

```bash
rm -f ml/data/embedding_cache.npz
rm -f ml/output/training/binary/evaluation_co_bank.csv
rm -f ml/model/checkpoints/presence_classifier_current.pt
```

Then retrain the PresenceClassifier with the fine-tuned backbone:

```bash
ml/.venv/Scripts/python.exe ml/scripts/run_training.py \
  --label "contrastive cold-start c1" \
  --co-neg-per-case 5 \
  --fp-neg-per-case 10 \
  --embedding-min-sim 0.05 \
  --epochs 25 \
  --recall-weight 0.25 \
  --hidden-dim 512 \
  --model ml/model/checkpoints/petbert_contrastive \
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
- [ ] Run it and record results here

---

## Training Runs

### Run 1 — (not yet run)

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
