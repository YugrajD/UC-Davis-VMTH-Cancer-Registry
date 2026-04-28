# Contrastive PetBERT Fine-tuning — Approach 3 (Production)

InfoNCE fine-tuning of PetBERT on (report_text, label_text) pairs.
This is the current production approach (Phase 17, 69.0% Good+Slight).

## Files
- `build_contrastive_dataset.py` — Build (report, label) pairs CSV from keyword annotations
- `train_contrastive.py`         — InfoNCE training loop; saves HuggingFace checkpoint

## Output
- Pairs: `ml/data/contrastive_pairs.csv`
- Checkpoint: `ml/output/checkpoints/contrastive/`

## Usage
See `ml/documentation/model-training.md` or run via:
```bash
ml/.venv/Scripts/python.exe ml/scripts/run_training.py --mode adapt-backbone ...
```

## See also
End-to-end fine-tuning (Approach 4, WIP) lives in `ml/training/finetune/`.
