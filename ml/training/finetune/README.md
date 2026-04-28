# End-to-end PetBERT Fine-tuning — Approach 4 (WIP)

Fine-tunes PetBERT end-to-end as a group sequence classifier using CrossEntropyLoss.
Not recommended until ~10,000+ confirmed cases; see model-training.md for details.

## Files
- `build_dataset.py` — Build HuggingFace `DatasetDict`; compute inverse-frequency class weights
- `train.py`         — Fine-tune with `WeightedTrainer` (class-weighted CrossEntropyLoss)

## Output
- Dataset: `ml/data/finetune_dataset/`
- Checkpoint: `ml/output/checkpoints/finetune/`

## See also
Contrastive fine-tuning (Approach 3, production best) lives in `ml/training/contrastive/`.
