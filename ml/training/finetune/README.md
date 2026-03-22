# PetBERT Fine-tuning (Planned)

## Intended interface
- Input: labeled (report_text, cancer_term) pairs + base PetBERT model path
- Output: fine-tuned checkpoint at `ml/model/checkpoints/petbert_finetuned/`

## Integration
`petbert_pipeline/embedding.py` already accepts `--model PATH`.
A fine-tuned checkpoint can be passed directly — no other code changes needed.

## Planned files
- `dataset.py`  — PyTorch Dataset for (text, label) pairs
- `train.py`    — Fine-tuning loop (HuggingFace Trainer or manual)
- `evaluate.py` — Embedding quality metrics post fine-tune
