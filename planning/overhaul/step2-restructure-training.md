# Step 2 — Restructure training/

## File Moves

| From | To |
|---|---|
| `training/run_training_cycle.py` | `training/binary/run_cycle.py` |
| `training/build_training_pairs.py` | `training/binary/build_training_pairs.py` |
| `training/train_classifier.py` | `training/binary/train.py` |
| `training/evaluate_predictions.py` | `training/binary/evaluate.py` |
| `training/update_co_bank.py` | `training/binary/update_co_bank.py` |
| `training/log_evaluation.py` | `training/binary/log_evaluation.py` |
| `training/build_group_training_data.py` | `training/group/build_training_data.py` |
| `training/train_group_classifier.py` | `training/group/train.py` |

## New Files Created
- `training/binary/__init__.py` (empty)
- `training/group/__init__.py` (empty)
- `training/finetune/__init__.py` (empty)
- `training/finetune/README.md` (placeholder — see below)

## Files Deleted (after moves confirmed working)
```
training/run_training_cycle.py
training/build_training_pairs.py
training/train_classifier.py
training/evaluate_predictions.py
training/update_co_bank.py
training/log_evaluation.py
training/build_group_training_data.py
training/train_group_classifier.py
```

## `training/finetune/README.md` Content
```
# PetBERT Fine-tuning (Planned)

## Intended interface
- Input: labeled (report_text, cancer_term) pairs + base PetBERT model path
- Output: fine-tuned checkpoint at ml/model/checkpoints/petbert_finetuned/

## Integration
petbert_pipeline/embedding.py already accepts --model PATH.
A fine-tuned checkpoint can be passed directly — no other code changes needed.

## Planned files
- dataset.py  — PyTorch Dataset for (text, label) pairs
- train.py    — Fine-tuning loop (HuggingFace Trainer or manual)
- evaluate.py — Embedding quality metrics post fine-tune
```
