# Fine-tuned PetBERT Training Log

No training runs yet. Known code issues must be resolved before a full run.

See the **Known Issues** section in [petbert-pipeline.md](petbert-pipeline.md) for the
list of bugs to fix before running.

---

## Prerequisite Checklist

- [ ] Fix `WeightedTrainer` constructor argument order (`class_weights` must be keyword-only)
- [ ] Move class weights to device in `compute_loss`, not `__init__`
- [ ] Add stratified val split in `build_dataset.py`
- [ ] Guard against `--finetuned-model-path` and `--presence-classifier` set simultaneously
- [ ] Replace deprecated `evaluation_strategy` with `eval_strategy`
- [ ] Add `local_files_only=True` to tokenizer call in `build_dataset.py`

---

## When to Run

Fine-tuned PetBERT is architecturally the same as the GroupClassifier (group prediction →
within-group cosine term selection) but replaces the frozen-embedding MLP with a full
transformer fine-tuned end-to-end. It shares the same data requirement: the GroupClassifier
needs to prove competitive (~10,000 confirmed cases) before adding the compute cost of
full transformer fine-tuning is justified.

Recommended sequence:
1. Resolve code issues above
2. Wait until GroupClassifier is competitive with binary (~10,000 cases)
3. Run a fine-tuned PetBERT benchmark against the GroupClassifier
