"""Fine-tune PetBERT on report text -> Vet-ICD group classification.

Use the Hugging Face Trainer API to finetune SAVSNET/PetBERT.
Load the dataset built by build_dataset.py and apply the computed
inverse frequency class weights to the CrossEntropyLoss.
"""

import argparse
import json
import os
from pathlib import Path

import evaluate
import numpy as np
import torch
import torch.nn as nn
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }


class WeightedTrainer(Trainer):
    """Custom Trainer subclass to inject class weights into the loss function."""
    
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        
        if self.class_weights is not None:
            # Move weights to the same device as the model
            self.loss_fct = nn.CrossEntropyLoss(
                weight=torch.tensor(self.class_weights, dtype=torch.float32).to(self.args.device)
            )
        else:
            self.loss_fct = nn.CrossEntropyLoss()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        loss = self.loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss


def train(
    dataset_dir: str = "ml/data/finetune_dataset",
    output_dir: str = "ml/model/checkpoints/petbert_finetuned",
    model_name: str = "SAVSNET/PetBERT",
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
):
    print(f"Loading dataset from {dataset_dir}...")
    dataset = load_from_disk(dataset_dir)
    
    config_path = Path(dataset_dir) / "config.json"
    weights_path = Path(dataset_dir) / "class_weights.npy"
    
    with open(config_path) as f:
        config = json.load(f)
        
    num_classes = config["num_classes"]
    class_weights = np.load(weights_path) if weights_path.exists() else None
    
    print(f"Loaded config: {num_classes} classes")
    if class_weights is not None:
        print("Found custom class weights.")

    print(f"Loading model {model_name}...")
    # map back to group strings for the Hugging Face config
    idx_to_group = {int(k): v for k, v in config["idx_to_group"].items()}
    group_to_idx = {v: k for k, v in idx_to_group.items()}
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_classes,
        id2label=idx_to_group,
        label2id=group_to_idx,
        ignore_mismatched_sizes=True # Ignore pre-trained classifier head if any
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available(), # MPS float16 has known issues with CELoss weights sometimes, keeping fp32 for MPS
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        report_to="none", # set to none so we don't report to wandb, etc.
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics,
    )

    print("Starting fine-tuning...")
    trainer.train()

    print(f"Saving final model to {output_dir}...")
    trainer.save_model(output_dir)
    # check later to debate if tokenizer should be saved with model or not
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_dir)
    
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ml/data/finetune_dataset")
    parser.add_argument("--out-dir", default="ml/model/checkpoints/petbert_finetuned")
    parser.add_argument("--model", default="SAVSNET/PetBERT")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()
    
    # put this to stop crash for some operations during training for Metal/MPS
    if torch.backends.mps.is_available():
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        
    train(
        dataset_dir=args.dataset,
        output_dir=args.out_dir,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
