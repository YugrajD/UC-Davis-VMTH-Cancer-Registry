"""k-fold cross-fit over the train partition: out-of-fold gate and term probabilities for the Report-Mapping audit.

Each fold trains a fresh model (the same loop as ``train.train``) on the other k-1 folds and scores the held-out
fold, so every train case gets a probability from a model that never saw it. The output is numbers only
(case ids, fold, targets, probabilities; term names are taxonomy terms), never report text.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

import io_utils
from generations import guards
from report_mapping.inference import scoring
from report_mapping.model import ENCODER_ID
from report_mapping.training import labels as labels_mod
from report_mapping.training import train as train_mod
from report_mapping.training.data import assign_folds, load_training_data
from report_mapping.training.recipe import Recipe

COLUMNS = ["case_id", "fold", "gate_target", "gate_prob", "term", "term_prob"]


def crossfit(labels: str, split_id: str, k: int, recipe: Recipe, device, out_csv: Path, limit: int = 0,
             max_steps: int = 0, model_id: str = ENCODER_ID) -> Path:
    """Write ``out_csv``: one row per labelled term of each cancer case (``term_prob`` is that term's probability
    from the fold model, empty if the term is outside the fold model's vocabulary), one row with an empty term per
    non-cancer case. ``limit`` / ``max_steps``: smoke runs. Returns ``out_csv``."""
    guards.check_all(split_id)
    data = load_training_data(labels_mod.load_labels_table(labels), split_id, limit)
    folds = assign_folds(data.gate_targets, k, recipe.seed)
    rows = []
    for fold in range(k):
        fit_idx, held_idx = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        in_vocab = data.term_targets[fit_idx].sum(0) > 0   # terms with a positive among the k-1 training folds
        fold_vocab = [t for t, keep in zip(data.vocab, in_vocab) if keep]
        print(f"fold {fold + 1}/{k}: train {len(fit_idx)}, held out {len(held_idx)}, terms {len(fold_vocab)}", flush=True)
        model, tokenizer, _ = train_mod.fit_model(
            [data.texts[i] for i in fit_idx], data.term_targets[fit_idx][:, in_vocab], data.gate_targets[fit_idx],
            fold_vocab, recipe, device, model_id, max_steps)
        term_logits, gate_logits, _ = scoring.logits_for(model, tokenizer, [data.texts[i] for i in held_idx],
                                                         recipe.max_length, device)
        term_probs, gate_probs = scoring.sigmoid(term_logits), scoring.sigmoid(gate_logits)
        fold_index = {t: j for j, t in enumerate(fold_vocab)}
        for row, i in enumerate(held_idx):
            base = {"case_id": data.case_ids[i], "fold": fold, "gate_target": int(data.gate_targets[i]),
                    "gate_prob": float(gate_probs[row])}
            terms = [data.vocab[j] for j in np.flatnonzero(data.term_targets[i])]
            for term in terms or [""]:
                prob = term_probs[row, fold_index[term]] if term in fold_index else ""
                rows.append({**base, "term": term, "term_prob": prob})
        del model
        torch.cuda.empty_cache()
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    io_utils.write_csv(pd.DataFrame(rows, columns=COLUMNS).sort_values(["case_id", "term"], kind="stable"), out_csv)
    return out_csv
