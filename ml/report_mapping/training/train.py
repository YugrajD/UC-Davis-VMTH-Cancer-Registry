"""Train a candidate generation: labels + split -> a generation directory with calibration "pending".

No thresholds are fitted here; ``calibrate.py`` does that on the calibration partition. Every training run
starts from the base checkpoint with a fresh head over the train-partition terms.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import transformers

import config
from generations import guards
from report_mapping import batching, generation
from report_mapping.inference import scoring
from report_mapping.model import ENCODER_ID, JointClassifier, load_tokenizer, new_encoder
from report_mapping.training import labels as labels_mod
from report_mapping.training import loop
from report_mapping.training.data import load_training_data, split_holdout
from report_mapping.training.recipe import Recipe, seed_all
from taxonomy.taxonomy import load_labels_taxonomy

RECALL_AT = 20


@dataclass(frozen=True)
class Holdout:
    texts: list[str]
    gate_targets: np.ndarray
    gold_idx: list[frozenset[int]]   # vocabulary indices of each case's terms


def roc_auc(scores: np.ndarray, positive: np.ndarray) -> float:
    """Area under the ROC curve (Mann-Whitney U with average ranks); NaN when one class is missing."""
    positive = positive.astype(bool)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def heldout_metrics(term_logits: np.ndarray, gate_logits: np.ndarray, gate_targets: np.ndarray,
                    gold_idx: list[frozenset[int]]) -> dict:
    """Gate AUC and BCE over all held-out cases; term top-1 accuracy and pair recall@RECALL_AT over the held-out
    CANCER cases (``gate_targets`` == 1; ``gold_idx[i]`` = vocabulary indices of case i's terms)."""
    loss = float(F.binary_cross_entropy_with_logits(torch.tensor(gate_logits), torch.tensor(gate_targets)))
    cancer = np.flatnonzero(gate_targets == 1.0)
    top = np.argsort(-term_logits[cancer], axis=1, kind="stable")[:, :RECALL_AT]
    hits = sum(len(gold_idx[i].intersection(top[k].tolist())) for k, i in enumerate(cancer))
    pairs = sum(len(gold_idx[i]) for i in cancer)
    top1 = sum(top[k, 0] in gold_idx[i] for k, i in enumerate(cancer)) / len(cancer) if len(cancer) else float("nan")
    recall = hits / pairs if pairs else float("nan")   # a --limit smoke can hold out no cancer case
    return {"heldout_gate_auc": roc_auc(gate_logits, gate_targets), "heldout_gate_loss": loss,
            "heldout_term_top1": float(top1), f"heldout_term_pair_recall_at_{RECALL_AT}": recall}


def fit_model(texts: list[str], term_targets: np.ndarray, gate_targets: np.ndarray, vocab: list[str], recipe: Recipe,
              device, model_id: str, max_steps: int = 0, holdout: Holdout | None = None
              ) -> tuple[JointClassifier, object, dict]:
    """A fresh model (base checkpoint ``model_id``, new heads over ``vocab``) trained on the given cases.
    With ``holdout``, its metrics are logged after every epoch. Returns (model, tokenizer, summary)."""
    seed_all(recipe.seed)
    cancer = gate_targets == 1.0
    if not cancer.any():
        raise ValueError("no cancer case to train on")
    tokenizer = load_tokenizer(model_id)
    token_ids, over = batching.encode_reports(tokenizer, texts, recipe.max_length)
    print(f"train {len(texts)} cases ({int(cancer.sum())} cancer), terms {len(vocab)}, "
          f"reports over {recipe.max_length} tokens: {over}", flush=True)
    model = JointClassifier(new_encoder(model_id), vocab).to(device)
    model.init_biases(torch.tensor(term_targets[cancer].mean(0)), float(gate_targets.mean()))

    def after_epoch(epoch: int) -> dict:
        term_logits, gate_logits, _ = scoring.logits_for(model, tokenizer, holdout.texts, recipe.max_length, device,
                                                         2 * recipe.token_budget)
        return heldout_metrics(term_logits, gate_logits, holdout.gate_targets, holdout.gold_idx)

    start = time.time()
    steps, epochs = loop.fit(model, tokenizer, token_ids, term_targets, gate_targets, recipe, device, max_steps,
                             after_epoch if holdout else None)
    summary = {"n_train": len(texts), "n_train_cancer": int(cancer.sum()), "n_terms": len(vocab),
               "terms_without_train_positive": int((term_targets[cancer].sum(0) == 0).sum()),
               "reports_over_max_length": over, "steps": len(steps), "train_seconds": time.time() - start,
               "epochs": epochs}
    return model, tokenizer, summary


def train(labels: str, split_id: str, out_dir: Path, recipe: Recipe, device, model_id: str = ENCODER_ID,
          limit: int = 0, max_steps: int = 0) -> dict:
    """Train on ``split_id``'s train partition and write the generation directory ``out_dir`` (which must be
    new or empty). ``labels``: a silver_id or labels CSV path. ``limit`` / ``max_steps``: smoke runs. Returns
    the manifest."""
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} is not empty; build the candidate into a new directory")
    guards.check_all(split_id)
    table = labels_mod.load_labels_table(labels)
    lineage = labels_mod.lineage(table, labels)  # before training, so a mixed-lineage table fails fast
    data = load_training_data(table, split_id, limit)

    term_info: dict[str, tuple[str, str]] = {}
    for label in load_labels_taxonomy(str(config.LABELS_CSV)):
        term_info.setdefault(label.term, (label.group, label.code))
    unknown = sorted(set(data.vocab) - set(term_info))
    if unknown:
        raise ValueError(f"{len(unknown)} label term(s) are not in the taxonomy, e.g. {unknown[:3]}")

    fit_idx, held_idx = split_holdout(data.gate_targets, recipe.holdout_fraction, recipe.holdout_seed)
    holdout = Holdout([data.texts[i] for i in held_idx], data.gate_targets[held_idx],
                      [frozenset(np.flatnonzero(data.term_targets[i]).tolist()) for i in held_idx])
    model, tokenizer, summary = fit_model(
        [data.texts[i] for i in fit_idx], data.term_targets[fit_idx], data.gate_targets[fit_idx], data.vocab, recipe,
        device, model_id, max_steps, holdout)

    model.save(out_dir, tokenizer)
    (out_dir / generation.TERM_INFO_FILE).write_text(
        json.dumps({t: list(v) for t, v in term_info.items()}, ensure_ascii=False), encoding="utf-8")
    final = summary["epochs"][-1]
    training = {
        "seed": recipe.seed, "device": str(device), "limit": limit, "max_steps": max_steps,
        "library_versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                             "python": platform.python_version()},
        "n_train": summary["n_train"], "n_holdout": len(held_idx),
        "holdout": {k: v for k, v in final.items() if k.startswith("heldout_")},
        "summary": {k: v for k, v in summary.items() if k not in ("n_train", "epochs")}, "epochs": summary["epochs"],
    }
    return generation.write_pending_manifest(
        out_dir, encoder_id=str(model_id), max_length=recipe.max_length, recipe=asdict(recipe),
        parents={"split_id": split_id, "labels_source": labels, **lineage}, training=training)
