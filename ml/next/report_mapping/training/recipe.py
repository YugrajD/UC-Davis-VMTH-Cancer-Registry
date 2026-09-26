"""The single source of production hyperparameters and seeds.

Every value here is verified against ``ml/documentation/training-guide.md``
(the exact production commands) and the old trainers it drives
(``ml/training/{binary,group,label_presence,contrastive}/*``,
``ml/scripts/run_training.py``). Where the guide and the old code's own
argparse defaults agree, that value is production; where the guide overrides
a looser code default, the guide wins. See the per-field comments below for
the source of each number. ``tests/test_recipe.py`` pins these values so a
future edit can't drift from the guide without a failing test.

A value that was hardcoded inside the old trainer (not exposed as a CLI flag)
is still pinned here since it shapes training the same way a flag would —
e.g. the gate's AdamW weight_decay=1e-4, or the backbone's warmup fraction.

Seeds: the old gate and LP trainers seeded ``torch.manual_seed`` +
``np.random.seed``; the old group trainer seeded only ``numpy`` (a local
``np.random.default_rng``, not the global state); the old contrastive
(backbone) trainer seeded nothing at all (ml-rewrite-plan.md, Findings). This
rewrite seeds every RNG (python, numpy, torch, and CUDA where present) for
every stage via ``seed_all`` — a deliberate improvement over legacy, called by
every trainer in this package before it touches any randomness.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch

# The production seed used throughout the old pipeline wherever a seed was set
# at all (gate, LP, the legacy split). ``train.py --seed`` may override this
# per run (e.g. WP1's L3 3-seed retrain); this is the default this repo has
# always used.
PRODUCTION_SEED = 42

# Legacy's LabelPresenceClassifier pair builder (ml/training/label_presence/
# build_training_pairs.py) takes its own ``seed: int = 42`` default that
# run_training.py never overrides — so within-group negative sampling is
# ALWAYS seed 42, independent of whatever --seed the run trains the model
# weights with. This is why the plan's L3 parity note says "LP negative
# sampling is seeded 42 in legacy, so the pairs should match exactly [across
# the 3 seeds]" (ml-rewrite-plan.md, Parity runbook). Carried over verbatim:
# labels.label_presence_pairs is always called with this constant, never with
# the run's --seed.
LP_PAIR_SAMPLING_SEED = 42


def seed_all(seed: int) -> None:
    """Seed python's ``random``, numpy, torch (CPU) and CUDA (if present).

    Call this first thing in every trainer, before building any dataset split
    or model. Deliberately broader than any single legacy trainer's seeding
    (see module docstring) so the same seed on the same device reproduces
    identical weights.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class GateRecipe:
    """CasePresenceClassifier (Stage 1 gate).

    Sources: ``training-guide.md`` Step 5 (``--epochs 20
    --case-presence-recall-weight 0.7``); ``ml/training/binary/
    train_case_presence.py`` argparse defaults for the rest (``--pos-weight``
    default 1.0, not overridden by the guide). ``dropout`` is
    ``model.constants.DEFAULT_DROPOUT`` (heads.py's default), never
    overridden anywhere. ``weight_decay`` is hardcoded in the old trainer's
    ``torch.optim.AdamW(..., weight_decay=1e-4)`` call, not exposed as a flag.
    """

    epochs: int = 20
    recall_weight: float = 0.7
    pos_weight: float = 1.0
    dropout: float = 0.3
    hidden_dim: int = 512
    lr: float = 1e-3
    batch_size: int = 256
    val_split: float = 0.15
    weight_decay: float = 1e-4
    seed: int = PRODUCTION_SEED


@dataclass(frozen=True)
class GroupRecipe:
    """GroupClassifier (Stage 2).

    Sources: ``training-guide.md`` Step 6 (``--epochs 300 --lr 5e-5
    --dropout 0.1 --max-class-weight 50 --weight-decay 1e-3``);
    ``ml/scripts/run_training.py::_train_groups`` hardcodes ``val_frac=0.2,
    threshold=0.3, min_group_cases=10`` in its call to ``train_group`` (not
    CLI-overridable); ``uncommon_threshold`` (200) and ``excluded_groups``
    (``"Neoplasms, NOS"``) are the argparse defaults, not overridden by the
    guide. Legacy's ``max_group_cases`` (a per-group positive-sample cap) is
    dropped here: production always ran with its default of 0 (no cap), and
    this trainer never implemented that code path, so there's nothing for the
    field to configure.
    """

    epochs: int = 300
    lr: float = 5e-5
    dropout: float = 0.1
    weight_decay: float = 1e-3
    max_class_weight: float = 50.0
    hidden_dim: int = 512
    val_frac: float = 0.2
    threshold: float = 0.3
    min_group_cases: int = 10
    lr_schedule: str = "none"
    uncommon_threshold: int = 200
    excluded_groups: tuple[str, ...] = ("Neoplasms, NOS",)
    seed: int = PRODUCTION_SEED


@dataclass(frozen=True)
class LabelPresenceRecipe:
    """Per-group LabelPresenceClassifier (Stage 3a).

    Sources: ``training-guide.md`` Step 7 (``--label-presence-epochs 25
    --label-presence-negs-per-pos 5 --label-presence-recall-weight 0.5
    --label-presence-n-cols 3 --label-presence-col-pair-mode
    --label-presence-col-combine learned``); ``ml/scripts/run_training.py``
    argparse defaults for ``dropout`` (0.3, "Phase 28") and ``weight_decay``
    (1e-4, "Phase 28"), not overridden by the guide.
    """

    epochs: int = 25
    negs_per_pos: int = 5
    recall_weight: float = 0.5
    dropout: float = 0.3
    weight_decay: float = 1e-4
    n_cols: int = 3
    col_pair_mode: bool = True
    col_combine: str = "learned"
    hidden_dim: int = 512
    batch_size: int = 256
    lr: float = 1e-3
    val_split: float = 0.15
    pos_weight: float = 1.0
    seed: int = PRODUCTION_SEED


@dataclass(frozen=True)
class BackboneRecipe:
    """Contrastive PetBERT backbone adaptation.

    Sources: ``training-guide.md`` Step 2 (``--epochs 3 --batch-size 32
    --lr 2e-5 --temperature 0.07``); ``max_length=256`` is the
    ``run_training.py`` argparse default (never overridden by the guide) and
    is distinct from the 512 used at inference (``report_mapping/model/
    backbone.py::INFERENCE_MAX_LENGTH`` — see ml-rewrite-plan.md, Findings).
    ``weight_decay`` (0.01) and ``warmup_frac`` (0.06) are hardcoded in the
    old trainer, not exposed as flags.
    """

    epochs: int = 3
    batch_size: int = 32
    lr: float = 2e-5
    temperature: float = 0.07
    max_length: int = 256
    weight_decay: float = 0.01
    warmup_frac: float = 0.06
    seed: int = PRODUCTION_SEED


@dataclass(frozen=True)
class UncommonMergeRecipe:
    """Groups below ``min_cases`` merge into the shared "Uncommon" head;
    ``forced`` groups merge in regardless of case count. Mirrors
    ``GroupRecipe.uncommon_threshold`` / ``excluded_groups`` — kept as its own
    type since ``labels.py`` needs it independent of the rest of GroupRecipe's
    training-only fields (lr, dropout, ...).
    """

    min_cases: int = 200
    forced: tuple[str, ...] = ("Neoplasms, NOS",)


GATE = GateRecipe()
GROUP = GroupRecipe()
LABEL_PRESENCE = LabelPresenceRecipe()
BACKBONE = BackboneRecipe()
UNCOMMON_MERGE = UncommonMergeRecipe(min_cases=GROUP.uncommon_threshold, forced=GROUP.excluded_groups)
