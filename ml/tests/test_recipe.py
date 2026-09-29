"""report_mapping/training/recipe.py: pins production hyperparameters/seeds
against ml/documentation/history/legacy-tree/training-guide.md and the old trainers (see
recipe.py's own per-field docstrings for each value's source), and checks
``seed_all`` actually makes python/numpy/torch reproducible.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from report_mapping.training import recipe


def test_gate_recipe_matches_training_guide_step_5():
    assert recipe.GATE.epochs == 20
    assert recipe.GATE.recall_weight == 0.7
    assert recipe.GATE.pos_weight == 1.0
    assert recipe.GATE.dropout == 0.3
    assert recipe.GATE.hidden_dim == 512
    assert recipe.GATE.weight_decay == 1e-4
    assert recipe.GATE.seed == 42


def test_group_recipe_matches_training_guide_step_6():
    assert recipe.GROUP.epochs == 300
    assert recipe.GROUP.lr == 5e-5
    assert recipe.GROUP.dropout == 0.1
    assert recipe.GROUP.weight_decay == 1e-3
    assert recipe.GROUP.max_class_weight == 50.0
    assert recipe.GROUP.hidden_dim == 512
    assert recipe.GROUP.val_frac == 0.2
    assert recipe.GROUP.threshold == 0.3
    assert recipe.GROUP.min_group_cases == 10
    assert recipe.GROUP.lr_schedule == "none"
    assert recipe.GROUP.uncommon_threshold == 200
    assert recipe.GROUP.excluded_groups == ("Neoplasms, NOS",)


def test_label_presence_recipe_matches_training_guide_step_7():
    assert recipe.LABEL_PRESENCE.epochs == 25
    assert recipe.LABEL_PRESENCE.negs_per_pos == 5
    assert recipe.LABEL_PRESENCE.recall_weight == 0.5
    assert recipe.LABEL_PRESENCE.dropout == 0.3
    assert recipe.LABEL_PRESENCE.weight_decay == 1e-4
    assert recipe.LABEL_PRESENCE.n_cols == 3
    assert recipe.LABEL_PRESENCE.col_pair_mode is True
    assert recipe.LABEL_PRESENCE.col_combine == "learned"


def test_backbone_recipe_matches_training_guide_step_2():
    assert recipe.BACKBONE.epochs == 3
    assert recipe.BACKBONE.batch_size == 32
    assert recipe.BACKBONE.lr == 2e-5
    assert recipe.BACKBONE.temperature == 0.07
    assert recipe.BACKBONE.max_length == 256
    assert recipe.BACKBONE.weight_decay == 0.01
    assert recipe.BACKBONE.warmup_frac == 0.06


def test_inference_max_length_differs_from_training_max_length():
    # ml-rewrite-plan.md Findings: 256 at training, 512 at inference -- both load-bearing.
    from report_mapping.model.backbone import INFERENCE_MAX_LENGTH

    assert recipe.BACKBONE.max_length == 256
    assert INFERENCE_MAX_LENGTH == 512
    assert recipe.BACKBONE.max_length != INFERENCE_MAX_LENGTH


def test_uncommon_merge_recipe_mirrors_group_recipe():
    assert recipe.UNCOMMON_MERGE.min_cases == recipe.GROUP.uncommon_threshold
    assert recipe.UNCOMMON_MERGE.forced == recipe.GROUP.excluded_groups


def test_production_seed_is_42():
    assert recipe.PRODUCTION_SEED == 42
    assert recipe.LP_PAIR_SAMPLING_SEED == 42


def test_seed_all_reproducible():
    recipe.seed_all(123)
    a = (random.random(), np.random.rand(), torch.rand(1).item())
    recipe.seed_all(123)
    b = (random.random(), np.random.rand(), torch.rand(1).item())
    assert a == b

    recipe.seed_all(999)
    c = (random.random(), np.random.rand(), torch.rand(1).item())
    assert c != a
