"""report_mapping/model/heads.py: save/load round-trips, state_dict compatibility.

Loading the real legacy ``.pt`` checkpoints is a test-free smoke check (per
the plan: tests never read ml/output/) — it happens naturally when
``report_mapping.model.generation.import_legacy_gen0()`` runs on real data
and calls ``load_generation`` on the result. See the WP4 report for that run.
"""

from __future__ import annotations

import torch

from report_mapping.model.heads import CasePresenceClassifier, GroupClassifier, LabelPresenceClassifier

EMB_DIM = 32
HIDDEN_DIM = 16


def test_case_presence_round_trip(tmp_path):
    model = CasePresenceClassifier(emb_dim=EMB_DIM, hidden_dim=HIDDEN_DIM)
    path = tmp_path / "case_presence.pt"
    model.save(path)
    loaded = CasePresenceClassifier.load(path)
    assert loaded.emb_dim == EMB_DIM
    x = torch.randn(3, EMB_DIM)
    torch.testing.assert_close(model.predict_proba(x), loaded.predict_proba(x))


def test_group_round_trip(tmp_path):
    group_names = ["A", "B", "Uncommon"]
    model = GroupClassifier(num_groups=len(group_names), emb_dim=EMB_DIM, hidden_dim=HIDDEN_DIM)
    path = tmp_path / "group.pt"
    model.save(path, group_names)
    loaded, loaded_names = GroupClassifier.load(path)
    assert loaded_names == group_names
    x = torch.randn(3, EMB_DIM)
    torch.testing.assert_close(model.predict_proba(x), loaded.predict_proba(x))


def test_label_presence_round_trip_learned_combine(tmp_path):
    model = LabelPresenceClassifier(emb_dim=EMB_DIM, hidden_dim=HIDDEN_DIM, n_cols=3,
                                     col_pair_mode=True, col_combine="learned")
    path = tmp_path / "lp.pt"
    model.save(path)
    loaded = LabelPresenceClassifier.load(path)
    assert loaded.n_cols == 3 and loaded.col_pair_mode is True and loaded.col_combine == "learned"
    report_emb = torch.randn(2, 3 * EMB_DIM)
    label_emb = torch.randn(4, EMB_DIM)
    torch.testing.assert_close(
        model.score_matrix(report_emb, label_emb), loaded.score_matrix(report_emb, label_emb)
    )


def test_label_presence_load_legacy_plain_state_dict(tmp_path):
    # Pre-Phase-13 checkpoints were a bare state_dict (n_cols=1, concat mode,
    # always at the real PETBERT_EMB_DIM=768) with no wrapping metadata dict;
    # load() must fall back to that exact shape with no dimension supplied.
    from report_mapping.model.heads import PETBERT_EMB_DIM

    model = LabelPresenceClassifier(emb_dim=PETBERT_EMB_DIM, hidden_dim=HIDDEN_DIM, n_cols=1, col_pair_mode=False)
    path = tmp_path / "legacy_plain.pt"
    torch.save(model.state_dict(), path)
    loaded = LabelPresenceClassifier.load(path, hidden_dim=HIDDEN_DIM)
    assert loaded.n_cols == 1 and loaded.col_pair_mode is False and loaded.emb_dim == PETBERT_EMB_DIM
    report_emb = torch.randn(2, PETBERT_EMB_DIM)
    label_emb = torch.randn(4, PETBERT_EMB_DIM)
    torch.testing.assert_close(
        model.score_matrix(report_emb, label_emb), loaded.score_matrix(report_emb, label_emb)
    )


def test_label_presence_max_combine_round_trip(tmp_path):
    model = LabelPresenceClassifier(emb_dim=EMB_DIM, hidden_dim=HIDDEN_DIM, n_cols=3,
                                     col_pair_mode=True, col_combine="max")
    path = tmp_path / "lp_max.pt"
    model.save(path)
    loaded = LabelPresenceClassifier.load(path)
    assert loaded.col_combine == "max"
    assert loaded.col_combiner is None
