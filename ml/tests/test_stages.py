"""report_mapping/inference/stages.py: one unit test per stage rule.

Uses small stub heads (fixed scores, no real weights) so tail-cut / threshold /
fallback logic is tested in isolation from actual model behavior.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from report_mapping.inference import stages
from taxonomy.taxonomy import TaxonomyLabel


class _FakeProbHead:
    """Stand-in for CasePresenceClassifier / GroupClassifier: predict_proba
    returns a fixed tensor regardless of input."""

    def __init__(self, probs):
        self._probs = torch.tensor(probs, dtype=torch.float32)

    def predict_proba(self, x):
        return self._probs


class _FakeLP:
    """Stand-in for LabelPresenceClassifier: score_matrix returns fixed per-label scores."""

    def __init__(self, scores):
        self._scores = torch.tensor(scores, dtype=torch.float32)

    def score_matrix(self, case_emb, label_embs):
        return self._scores.unsqueeze(0)


# ---------------------------------------------------------------------------
# Stage 1 — gate threshold
# ---------------------------------------------------------------------------


def test_run_case_presence_threshold():
    head = _FakeProbHead([0.9, 0.4, 0.8])
    mask, probs = stages.run_case_presence(head, np.zeros((3, 4), dtype=np.float32), threshold=0.5)
    assert mask.tolist() == [True, False, True]
    np.testing.assert_allclose(probs, [0.9, 0.4, 0.8], atol=1e-6)


# ---------------------------------------------------------------------------
# Stage 2 — group threshold + gate zeroing
# ---------------------------------------------------------------------------


def test_run_group_zeroes_gate_rejected_cases():
    head = _FakeProbHead([[0.9, 0.1], [0.8, 0.7]])
    gate_mask = np.array([True, False])
    probs = stages.run_group(head, np.zeros((2, 4), dtype=np.float32), gate_mask)
    np.testing.assert_allclose(probs[0], [0.9, 0.1])
    np.testing.assert_allclose(probs[1], [0.0, 0.0])


# ---------------------------------------------------------------------------
# Stage 3a — per-LP threshold + argmax fallback
# ---------------------------------------------------------------------------


def test_score_within_group_threshold_selects_all_above():
    lp = _FakeLP([0.9, 0.3, 0.6])
    selected, score_map = stages.score_within_group(
        case_embedding=np.zeros((1, 4), dtype=np.float32), label_indices=[10, 11, 12],
        label_embeddings=np.zeros((13, 4), dtype=np.float32), lp_model=lp, threshold=0.5,
    )
    assert selected == [10, 12]
    assert score_map[10] == pytest.approx(0.9)
    assert score_map[12] == pytest.approx(0.6)


def test_score_within_group_argmax_fallback_when_nothing_clears_threshold():
    lp = _FakeLP([0.1, 0.4, 0.2])
    selected, _ = stages.score_within_group(
        case_embedding=np.zeros((1, 4), dtype=np.float32), label_indices=[5, 6, 7],
        label_embeddings=np.zeros((8, 4), dtype=np.float32), lp_model=lp, threshold=0.5,
    )
    assert selected == [6]  # argmax of [0.1, 0.4, 0.2]


# ---------------------------------------------------------------------------
# categorize_cases: tail gate (K cap, prob-gap cut), group argmax fallback,
# per-group LP threshold lookup + global fallback, and the sorted-Uncommon-merge fix.
# ---------------------------------------------------------------------------


def _single_label_taxonomy(group_names: list[str]) -> tuple[list[TaxonomyLabel], list[str]]:
    labels_obj = [TaxonomyLabel(code=f"{i}/3", group=g, term=f"{g} term") for i, g in enumerate(group_names)]
    return labels_obj, [l.term for l in labels_obj]


def _categorize_one_case(*, group_probs, group_names, tail_max_predictions=2, tail_max_group_prob_gap=0.08,
                          group_threshold=0.5, lp_thresholds=None, label_presence_fallback=0.5,
                          uncommon_groups=frozenset(), taxonomy_labels=None, labels=None,
                          lp_scores_per_group=None, gate_passed=True):
    taxonomy_labels = taxonomy_labels or []
    labels = labels or []
    lp_thresholds = lp_thresholds or {}
    lp_scores_per_group = lp_scores_per_group or {}
    group_to_indices: dict[str, list[int]] = {}
    for j, tl in enumerate(taxonomy_labels):
        group_to_indices.setdefault(tl.group, []).append(j)
    label_presence_heads = {}
    for group_name, scores in lp_scores_per_group.items():
        label_presence_heads[group_name] = _FakeLP(scores)

    result = stages.categorize_cases(
        texts=["plain report text with no keyword signal"],
        lp_embeddings=np.zeros((1, 4), dtype=np.float32),
        label_embeddings=np.zeros((max(len(taxonomy_labels), 1), 4), dtype=np.float32),
        taxonomy_labels=taxonomy_labels,
        labels=labels,
        group_probs=np.array([group_probs], dtype=np.float32),
        group_names=group_names,
        group_threshold=group_threshold,
        tail_max_predictions=tail_max_predictions,
        tail_max_group_prob_gap=tail_max_group_prob_gap,
        presence_mask=np.array([gate_passed]),
        uncommon_groups=uncommon_groups,
        label_presence_heads=label_presence_heads,
        label_presence_fallback=label_presence_fallback,
        lp_thresholds=lp_thresholds,
    )
    return result


def test_tail_cap_limits_to_k():
    group_names = ["G0", "G1", "G2"]
    taxonomy_labels, labels = _single_label_taxonomy(group_names)
    result = _categorize_one_case(
        group_probs=[0.9, 0.85, 0.7], group_names=group_names, tail_max_predictions=2,
        tail_max_group_prob_gap=0.5,  # generous gap: only the K cap should bind
        taxonomy_labels=taxonomy_labels, labels=labels,
        lp_scores_per_group={"G0": [0.9], "G1": [0.9], "G2": [0.9]},
    )
    assert result.top_k_indices[0] == [0, 1]  # G2 dropped by the K=2 cap, not the gap


def test_tail_gap_drops_wide_tail():
    group_names = ["G0", "G1"]
    taxonomy_labels, labels = _single_label_taxonomy(group_names)
    result = _categorize_one_case(
        group_probs=[0.9, 0.5], group_names=group_names, tail_max_predictions=2,
        tail_max_group_prob_gap=0.08,  # 0.9 - 0.5 = 0.4 > 0.08 -> G1 dropped
        taxonomy_labels=taxonomy_labels, labels=labels,
        lp_scores_per_group={"G0": [0.9], "G1": [0.9]},
    )
    assert result.top_k_indices[0] == [0]


def test_group_argmax_fallback_when_no_group_clears_threshold():
    group_names = ["G0", "G1"]
    taxonomy_labels, labels = _single_label_taxonomy(group_names)
    result = _categorize_one_case(
        group_probs=[0.3, 0.2], group_names=group_names, group_threshold=0.5,
        taxonomy_labels=taxonomy_labels, labels=labels,
        lp_scores_per_group={"G0": [0.9], "G1": [0.9]},
    )
    assert result.methods[0] == "label_presence"
    assert result.final_labels[0] == "G0 term"  # argmax group (0.3 > 0.2), not a threshold survivor


def test_gate_rejected_case_is_uncategorized_even_with_no_predicted_group():
    group_names = ["G0"]
    taxonomy_labels, labels = _single_label_taxonomy(group_names)
    result = _categorize_one_case(
        group_probs=[0.0], group_names=group_names, group_threshold=0.5,
        taxonomy_labels=taxonomy_labels, labels=labels, gate_passed=False,
    )
    assert result.methods[0] == "low_confidence"
    assert result.final_labels[0] == "Uncategorized"


def test_per_group_lp_threshold_overrides_global_fallback():
    group_names = ["G0"]
    taxonomy_labels = [
        TaxonomyLabel(code="1/3", group="G0", term="G0 low"),
        TaxonomyLabel(code="2/3", group="G0", term="G0 high"),
    ]
    labels = [l.term for l in taxonomy_labels]
    # Global fallback 0.9 would clear neither; the per-group override 0.5 clears the second.
    result = _categorize_one_case(
        group_probs=[0.9], group_names=group_names, taxonomy_labels=taxonomy_labels, labels=labels,
        lp_scores_per_group={"G0": [0.3, 0.6]}, lp_thresholds={"G0": 0.5}, label_presence_fallback=0.9,
    )
    assert result.final_labels[0] == "G0 high"


def test_label_presence_fallback_used_when_group_missing_from_lp_thresholds():
    group_names = ["G0"]
    taxonomy_labels, labels = _single_label_taxonomy(group_names)
    result = _categorize_one_case(
        group_probs=[0.9], group_names=group_names, taxonomy_labels=taxonomy_labels, labels=labels,
        lp_scores_per_group={"G0": [0.9]}, lp_thresholds={}, label_presence_fallback=0.5,
    )
    assert result.methods[0] == "label_presence"


def test_uncommon_pool_ranks_by_confidence_descending_not_pool_order():
    """Regression test for the WP1 nondeterminism, round 2: sorting the merged
    "Uncommon" group list made the *pool* deterministic, but picking pool[0] as
    top-1 was still arbitrary (whichever group sorted first, not the best
    scoring survivor). Zeta's label scores higher than Alpha's here even though
    Alpha is the alphabetically-first (and thus pool-first) group — the winner
    must be the higher-confidence survivor regardless of pool position.
    """
    taxonomy_labels = [
        TaxonomyLabel(code="1/3", group="Zeta Group", term="Z1"),
        TaxonomyLabel(code="2/3", group="Alpha Group", term="A1"),
    ]
    labels = [l.term for l in taxonomy_labels]
    result = _categorize_one_case(
        group_probs=[0.9], group_names=["Uncommon"],
        uncommon_groups=frozenset({"Zeta Group", "Alpha Group"}),
        taxonomy_labels=taxonomy_labels, labels=labels,
        # pool order (from sorted merge) is [A1, Z1]; both clear threshold, scores favor Z1.
        lp_scores_per_group={"Uncommon": [0.6, 0.9]},
    )
    assert result.final_labels[0] == "Z1"
    assert result.top_k_indices[0] == [0, 1]  # Z1 (0.9) ranked ahead of A1 (0.3)


def test_uncommon_pool_ties_break_by_term():
    """Equal LP confidence: deterministic tie-break by term (ascending), still
    independent of PYTHONHASHSEED / pool position."""
    taxonomy_labels = [
        TaxonomyLabel(code="1/3", group="Zeta Group", term="Z1"),
        TaxonomyLabel(code="2/3", group="Alpha Group", term="A1"),
    ]
    labels = [l.term for l in taxonomy_labels]
    result = _categorize_one_case(
        group_probs=[0.9], group_names=["Uncommon"],
        uncommon_groups=frozenset({"Zeta Group", "Alpha Group"}),
        taxonomy_labels=taxonomy_labels, labels=labels,
        lp_scores_per_group={"Uncommon": [0.9, 0.9]},  # tied
    )
    assert result.final_labels[0] == "A1"  # "A1" < "Z1"
    assert result.top_k_indices[0] == [1, 0]


def test_common_group_pool_order_stays_exactly_legacy_not_confidence_ranked():
    """Confidence ranking is scoped to the Uncommon bucket only: a named group's
    label_idxs come from the taxonomy file's fixed order and were never
    hash-dependent, so legacy's positional pool order (not confidence) still
    decides ties there — changing it would diverge from legacy for no
    determinism benefit."""
    taxonomy_labels = [
        TaxonomyLabel(code="1/3", group="G0", term="Low"),
        TaxonomyLabel(code="2/3", group="G0", term="High"),
    ]
    labels = [l.term for l in taxonomy_labels]
    result = _categorize_one_case(
        group_probs=[0.9], group_names=["G0"], taxonomy_labels=taxonomy_labels, labels=labels,
        # "Low" scores higher than "High" but sits first in taxonomy (pool) order.
        lp_scores_per_group={"G0": [0.95, 0.6]},
    )
    assert result.final_labels[0] == "Low"  # positional (pool) order wins, not confidence
    assert result.top_k_indices[0] == [0, 1]


def test_empty_text_case_produces_empty_method():
    result = stages.categorize_cases(
        texts=[""],
        lp_embeddings=np.zeros((1, 4), dtype=np.float32),
        label_embeddings=np.zeros((1, 4), dtype=np.float32),
        taxonomy_labels=[], labels=[],
        group_probs=np.zeros((1, 1), dtype=np.float32),
        group_names=["G0"], group_threshold=0.5, tail_max_predictions=2, tail_max_group_prob_gap=0.08,
        presence_mask=np.array([True]), uncommon_groups=frozenset(), label_presence_heads={},
        label_presence_fallback=0.5, lp_thresholds={},
    )
    assert result.methods == ["empty"]
    assert result.top_k_indices == [[]]
