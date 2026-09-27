"""Stage dispatch: gate -> group (+ tail gate) -> per-group label-presence -> keyword correction.

Behaviour-equivalent port of ``production/petbert_pipeline/stages/{__init__,
case_presence_classifier, group_classifier, label_presence_classifier}.py``,
minus the dropped demographics path (Decisions: ``features/`` is dropped) and
the never-enabled ``--rerank-stage3`` flag (production never sets it; the
group-argmax fallback it could partially disable is likewise always on in
production, so it is hardcoded here rather than exposed as a flag with one
value ever used — CLAUDE.md: "No params for hypothetical futures"). All
thresholds are mandatory parameters, resolved by the caller from the
generation's ``thresholds.json`` + ``lp_thresholds.json`` — never a code
default here.

**Fixes one legacy nondeterminism** (WP1 finding, confirmed by this WP): the
merged "Uncommon" bucket's label order used to come from iterating a
``frozenset`` of group names, whose order depends on ``PYTHONHASHSEED``.
Per-label LP scores never depended on that order — each (case, label) pair is
scored independently by the LP head, and a seed-0-vs-seed-1 legacy rerun on
the full corpus showed 0 mismatches in ``case_presence_prob``/``group_prob``
(max |diff| 0.0). But the dispatcher's *winner selection* was positional: the
first surviving label in pool order became the top-1 pick, not the
highest-scoring one. So the hash-dependent group order flipped which
survivor won top-1 on 279/68,800 legacy prediction rows (98 cases) across
seeds, despite every underlying score being identical.

Two changes fix this, both scoped to the "Uncommon" bucket specifically — a
named group's ``label_idxs`` come from a fixed taxonomy-file order and were
never hash-dependent, so their legacy pool order is left exactly alone: (1)
the merged Uncommon group list is built from ``sorted(uncommon_groups)``, not
raw frozenset iteration, so which labels are even in the pool no longer
depends on ``PYTHONHASHSEED``; (2) survivors within the Uncommon pool are
ranked by LP confidence, descending, ties broken by term — so the top-1 pick
is the best-scoring survivor, not merely whichever group happened to sort
first. Which groups get tried, in what order, the tail gate, and every named
group's pool order all stay exactly as legacy. See the WP4 report for the full
comparison against seed-1 and seed-0 legacy reruns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from report_mapping.inference import keyword_correction
from taxonomy.taxonomy import TaxonomyLabel

EMPTY_TERM = ""  # sentinel term for an empty-text row (no prediction rows are written for it)


@dataclass
class CategorizationResult:
    final_labels: list[str]              # chosen term, "Unidentified Cancer", "Uncategorized", or ""
    final_indices: list[int]             # index into labels (-1 if empty)
    final_scores: list[float]
    methods: list[str]                   # "label_presence" | "lipoma_rescue" | "low_confidence"
                                          # | "unidentified_cancer" | "empty"
    top_k_indices: list[list[int]]       # per row: up to tail_max_predictions label indices
    top_k_scores: list[list[float]]
    top_k_methods: list[list[str]]
    top_k_group_probs: list[list[float]]


def run_case_presence(case_presence_head, embeddings: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Stage 1 gate. Returns (gate_mask, cancer_probs); cases below ``threshold``
    don't reach Stage 2+."""
    probs = case_presence_head.predict_proba(torch.from_numpy(embeddings)).numpy().astype(np.float32, copy=False)
    return probs >= threshold, probs


def run_group(group_head, concat_embeddings: np.ndarray, gate_mask: np.ndarray) -> np.ndarray:
    """Stage 2. Per-group sigmoid probabilities, zeroed for gate-rejected cases."""
    probs = group_head.predict_proba(torch.from_numpy(concat_embeddings)).numpy()
    probs[~gate_mask] = 0.0
    return probs


def score_within_group(
    *, case_embedding: np.ndarray, label_indices: list[int], label_embeddings: np.ndarray,
    lp_model, threshold: float,
) -> tuple[list[int], dict[int, float]]:
    """Stage 3a for one case within one group. Argmax fallback when nothing clears ``threshold``."""
    case_emb_t = torch.from_numpy(case_embedding)
    group_embs_t = torch.from_numpy(label_embeddings[label_indices])
    lp_probs = lp_model.score_matrix(case_emb_t, group_embs_t)[0].numpy()
    selected_within = [j for j, p in enumerate(lp_probs) if p >= threshold]
    if not selected_within:
        selected_within = [int(np.argmax(lp_probs))]
    selected = [label_indices[j] for j in selected_within]
    score_map = {label_indices[j]: float(lp_probs[j]) for j in selected_within}
    return selected, score_map


def categorize_cases(
    *,
    texts: list[str],                        # merged per-case text (sections.merged_texts)
    lp_embeddings: np.ndarray,               # (N, 2304) concat-3, feeds the gate + Stage 3a
    label_embeddings: np.ndarray,            # (M, 768)
    taxonomy_labels: list[TaxonomyLabel],
    labels: list[str],
    group_probs: np.ndarray,                 # (N, num_groups)
    group_names: list[str],
    group_threshold: float,
    tail_max_predictions: int,
    tail_max_group_prob_gap: float,
    presence_mask: np.ndarray,               # (N,) bool — Stage 1 gate mask
    uncommon_groups: frozenset[str],
    label_presence_heads: dict,
    label_presence_fallback: float,
    lp_thresholds: dict[str, float],
) -> CategorizationResult:
    """Dispatch each case to Stage 3a (LP) + Stage 3b (keyword correction), then
    the Lipoma rescue. Gate-rejected cases are the only ones that fall through
    to "Uncategorized"; a gate-passed cancer case never abstains (group-level
    argmax fallback is always on, matching production, which never disables it)."""
    group_to_label_indices: dict[str, list[int]] = {}
    for j, tl in enumerate(taxonomy_labels):
        group_to_label_indices.setdefault(tl.group, []).append(j)

    # Sorted merge (not frozenset iteration order) — see module docstring.
    uncommon_label_indices: list[int] = [
        j for g in sorted(uncommon_groups) for j in group_to_label_indices.get(g, [])
    ]

    group_name_to_idx = {g: i for i, g in enumerate(group_names)}
    lipo_group_idx = group_name_to_idx.get(keyword_correction.LIPOMA_RESCUE_GROUP)
    lipoma_nos_idx = keyword_correction.find_lipoma_rescue_index(taxonomy_labels, labels)

    final_labels: list[str] = []
    final_indices: list[int] = []
    final_scores: list[float] = []
    methods: list[str] = []
    top_k_indices: list[list[int]] = []
    top_k_scores: list[list[float]] = []
    top_k_methods: list[list[str]] = []
    top_k_group_probs: list[list[float]] = []

    for i, text in enumerate(texts):
        if not text:
            final_labels.append("")
            final_indices.append(-1)
            final_scores.append(0.0)
            methods.append("empty")
            top_k_indices.append([])
            top_k_scores.append([])
            top_k_methods.append([])
            top_k_group_probs.append([])
            continue

        case_probs = group_probs[i]
        top_group_idx = int(np.argmax(case_probs))
        predicted = sorted(
            (g for g in range(len(group_names)) if case_probs[g] >= group_threshold),
            key=lambda g: -case_probs[g],
        )

        gate_passed = bool(presence_mask[i])
        if not predicted:
            if gate_passed:
                predicted = [top_group_idx]  # group-level argmax fallback (always on)
            else:
                final_labels.append("Uncategorized")
                final_indices.append(-1)
                final_scores.append(float(case_probs[top_group_idx]))
                methods.append("low_confidence")
                top_k_indices.append([-1])
                top_k_scores.append([float(case_probs[top_group_idx])])
                top_k_methods.append(["low_confidence"])
                top_k_group_probs.append([float(case_probs[top_group_idx])])
                continue

        k_idxs: list[int] = []
        k_scores: list[float] = []
        k_meths: list[str] = []
        k_group_probs: list[float] = []
        seen_winners: set[int] = set()
        top_prob = float(case_probs[predicted[0]])

        for rank, g_idx in enumerate(predicted[:tail_max_predictions]):
            if rank > 0 and (top_prob - float(case_probs[g_idx])) > tail_max_group_prob_gap:
                break
            group_name = group_names[g_idx]
            group_prob = float(case_probs[g_idx])
            label_idxs = (
                uncommon_label_indices if group_name == "Uncommon"
                else group_to_label_indices.get(group_name, [])
            )
            if not label_idxs:
                continue

            lp_model = label_presence_heads.get(group_name)
            if lp_model is None:
                continue  # no LP head for this group: skipped, as in production

            lp_t = lp_thresholds.get(group_name, label_presence_fallback)
            lp_pool, lp_score_map = score_within_group(
                case_embedding=lp_embeddings[i : i + 1], label_indices=label_idxs,
                label_embeddings=label_embeddings, lp_model=lp_model, threshold=lp_t,
            )
            pool = keyword_correction.apply_keyword_correction(
                text=text, pool=lp_pool, taxonomy_labels=taxonomy_labels, labels=labels, group_name=group_name,
            )
            if group_name == "Uncommon":
                # Only the merged Uncommon bucket ever had a hash-order-dependent
                # pool (the frozenset merge above) — a named group's label_idxs
                # come from a fixed taxonomy-file order and were never
                # nondeterministic, so their legacy pool order is left exactly
                # alone. Within Uncommon, rank survivors by LP confidence,
                # descending, ties broken by term: the top-1 pick is the
                # best-scoring survivor, not whichever group sorted first.
                pool = sorted(pool, key=lambda idx: (-lp_score_map.get(idx, group_prob), labels[idx]))
            for best_label_idx in pool:
                if best_label_idx in seen_winners:
                    continue
                seen_winners.add(best_label_idx)
                k_idxs.append(best_label_idx)
                k_scores.append(lp_score_map.get(best_label_idx, group_prob))
                k_meths.append("label_presence")
                k_group_probs.append(group_prob)

        if not k_idxs:
            label_str = "Unidentified Cancer" if gate_passed else "Uncategorized"
            method_str = "unidentified_cancer" if gate_passed else "low_confidence"
            final_labels.append(label_str)
            final_indices.append(-1)
            final_scores.append(0.0)
            methods.append(method_str)
            top_k_indices.append([-1])
            top_k_scores.append([0.0])
            top_k_methods.append([method_str])
            top_k_group_probs.append([0.0])
            continue

        if (
            lipoma_nos_idx is not None
            and lipoma_nos_idx not in seen_winners
            and lipo_group_idx is not None
            and keyword_correction.lipoma_rescue_applies(
                text=text, lipomatous_group_prob=float(case_probs[lipo_group_idx])
            )
        ):
            k_idxs.append(lipoma_nos_idx)
            k_scores.append(float(case_probs[lipo_group_idx]))
            k_meths.append("lipoma_rescue")
            k_group_probs.append(float(case_probs[lipo_group_idx]))

        final_labels.append(labels[k_idxs[0]])
        final_indices.append(k_idxs[0])
        final_scores.append(k_scores[0])
        methods.append(k_meths[0])
        top_k_indices.append(k_idxs)
        top_k_scores.append(k_scores)
        top_k_methods.append(k_meths)
        top_k_group_probs.append(k_group_probs)

    return CategorizationResult(
        final_labels=final_labels,
        final_indices=final_indices,
        final_scores=final_scores,
        methods=methods,
        top_k_indices=top_k_indices,
        top_k_scores=top_k_scores,
        top_k_methods=top_k_methods,
        top_k_group_probs=top_k_group_probs,
    )
