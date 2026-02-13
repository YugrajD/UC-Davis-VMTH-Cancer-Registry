"""Apply optional carcinoma/sarcoma auxiliary supervision to final label picks.

When enabled (--use-auxiliary-labels), this module overrides the embedding-based
prediction for patients that appear in known carcinoma or sarcoma patient lists.
Instead of using the overall best-matching label, the prediction is constrained
to only taxonomy terms that contain "carcinoma" or "sarcoma" in their name,
and the highest-scoring term within that subset is chosen.

This acts as a form of semi-supervised correction: if we already know a patient
has a carcinoma, we force the predicted label to be some type of carcinoma
(rather than, say, an unrelated tumor that happened to score slightly higher).
"""

from __future__ import annotations

from dataclasses import dataclass

from labels.auxiliary import (
    best_index_with_constraint,
    candidate_indices_for_aux_label,
    load_anon_ids,
)
from .categorization import CategorizationResult
from .types import ScanConfig


@dataclass(frozen=True)
class AuxiliaryDecision:
    """Per-row auxiliary label annotations (empty string if no override)."""
    labels: list[str]


class AuxiliaryLabelPolicy:
    def __init__(self, config: ScanConfig, labels: list[str]):
        self.config = config
        self.labels = labels

    def apply(self, *, ids: list[str], categorization: CategorizationResult) -> AuxiliaryDecision:
        """Optionally override predictions for known carcinoma/sarcoma patients.

        For each row:
          - If the patient ID is in the carcinoma list: restrict candidates to
            labels containing "carcinoma" and pick the highest-scoring one.
          - Same logic for sarcoma.
          - If the patient is in BOTH lists: mark as "conflict" and keep the
            original embedding-based prediction (no override).
          - If the patient is in neither list or the text was empty: no change.
        """
        auxiliary_labels = [""] * len(ids)
        if not self.config.use_auxiliary_labels:
            return AuxiliaryDecision(labels=auxiliary_labels)

        # Load the sets of patient IDs known to have carcinoma / sarcoma.
        carcinoma_ids = load_anon_ids(self.config.carcinoma_csv_path, id_col=self.config.id_col)
        sarcoma_ids = load_anon_ids(self.config.sarcoma_csv_path, id_col=self.config.id_col)

        # Find which taxonomy label indices contain "carcinoma" or "sarcoma".
        carcinoma_candidate_idx = candidate_indices_for_aux_label(self.labels, aux_label="carcinoma")
        sarcoma_candidate_idx = candidate_indices_for_aux_label(self.labels, aux_label="sarcoma")

        for row_idx, anon_id in enumerate(ids):
            if categorization.methods[row_idx] == "empty":
                # Keep truly blank diagnosis rows blank, even if anon_id is in aux sets.
                continue

            is_carcinoma = anon_id in carcinoma_ids
            is_sarcoma = anon_id in sarcoma_ids
            if is_carcinoma and is_sarcoma:
                auxiliary_labels[row_idx] = "conflict"
                continue
            if not (is_carcinoma or is_sarcoma):
                continue

            aux_label = "carcinoma" if is_carcinoma else "sarcoma"
            candidate_idx = carcinoma_candidate_idx if is_carcinoma else sarcoma_candidate_idx
            auxiliary_labels[row_idx] = aux_label
            if not candidate_idx:
                continue

            # From the full similarity score row, pick the best label within
            # the constrained candidate set.
            best = best_index_with_constraint(categorization.label_scores[row_idx], candidate_idx)
            if best is None:
                continue

            # Override the final prediction with the constrained best match.
            best_idx, best_score = best
            categorization.final_indices[row_idx] = int(best_idx)
            categorization.final_labels[row_idx] = str(self.labels[best_idx])
            categorization.final_scores[row_idx] = float(best_score)
            categorization.methods[row_idx] = f"auxiliary_{aux_label}"

        return AuxiliaryDecision(labels=auxiliary_labels)
