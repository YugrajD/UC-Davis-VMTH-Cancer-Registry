"""Apply optional carcinoma/sarcoma auxiliary supervision to final label picks."""

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
    labels: list[str]


class AuxiliaryLabelPolicy:
    def __init__(self, config: ScanConfig, labels: list[str]):
        self.config = config
        self.labels = labels

    def apply(self, *, ids: list[str], categorization: CategorizationResult) -> AuxiliaryDecision:
        auxiliary_labels = [""] * len(ids)
        if not self.config.use_auxiliary_labels:
            return AuxiliaryDecision(labels=auxiliary_labels)

        carcinoma_ids = load_anon_ids(self.config.carcinoma_csv_path, id_col=self.config.id_col)
        sarcoma_ids = load_anon_ids(self.config.sarcoma_csv_path, id_col=self.config.id_col)
        carcinoma_candidate_idx = candidate_indices_for_aux_label(self.labels, aux_label="carcinoma")
        sarcoma_candidate_idx = candidate_indices_for_aux_label(self.labels, aux_label="sarcoma")

        for row_idx, anon_id in enumerate(ids):
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

            best = best_index_with_constraint(categorization.label_scores[row_idx], candidate_idx)
            if best is None:
                continue

            best_idx, best_score = best
            categorization.final_indices[row_idx] = int(best_idx)
            categorization.final_labels[row_idx] = str(self.labels[best_idx])
            categorization.final_scores[row_idx] = float(best_score)
            categorization.methods[row_idx] = f"auxiliary_{aux_label}"

        return AuxiliaryDecision(labels=auxiliary_labels)
