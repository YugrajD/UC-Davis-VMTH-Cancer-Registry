"""Phase 0 data: one report.csv row = one case; gold = the distinct silver terms of the case.

Only public loaders from ml/ are used (split, silver, taxonomy, io_utils). The term vocabulary is
the set of terms in the train partition's silver labels; calibration gold terms outside it stay in
the gold sets (they count as retrieval misses) and are reported as out-of-vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

import config
import io_utils
from generations.splits import load_split
from report_mapping.sections import clean_text
from report_mapping.training import labels as labels_mod
from report_mapping.training import recipe
from taxonomy.taxonomy import load_labels_taxonomy

SILVER_ID = "silver-0-legacy"


@dataclass(frozen=True)
class Phase0Data:
    reports: pd.DataFrame             # raw report.csv rows (all cases)
    train_ids: list[str]              # train-partition cases with >= 1 silver term, report order
    calibration_ids: list[str]        # ALL calibration cases that have a report, report order
    gold: dict[str, frozenset[str]]   # case_id -> distinct silver terms (only cases with >= 1 term)
    vocab: list[str]                  # sorted train-partition terms
    train_freq: dict[str, int]        # term -> number of train cases carrying it
    term_group: dict[str, str]        # taxonomy term -> group (first occurrence)
    term_code: dict[str, str]         # taxonomy term -> code (first occurrence)
    uncommon_groups: frozenset[str]   # groups the production group head merges into "Uncommon"

    def calibration_cancer_ids(self) -> list[str]:
        return [c for c in self.calibration_ids if c in self.gold]


def load_phase0_data(split_id: str = config.DEFAULT_SPLIT_ID, silver_id: str = SILVER_ID) -> Phase0Data:
    reports = io_utils.read_csv(config.REPORT_CSV, encoding="latin-1")
    reports["case_id"] = reports["case_id"].map(clean_text)
    split = load_split(split_id)
    labels = labels_mod.load_labels_table(silver_id)
    labels["case_id"] = labels["case_id"].astype(str).str.strip()
    labels["matched_term"] = labels["matched_term"].map(clean_text)
    with_term = labels[labels["matched_term"] != ""]
    gold = {c: frozenset(t) for c, t in with_term.groupby("case_id")["matched_term"].agg(set).items()}

    in_report = reports["case_id"].tolist()
    train_ids = [c for c in in_report if c in split.train and c in gold]
    calibration_ids = [c for c in in_report if c in split.calibration]

    train_freq: dict[str, int] = {}
    for case_id in train_ids:
        for term in gold[case_id]:
            train_freq[term] = train_freq.get(term, 0) + 1

    taxonomy = load_labels_taxonomy(str(config.LABELS_CSV))
    term_group, term_code = {}, {}
    for label in taxonomy:
        term_group.setdefault(label.term, label.group)
        term_code.setdefault(label.term, label.code)

    # Same universe the production group head uses (train cases that have a labels row).
    group_ids = labels_mod.training_case_ids(labels[labels["case_id"].isin(split.train)], in_report, split)
    targets = labels_mod.group_targets(
        labels[labels["case_id"].isin(split.train)], group_ids,
        uncommon_threshold=recipe.GROUP.uncommon_threshold, forced_uncommon=recipe.GROUP.excluded_groups)

    return Phase0Data(reports=reports, train_ids=train_ids, calibration_ids=calibration_ids, gold=gold,
                      vocab=sorted(train_freq), train_freq=train_freq, term_group=term_group,
                      term_code=term_code, uncommon_groups=frozenset(targets.uncommon_groups))
