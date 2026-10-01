"""Phase 1 data: report.csv cases, split partitions, silver gold terms, the gate's training universe.

Only public loaders from ml/ are used (split, silver labels, report_mapping.training.labels, taxonomy).
One report.csv row is one case. The retriever's vocabulary is the set of terms in the train partition's
silver labels; calibration gold terms outside it stay in ``gold`` (they count as retrieval misses).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import config
import io_utils
from bronze_rr import report_text
from generations.splits import load_split
from report_mapping.sections import clean_text
from report_mapping.training import labels as labels_mod
from taxonomy.taxonomy import load_labels_taxonomy

SILVER_ID = "silver-0-legacy"
PARTITIONS = ("train", "calibration", "test")


@dataclass(frozen=True)
class BronzeData:
    split_id: str
    silver_id: str
    reports: pd.DataFrame                   # raw report.csv rows (all cases)
    case_ids: list[str]                     # every report case, report order
    partition_ids: dict[str, list[str]]     # "train" | "calibration" | "test" -> report cases in it, report order
    split_cases: dict[str, frozenset[str]]  # the split's full partitions (may hold labelled cases with no report)
    has_text: frozenset[str]                # cases production writes rows for (see report_text.cases_with_text)
    gold: dict[str, frozenset[str]]         # case_id -> distinct silver terms (only cases with >= 1 term)
    retriever_ids: list[str]                # train cases with >= 1 silver term
    vocab: list[str]                        # sorted train-partition terms
    gate_ids: list[str]                     # train cases with >= 1 silver label row (the old gate's universe)
    gate_targets: np.ndarray                # (len(gate_ids),) 1.0 iff the case has >= 1 non-empty matched_term
    term_info: dict[str, tuple[str, str]]   # taxonomy term -> (group, code), first occurrence

    def ids_for(self, partition: str) -> list[str]:
        """``partition`` is a split partition name or "all" (train + calibration + test, report order)."""
        return self.case_ids if partition == "all" else self.partition_ids[partition]


def load_data(split_id: str = config.DEFAULT_SPLIT_ID, silver_id: str = SILVER_ID) -> BronzeData:
    reports = io_utils.read_csv(config.REPORT_CSV, encoding="latin-1")
    reports["case_id"] = reports["case_id"].map(clean_text)
    split = load_split(split_id)
    labels = labels_mod.load_labels_table(silver_id)
    labels["case_id"] = labels["case_id"].astype(str).str.strip()
    labels["matched_term"] = labels["matched_term"].map(clean_text)
    with_term = labels[labels["matched_term"] != ""]
    gold = {c: frozenset(t) for c, t in with_term.groupby("case_id")["matched_term"].agg(set).items()}

    case_ids = reports["case_id"].tolist()
    partition_ids = {name: [c for c in case_ids if c in getattr(split, name)] for name in PARTITIONS}
    retriever_ids = [c for c in partition_ids["train"] if c in gold]
    vocab = sorted({t for c in retriever_ids for t in gold[c]})

    labels_train = labels_mod.select_train(labels, split_id)  # also runs the leakage guards
    gate_ids = labels_mod.training_case_ids(labels_train, case_ids, split)
    gate_targets = labels_mod.gate_targets(labels_train, gate_ids)

    term_info: dict[str, tuple[str, str]] = {}
    for label in load_labels_taxonomy(str(config.LABELS_CSV)):
        term_info.setdefault(label.term, (label.group, label.code))

    return BronzeData(split_id=split_id, silver_id=silver_id, reports=reports, case_ids=case_ids,
                      partition_ids=partition_ids,
                      split_cases={name: getattr(split, name) for name in PARTITIONS}, has_text=frozenset(report_text.cases_with_text(reports)),
                      gold=gold, retriever_ids=retriever_ids, vocab=vocab, gate_ids=gate_ids,
                      gate_targets=gate_targets, term_info=term_info)


def load_uncommon_groups(path: Path = config.RR_PHASE1_UNCOMMON_GROUPS_TXT) -> frozenset[str]:
    """WP14's uncommon-groups list, read the way silver_eval reads a generation's uncommon_groups.txt."""
    text = path.read_text(encoding="utf-8")
    return frozenset(line.strip() for line in text.splitlines() if line.strip())
