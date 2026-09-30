"""Silver-eval: the cheap ruler. Bronze predictions vs a labels table on one partition.

Carries the core of the legacy ``run_evaluation.py`` (pipeline stage) and
``log_evaluation.py``: score predictions against a labels table (a silver
generation, or corrected annotations) restricted to one partition of a split,
with ``verdicts.score``, and append one line to a history CSV.

- Both tables are filtered to the partition's cases before scoring, as legacy
  ``--test-cases`` did. A case with no label rows counts as non-cancer.
- ``half`` optionally keeps only the md5 "eval" or "sweep" half of the
  partition (``generations.splits.in_sweep_half``), so the legacy split's test
  partition can be scored on the same cases as the frozen parity reference.
- The uncommon-groups list comes from the report-mapping generation that made
  the predictions; the predictions' ``generation_id`` must match it.
- Breakdowns are keyed on the *expected* group / term: a row counts toward every
  group (term) its case expects, and false positives (no expectation) are left
  out. Legacy keyed its per-group table on the predicted group and dropped
  false negatives; keying on the truth keeps misses visible.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import pandas as pd

import config
import io_utils
from diagnosis_mapping.silver import load_silver
from evaluation import verdicts
from generations.manifest import read_manifest
from generations.splits import in_sweep_half, load_split
from report_mapping.model.generation import generation_paths, resolve_generation_dir

HALVES = ("eval", "sweep")
TEXT_COLUMNS = ["diagnosis"]  # never carried into evaluation

HISTORY_FIELDS = (
    ["timestamp", "generation_id", "labels", "split_id", "partition", "total"]
    + [f for v in verdicts.VERDICTS for f in (v, f"{v}_pct")]
    + ["good_plus_slight", "good_plus_slight_pct"]
)


class SilverEvalError(Exception):
    """A silver-eval run was refused."""


def read_predictions(predictions_csv: str | Path) -> pd.DataFrame:
    return io_utils.read_csv(predictions_csv, encoding="utf-8", dtype=str, keep_default_na=False)


def generation_uncommon_groups(generation: str, predictions: pd.DataFrame) -> tuple[str, frozenset[str]]:
    """(generation_id, uncommon groups) of a report-mapping generation ("current", "candidate" or a dir).

    Refuses predictions that carry a different (or more than one) ``generation_id``:
    scoring them with another generation's uncommon list would shift verdicts silently.
    """
    directory = resolve_generation_dir(generation)
    generation_id = read_manifest(directory)["generation_id"]
    found = sorted(set(predictions["generation_id"])) if "generation_id" in predictions.columns else []
    if found != [generation_id]:
        raise SilverEvalError(f"predictions carry generation_id {found}, but generation {generation!r} "
                              f"is {generation_id!r}; pass the generation that made them")
    text = generation_paths(directory).uncommon_groups_txt.read_text(encoding="utf-8")
    return generation_id, frozenset(line.strip() for line in text.splitlines() if line.strip())


def load_labels(labels: str) -> pd.DataFrame:
    """A labels table from a CSV path, else from the silver generation of that id. Text columns dropped."""
    table = (io_utils.read_csv(labels, encoding="utf-8", dtype=str, keep_default_na=False)
             if Path(labels).is_file() else load_silver(labels))
    return table.drop(columns=TEXT_COLUMNS, errors="ignore")


def partition_case_ids(split_id: str, partition: str, half: str | None = None) -> frozenset[str]:
    ids = getattr(load_split(split_id), partition)
    if not ids:
        raise SilverEvalError(f"split {split_id!r} has no {partition!r} partition")
    if half is not None:
        ids = frozenset(c for c in ids if in_sweep_half(c) == (half == "sweep"))
    return ids


def breakdown(table: pd.DataFrame, column: str) -> pd.DataFrame:
    """Per value of ``expected_group`` / ``expected_term``: row count, verdict counts, good and G+S shares."""
    rows = table.loc[table[column].fillna("") != "", [column, "verdict"]].copy()
    rows[column] = rows[column].str.split(" | ", regex=False)
    rows = rows.explode(column)
    counts = pd.crosstab(rows[column], rows["verdict"]).reindex(columns=list(verdicts.VERDICTS), fill_value=0)
    counts.insert(0, "n", counts.sum(axis=1))
    counts["good_share"] = counts["good"] / counts["n"]
    counts["gs_share"] = (counts["good"] + counts["slightly_off"]) / counts["n"]
    return counts.sort_values("n", ascending=False)


def evaluate(predictions: pd.DataFrame, labels: pd.DataFrame, case_ids, uncommon_groups) -> dict:
    """Score ``predictions`` vs ``labels`` on ``case_ids``: summary, per-group and per-term tables."""
    table = verdicts.score(labels[labels["case_id"].isin(case_ids)],
                           predictions[predictions["case_id"].isin(case_ids)], uncommon_groups)
    return {
        "summary": verdicts.summarize(table),
        "by_group": breakdown(table, "expected_group"),
        "by_term": breakdown(table, "expected_term"),
        "table": table,
    }


def append_history(summary: dict, *, generation_id: str, labels: str, split_id: str, partition: str,
                   history_csv: str | Path | None = None) -> Path:
    """Append one line to the silver-eval history (header written on first use)."""
    path = Path(history_csv) if history_csv is not None else config.SILVER_EVAL_HISTORY_CSV
    entry = {
        "timestamp": datetime.now().isoformat(sep=" ", timespec="seconds"),
        "generation_id": generation_id, "labels": labels, "split_id": split_id, "partition": partition,
        "total": summary["total"],
        "good_plus_slight": summary["good_plus_slight"],
        "good_plus_slight_pct": round(100 * summary["good_plus_slight_share"], 2),
    }
    for verdict in verdicts.VERDICTS:
        entry[verdict] = summary[verdict]
        entry[f"{verdict}_pct"] = round(100 * summary[f"{verdict}_share"], 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    with open(path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=HISTORY_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(entry)
    return path


def run(predictions_csv: str | Path, labels: str, split_id: str, partition: str, *,
        generation: str = "current", half: str | None = None, history_csv: str | Path | None = None) -> dict:
    """Load, score, and append the history line. ``labels`` is a silver_id or a labels CSV path."""
    predictions = read_predictions(predictions_csv)
    generation_id, uncommon_groups = generation_uncommon_groups(generation, predictions)
    case_ids = partition_case_ids(split_id, partition, half)
    result = evaluate(predictions, load_labels(labels), case_ids, uncommon_groups)
    partition_label = partition if half is None else f"{partition}:{half}-half"
    result["history_csv"] = append_history(result["summary"], generation_id=generation_id, labels=labels,
                                           split_id=split_id, partition=partition_label, history_csv=history_csv)
    result.update(generation_id=generation_id, partition=partition_label, n_cases=len(case_ids))
    return result
