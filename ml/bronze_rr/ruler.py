"""The ruler check: score the WP14 baseline's existing predictions with WP14's real uncommon-groups list.

Reads predictions that already exist; no model runs. On the three-way-v1 test partition this must reproduce
the WP14 numbers recorded in RR_HANDOFF.md ("Baseline"), which proves the uncommon list and the scoring path.
"""

from __future__ import annotations

from pathlib import Path

from bronze_rr.data import SILVER_ID, load_uncommon_groups
from evaluation import silver_eval, verdicts

WP14_TEST = {"total": 4299, "good": 46.27, "slightly_off": 15.86, "completely_off": 12.84,
             "false_positive": 2.35, "false_negative": 22.68}


def score_partition(predictions_csv: Path, split_id: str, partition: str) -> dict:
    """verdicts.summarize of the predictions vs silver on one partition, with WP14's uncommon groups."""
    return silver_eval.evaluate(silver_eval.read_predictions(predictions_csv), silver_eval.load_labels(SILVER_ID),
                                silver_eval.partition_case_ids(split_id, partition), load_uncommon_groups())["summary"]


def matches_wp14_test(summary: dict) -> bool:
    return summary["total"] == WP14_TEST["total"] and all(
        round(100 * summary[f"{v}_share"], 2) == WP14_TEST[v] for v in verdicts.VERDICTS)


def summary_row(name: str, summary: dict) -> str:
    """One markdown / console table row: rows, Good, Slight, CO, FP, FN, G+S shares (%) and the counts."""
    shares = " | ".join(f"{100 * summary[f'{v}_share']:.2f}" for v in verdicts.VERDICTS)
    counts = "/".join(str(summary[v]) for v in verdicts.VERDICTS)
    return f"| {name} | {summary['total']} | {shares} | {100 * summary['good_plus_slight_share']:.2f} | {counts} |"


SUMMARY_HEADER = ("| system | rows | Good | Slight | CO | FP | FN | G+S | counts good/slight/co/fp/fn |\n"
                  "|---|---|---|---|---|---|---|---|---|")
