"""Corrected annotations: the report mapping's training labels.

Per icd-mapping-strategy.md ("Improving the methods") and ml-rewrite-plan.md's
Artefacts: per case, gold where it exists, silver elsewhere — **train
partition only**. Gold-train (``manual_audit.gold.gold_train``, origin in
review_queue/report_mapping_audit, restricted to the split's train cases) replaces a case's silver
rows entirely, whether or not that case was actually vague; a vague case
*without* gold-train is excluded outright (its silver rows are not usable
training signal and there is nothing to replace them with). A decisive silver
case with no gold-train keeps its silver rows as-is (a non-cancer row already
carries an empty ``matched_term``).

Gold-eval (``eval_batch``/``random_slice`` origin) and the row-level audit
store are never read here at all — the only gold source touched is
``gold_train()``, which is already restricted to the gold-train origins and
to train-partition cases, so an eval-side queue-gold row (a review_queue row
that happens to land on a calibration/test case) structurally cannot appear
(WP3 -> WP9: "the corrected-annotations builder must itself ensure that
audit-store codes never enter silver rows" — satisfied the same way: this
module never opens the audit store).

Output is the labels-table shape (``case_id, matched_term, matched_group,
matched_code``, ``diagnosis_mapping.silver``'s ``ANNOTATION_COLUMNS`` subset
that ``report_mapping.training.labels`` reads) plus ``label_source``
(gold|silver) and two constant lineage columns stamped on every row:
``silver_generation`` (the silver_id this run read) and ``gold_snapshot``
(``manual_audit.gold.gold_snapshot_hash`` for this split) — the same pair a
report-mapping generation's manifest records as its parents.

**Write target.** Written to ``config.CORRECTED_ANNOTATIONS_CSV`` as a plain
file, not a versioned generation directory: ``generations.guards.check_all``
reads corrected annotations from that exact fixed path (it is not a
parameter), so writing anywhere else would make ``check_corrected_sources``
silently skip validating this output. There is no conflict with the
"generations are immutable" convention either — corrected annotations are a
reproducible, regenerate-on-demand derived artefact (recomputed whenever
silver or gold-train changes), not a lineage-tracked generation of their own;
the two lineage columns above already record what produced a given file.

``write_corrected_annotations`` is crash-safe (WP9 fix 5): it runs the pure
guards that apply to this output — ``guards.check_labels_train_only``,
``check_eval_queue_gold_not_trained``, ``check_corrected_sources`` (the public
``generations.guards`` API; ``check_disjoint``/``check_gold_origins``/etc. are
about the split and gold store themselves, not this file, so ``check_all``
isn't called here) — on the **in-memory** frame first, before touching disk at
all. Only once those pass does it write to a temp file beside the target and
``os.replace`` it into place. A guard failure therefore never creates or
modifies the target path — there is no write-then-rollback window where a
crash could leave a half-written or invalid file where a caller expects a
valid (or absent) one. A caller passing a non-default ``out_csv`` still gets
the same guard-then-atomic-write treatment on whatever it is writing.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

import config
import io_utils
from coding.rule import case_is_vague
from diagnosis_mapping.silver import load_silver
from generations import guards
from generations.splits import load_split
from manual_audit.gold import NO_CANCER, gold_snapshot_hash, gold_train, load_gold

CORRECTED_COLUMNS = [
    "case_id", "matched_term", "matched_group", "matched_code",
    "label_source", "silver_generation", "gold_snapshot",
]


def _gold_rows(case_id: str, rows: pd.DataFrame) -> list[dict]:
    return [
        {
            "case_id": case_id, "matched_term": r["term"], "matched_group": r["group"],
            # A gold NO_CANCER row carries the literal sentinel in "code" (the
            # combined-codes table's own convention); the labels-table
            # contract instead reads an empty matched_term as non-cancer (see
            # report_mapping.training.labels), same as a silver non-cancer row
            # (which already has matched_code == ""). Normalise here so gold
            # and silver non-cancer rows are byte-identical in this shape.
            "matched_code": "" if r["code"] == NO_CANCER else r["code"],
            "label_source": "gold",
        }
        for r in rows.to_dict("records")
    ]


def _silver_rows(case_id: str, rows: pd.DataFrame) -> list[dict]:
    return [
        {"case_id": case_id, "matched_term": r["matched_term"], "matched_group": r["matched_group"],
         "matched_code": r["matched_code"], "label_source": "silver"}
        for r in rows.to_dict("records")
    ]


def build_corrected_annotations(silver_id: str, split_id: str) -> pd.DataFrame:
    """The corrected-annotations table for ``split_id``'s train partition."""
    train_cases = load_split(split_id).train
    silver = load_silver(silver_id)
    silver_train = silver[silver["case_id"].astype(str).str.strip().isin(train_cases)]
    gold = gold_train(split_id)  # already train-only, gold-train origins only

    silver_by_case = {cid: g for cid, g in silver_train.groupby("case_id")}
    gold_by_case = {cid: g for cid, g in gold.groupby("case_id")}

    all_cases = sorted(set(silver_by_case) | set(gold_by_case))
    rows: list[dict] = []
    for case_id in all_cases:
        if case_id in gold_by_case:
            rows.extend(_gold_rows(case_id, gold_by_case[case_id]))
        elif case_is_vague(silver_by_case[case_id]):
            continue  # vague, without gold-train to replace it: excluded
        else:
            rows.extend(_silver_rows(case_id, silver_by_case[case_id]))

    df = pd.DataFrame(rows, columns=CORRECTED_COLUMNS[:5])
    df["silver_generation"] = silver_id
    df["gold_snapshot"] = gold_snapshot_hash(split_id)
    return df[CORRECTED_COLUMNS]


def write_corrected_annotations(
    silver_id: str, split_id: str, *, out_csv: str | Path | None = None,
) -> Path:
    """Build the table, guard it in memory, then write it atomically (see the
    module docstring's "Write target" — this is crash-safe: a guard failure
    never touches ``out_path`` at all)."""
    out_path = Path(out_csv) if out_csv is not None else config.CORRECTED_ANNOTATIONS_CSV
    df = build_corrected_annotations(silver_id, split_id)

    split = load_split(split_id)
    gold = load_gold()
    guards.check_labels_train_only(df, split)
    guards.check_eval_queue_gold_not_trained(gold, df, split)
    guards.check_corrected_sources(df, gold)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    io_utils.write_csv(df, tmp_path)
    os.replace(tmp_path, out_path)
    return out_path
