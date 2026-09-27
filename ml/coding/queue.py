"""The review queue: cases the specialist needs to look at.

Three reasons, per icd-mapping-strategy.md ("Coding a case" / "Bronze never
overrides silver"):

- ``vague_silver`` — the case has diagnosis rows and at least one is vague
  (``coding.rule.case_is_vague``).
- ``low_conf_bronze`` — the case has no diagnosis rows at all, and bronze's
  own review gate (``coding.combine.bronze_case_is_low_confidence``) flags it.
  This also covers an ``unidentified_cancer`` bronze case (the gate passed but
  no label resolved): ``coding.combine`` refuses to code it as ``NO_CANCER``,
  and its score is always 0.0, so the low-confidence check picks it up here.
- ``no_evidence`` — the case has no gold, no diagnosis row and no bronze
  prediction at all (WP9 fix 2: e.g. an upload whose report never made it
  into ``report.csv``). ``coding.combine`` has nothing to code for such a
  case either, so without this reason it would silently vanish from both
  the combined-codes table and the review queue. Always the lowest priority
  (``NO_EVIDENCE_PRIORITY``, below any real bronze probability): there is no
  signal at all to rank it by, and it must never crowd out a case bronze or
  silver actually ran on.

**Cases that already have gold are skipped** (any origin) — per WP7's brief,
a case already resolved by the specialist has nothing left to queue.

**Coverage invariant**: every case_id in ``split.train ∪ split.calibration ∪
split.test ∪ silver ∪ bronze`` ends up either coded (``coding.combine``) or
queued (here) — never neither. See ``test_combine.py``'s
``test_every_case_is_coded_or_queued`` for the cross-module check.

**Priority is bronze's case-presence probability, descending**: a vague case
where bronze confidently says cancer goes first, since that is the case most
likely to be a real miss. Bronze runs on every case (icd-mapping-strategy.md:
"The report mapping runs on every case, so bronze always exists for
comparison."), so this applies to ``vague_silver`` rows too, not only
``low_conf_bronze`` ones.

**Ordering: partition=test items sort after every train/calibration item**
(WP9 fix 3), priority-descending within each of those two blocks. A queue
item under review becomes gold, and ``eval_batch.generate_batch`` excludes
every case that already has *any* gold from its sampling pool regardless of
origin — so working through test-partition queue items depletes them from
future gold-eval draws before their weakest-silver siblings get counted.
Draining train/calibration first delays that depletion as long as possible.
A case with no historical split membership (``partition == "none"``) is
treated as non-test here, since nothing says it will ever reach test.

**Partition** is read from the split: ``train``, ``calibration`` or ``test``
for a case that's in one of those partitions, else ``none`` (an upload with no
historical split membership). **A vague *test*-partition case can be queued
and reviewed like any other** — the strategy explicitly allows this — but its
resulting gold stays on the evaluation side: ``manual_audit.gold.gold_train``
only ever draws train-partition cases, so a test-side review-queue gold row
never becomes gold-train, and it is not gold-eval either (only ``eval_batch``/
``random_slice`` origins are). It simply measures nothing and trains nothing
until a future WP defines a use for it; this module does not special-case it
beyond stamping its partition as ``test`` like any other test case (and,
as above, sorting it after every non-test item).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import config
import io_utils
from coding.combine import bronze_case_is_low_confidence, load_bronze_predictions, resolve_generation_id
from coding.rule import case_is_vague
from diagnosis_mapping.silver import load_silver
from generations.splits import load_split
from manual_audit.gold import load_gold

REVIEW_QUEUE_COLUMNS = ["case_id", "reason", "priority", "partition", "silver_generation", "bronze_generation"]

VAGUE_SILVER = "vague_silver"
LOW_CONF_BRONZE = "low_conf_bronze"
NO_EVIDENCE = "no_evidence"
# Below any real bronze case_presence_prob (which is in [0, 1]), so a
# no_evidence row always sorts last within its partition block.
NO_EVIDENCE_PRIORITY = "-1.0000"


def _case_presence_prob(case_rows: pd.DataFrame) -> str:
    """The case's bronze case_presence_prob, as the predictions CSV already
    formats it (same value on every diagnosis_index row of that case)."""
    return case_rows.iloc[0]["case_presence_prob"] or "0.0000"


def _partition_of(case_id: str, split) -> str:
    if case_id in split.train:
        return "train"
    if case_id in split.calibration:
        return "calibration"
    if case_id in split.test:
        return "test"
    return "none"


def build_review_queue(
    silver_id: str,
    split_id: str,
    predictions_csv: str | Path,
    *,
    generation_id: str | None = None,
) -> pd.DataFrame:
    """The review queue for ``silver_id`` (diagnosis rows) x ``predictions_csv``
    (bronze), skipping cases with any gold. Ordered with every test-partition
    item after every train/calibration item, priority-descending within each
    block (see the module docstring)."""
    silver = load_silver(silver_id)
    bronze = load_bronze_predictions(predictions_csv)
    bronze_generation_id = resolve_generation_id(bronze, generation_id)
    gold_case_ids = set(load_gold()["case_id"])
    split = load_split(split_id)

    silver_by_case = {cid: g for cid, g in silver.groupby("case_id")}
    bronze_by_case = {cid: g for cid, g in bronze.groupby("case_id")}

    rows: list[dict] = []
    # Every case accounted for by coding or queueing so far, so the coverage
    # sweep below (no_evidence) can tell what's left uncovered.
    covered: set[str] = set(gold_case_ids)  # gold cases are coded, never queued
    for case_id, case_rows in silver_by_case.items():
        if case_id in gold_case_ids:
            continue
        if not case_is_vague(case_rows):
            covered.add(case_id)  # decisive silver -> coded (coding.combine), not queued
            continue
        bronze_rows = bronze_by_case.get(case_id)
        priority = _case_presence_prob(bronze_rows) if bronze_rows is not None else "0.0000"
        rows.append({
            "case_id": case_id, "reason": VAGUE_SILVER, "priority": priority,
            "partition": _partition_of(case_id, split),
            "silver_generation": silver_id, "bronze_generation": bronze_generation_id,
        })
        covered.add(case_id)

    diagnosed_cases = set(silver_by_case)
    for case_id, case_rows in bronze_by_case.items():
        if case_id in gold_case_ids or case_id in diagnosed_cases:
            continue
        # A bronze-only case always gets a combined row (coding.combine) except
        # an unidentified_cancer one, which is always low-confidence too (see
        # coding.combine._bronze_case_codes) — either way it's covered.
        covered.add(case_id)
        if not bronze_case_is_low_confidence(case_rows):
            continue
        rows.append({
            "case_id": case_id, "reason": LOW_CONF_BRONZE, "priority": _case_presence_prob(case_rows),
            "partition": _partition_of(case_id, split),
            "silver_generation": silver_id, "bronze_generation": bronze_generation_id,
        })

    # Coverage invariant (WP9 fix 2): a case in the split ∪ silver ∪ bronze
    # universe with no gold, no diagnosis row and no bronze prediction at all
    # falls through both loops above and coding.combine entirely — queue it
    # here so it is never silently dropped.
    universe = split.train | split.calibration | split.test | set(silver_by_case) | set(bronze_by_case)
    for case_id in sorted(universe - covered):
        rows.append({
            "case_id": case_id, "reason": NO_EVIDENCE, "priority": NO_EVIDENCE_PRIORITY,
            "partition": _partition_of(case_id, split),
            "silver_generation": silver_id, "bronze_generation": bronze_generation_id,
        })

    df = pd.DataFrame(rows, columns=REVIEW_QUEUE_COLUMNS)
    if df.empty:
        return df
    is_test = (df["partition"] == "test").astype(int)
    priority = df["priority"].astype(float)
    order = pd.DataFrame({"is_test": is_test, "priority": priority}).sort_values(
        ["is_test", "priority"], ascending=[True, False], kind="stable",
    ).index
    return df.loc[order].reset_index(drop=True)


def write_review_queue(
    silver_id: str, split_id: str, predictions_csv: str | Path, *,
    generation_id: str | None = None, out_csv: str | Path | None = None,
) -> Path:
    out_path = Path(out_csv) if out_csv is not None else config.REVIEW_QUEUE_CSV
    df = build_review_queue(silver_id, split_id, predictions_csv, generation_id=generation_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    io_utils.write_csv(df, out_path)
    return out_path
