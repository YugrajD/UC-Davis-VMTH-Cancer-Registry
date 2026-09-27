"""Case-level evaluation batch: the gold-eval sample.

Draws cases from the **test** partition of a split, minus the Tier-3 audit's
198 cases and any case already drawn by an earlier eval batch, stratified by
the case's silver group (plus a ``no_cancer`` stratum), and writes a review
sheet for the specialist to fill in.

**Review is not blind** (2026-09-26 user decision, superseding the original
"blind by construction" design). The specialist looks each case up in the
registry app by its ``case_id`` — the app already shows the case's predicted
code(s) — so ``record_pointer`` is simply the ``case_id`` and the sheet itself
never carries a prediction column. The ledger's ``review_mode`` column records
``app_non_blind`` for this reason, and any gold-eval report built from this
batch should state that reported accuracy may be optimistic from anchoring.

**Allocation is code-targeted, not case-targeted** (icd-mapping-strategy.md,
"Measuring accuracy": "every group making up at least ~1% of adopted codes
has at least ~30 gold codes"). For each "big" stratum — one whose own-group
codes make up at least ``big_group_share`` (default 1%) of all own-group
codes in the eligible frame — the target is enough cases to reach about
``target_codes_per_group`` (default 30) gold codes, given how many of that
stratum's own-group codes its cases average
(``target_n_h = min(N_h, ceil(target_codes_per_group / codes_per_case_h))``).
A stratum that never reaches 1% share (e.g. "Specialized gonadal neoplasms",
21 eligible cases total) gets a fixed ``rare_stratum_n`` (default 2, or its
whole population if smaller) instead of trying and failing to reach 30.
``no_cancer`` is fixed at ``no_cancer_target`` (default 100, picked from the
plan's 60-120 range; the real eligible no_cancer population is in the
thousands, so 100 is not populations-constrained).

**Batches are issued in shares of that fixed target, not the whole thing at
once** (about 865 total cases in 2-3 batches, per the plan). Each call takes
``fraction`` of whatever is *still remaining* toward each stratum's target —
not of the total target, and not a shared batch-wide total — so a stratum
that is behind gets more of this batch's cases than one already near its
target. There is deliberately no "hand back the shortfall" step: each
stratum's ``ceil(fraction * remaining_h)`` is capped only by its own
remaining eligible pool, so leftover capacity is inherently handed out
proportionally to each stratum's own remaining need, never clawed back or
redistributed in equal absolute amounts. With ``fraction=0.5`` on the full
plan this reads roughly as "half of what's left," so two such calls draw
roughly 3/4 of the total and a third mops up the rest.

**Pooling N_h across batches.** ``N_h`` (and hence the target) is computed
once from the *first* batch's reference frame (test partition minus Tier-3
cases — see "Gold exclusion" below for why gold is NOT subtracted here) and
does not shrink as later batches consume cases from it — only the *sampling
pool* per batch shrinks (by cases already ledgered or already gold). Because
the frame, the silver generation and the targets are meant to stay fixed
across a batch series, ``generate_batch`` refuses a call whose ``silver_id``,
``split_id``, ``target_codes_per_group``, ``no_cancer_target`` or
``rare_stratum_n`` differs from what an earlier batch in the series recorded
(see "Series consistency" on ``generate_batch`` itself); when the whole series
is done, use ``pooled_weights()`` below rather than any single batch's own
``sample_weight`` column, which is only that batch's local weight and is
provisional until the series is complete. The Tier-3 exclusion list itself is
read from ``config.TIER3_AUDIT_BATCH1_EXCLUSION_TXT`` (see
``_tier3_batch1_cases``); every call, in every series, reads that one file.

**Gold exclusion (WP7 fix 8).** Any case that already has a gold-store row —
regardless of that row's origin — is excluded from the *sampling pool* only,
so the eval batch never redraws a case someone has already coded. This is
deliberately *not* origin-aware (a review_queue gold case is excluded exactly
like an eval_batch or random_slice one). It does **not** touch the reference
frame that sets each stratum's ``N_h``/target (see "Pooling N_h" above): an
earlier version subtracted gold there too, so ingesting a batch's own gold
before drawing the next batch in the same series would shrink that stratum's
population and recompute a smaller target mid-series — quietly breaking both
"N_h is fixed for the series" and ``pooled_weights()``'s assumption that every
row of a stratum shares one ``N_h``. Keeping the reference frame gold-blind
fixes this: two series drawn from the same split/silver/targets get the same
``N_h`` and targets whether or not gold was ingested in between.

**Multi-group stratum rule.** A case can carry more than one silver
``matched_group`` across its diagnosis rows. This module reuses the exact
rule ``generations/splits.py`` already documents for the legacy train/test
split ("cancer cases stratified by their first matched_group in
annotation.csv"): a case's stratum is the ``matched_group`` of its lowest
``diagnosis_number`` row that has a non-empty group. A case with no
non-empty group anywhere among its rows — including a case with no diagnosis
at all, absent from the silver generation entirely — falls into the
``no_cancer`` stratum.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

import config
import io_utils
from diagnosis_mapping.silver import load_silver
from generations.splits import load_split
from manual_audit import gold, sheets
from taxonomy.taxonomy import load_labels_taxonomy

NO_CANCER_STRATUM = "no_cancer"
N_TERM_COLUMNS = 5  # 26,759 / 26,791 real cancer cases (99.9%) carry <= 5 distinct codes (max 10); see notes overflow below.
REVIEW_MODE = "app_non_blind"

EVAL_BATCH_LEDGER_FIELDS = [
    "case_id", "batch_id", "split_id", "silver_id", "seed", "stratum",
    "N_h", "n_h", "sample_weight", "review_mode", "excluded_ledgers",
    "target_codes_per_group", "no_cancer_target", "rare_stratum_n",
]

_TIER3_BATCH1_CASES_TXT = "tier3_audit_batch1_cases.txt"

_INSTRUCTIONS = """\
# Evaluation batch review — instructions

This batch is reviewed **in the registry app**, not from this sheet alone: look
up each `case_id` there to see the case's full record and its predicted
code(s). This review is deliberately **not blind** — you will see the model's
prediction before you judge the case, a tradeoff accepted so review can happen
in the app you already use. Reports built from this batch will say so, since
it may make the measured accuracy optimistic.

For each case:

- Confirm the case's true code set. Copy the exact taxonomy term into
  `term_1`, `term_2`, ... — one column per code, in any order. Copy terms
  **exactly** from `__TAXONOMY_CSV__`; there are no dropdowns, so a typo
  stops the whole import.
- **More than 5 codes is rare, but if it happens:** fill `term_1..term_5` and
  list anything beyond that in `notes`.
- **No reportable cancer:** mark `no_cancer` (e.g. `x`) and leave every
  `term_*` column blank.
- Never fill in both `term_*` and `no_cancer` for the same row — pick one.
- `record_pointer` is the case_id itself; that is what you look the case up
  by in the app.

Save as CSV (not .xlsx) when you are done.
"""


class EvalBatchError(Exception):
    """An eval-batch draw or sheet ingest was refused."""


def _load_case_ids(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _tier3_batch1_cases(explicit_path: str | Path | None) -> set[str]:
    """The 198 Tier-3 audit batch-1 cases, refusing (not silently skipping) if missing.

    Reads from ``config.TIER3_AUDIT_BATCH1_EXCLUSION_TXT`` — its own path,
    deliberately distinct from ``tier3_audit.batch_ledger_path(1)`` (that name
    is what a *new* ``tier3_audit.sample(batch=1)`` writes its own case ledger
    to, and the two must never collide).
    """
    if explicit_path is not None:
        path = Path(explicit_path)
        if not path.is_file():
            raise EvalBatchError(f"Tier-3 batch-1 cases file is missing: {path}")
        return _load_case_ids(path)

    new_path = config.TIER3_AUDIT_BATCH1_EXCLUSION_TXT
    if not new_path.is_file():
        raise EvalBatchError(
            f"Tier-3 batch-1 cases file is missing at {new_path} — refusing to draw an eval "
            f"batch without it, since it must exclude those 198 already-audited cases"
        )
    return _load_case_ids(new_path)


def _silver_strata(silver_df: pd.DataFrame, case_ids: set[str]) -> dict[str, str]:
    """Return {case_id: stratum} for every id in ``case_ids`` (see module docstring)."""
    df = silver_df[silver_df["case_id"].isin(case_ids)].copy()
    df["diagnosis_number"] = pd.to_numeric(df["diagnosis_number"], errors="coerce").fillna(0)
    strata: dict[str, str] = {case_id: NO_CANCER_STRATUM for case_id in case_ids}
    coded = df[df["matched_group"].str.strip() != ""].sort_values(["case_id", "diagnosis_number"])
    first_group = coded.groupby("case_id")["matched_group"].first()
    for case_id, group in first_group.items():
        strata[case_id] = group
    return strata


def _own_group_code_counts(silver_df: pd.DataFrame, strata: dict[str, str]) -> dict[str, int]:
    """Distinct (case, code) pairs per stratum, counting only rows in a case's OWN stratum group.

    A multi-group case's codes in its *other* groups don't count toward this
    stratum's rate — see the module docstring's "codes_per_case_h".
    """
    df = silver_df[silver_df["case_id"].isin(strata)].copy()
    if df.empty:
        return {}
    df["stratum"] = df["case_id"].map(strata)
    own = df[(df["matched_group"] == df["stratum"]) & (df["matched_code"].str.strip() != "")]
    if own.empty:
        return {}
    pairs = own.drop_duplicates(["case_id", "matched_code"])
    return pairs.groupby("stratum").size().to_dict()


def _plan_targets(
    populations: dict[str, int],
    own_code_counts: dict[str, int],
    no_cancer_population: int,
    *,
    target_codes_per_group: int,
    big_group_share: float,
    rare_stratum_n: int,
    no_cancer_target: int,
) -> dict[str, int]:
    """The fixed, series-wide target T_h per stratum (see module docstring)."""
    total_codes = sum(own_code_counts.values())
    targets = {NO_CANCER_STRATUM: min(no_cancer_population, no_cancer_target)}
    for stratum, n_h_population in populations.items():
        code_count = own_code_counts.get(stratum, 0)
        share = code_count / total_codes if total_codes else 0.0
        if code_count > 0 and share >= big_group_share:
            codes_per_case = code_count / n_h_population
            targets[stratum] = min(n_h_population, math.ceil(target_codes_per_group / codes_per_case))
        else:
            targets[stratum] = min(n_h_population, rare_stratum_n)
    return targets


def generate_batch(
    batch_id: str,
    *,
    silver_id: str,
    fraction: float,
    split_id: str | None = None,
    seed: int = 42,
    target_codes_per_group: int = 30,
    big_group_share: float = 0.01,
    rare_stratum_n: int = 2,
    no_cancer_target: int = 100,
    labels_csv: str | Path | None = None,
    ledger_csv: str | Path | None = None,
    sheet_dir: str | Path | None = None,
    tier3_batch1_ledger: str | Path | None = None,
    gold_csv: str | Path | None = None,
) -> dict:
    """Draw one batch of a (possibly multi-call) eval-batch series and write its sheet + ledger rows.

    Refuses to redraw a ``batch_id`` that already has rows in the ledger.
    Writes the sheet before the ledger, so a failure while writing the sheet
    never burns the ``batch_id`` against the write-once ledger check.

    **Series consistency.** If the ledger already has rows, ``silver_id``,
    ``split_id``, ``target_codes_per_group``, ``no_cancer_target`` and
    ``rare_stratum_n`` must match what an earlier batch in the series
    recorded — these are exactly the parameters ``_plan_targets`` uses, so a
    change mid-series would silently redefine every stratum's target between
    calls. Each is recorded on every ledger row so this is checkable without
    keeping any state outside the ledger itself.

    **Cases with existing gold are excluded from the draw**, regardless of
    that gold row's origin (no exception for a case whose only gold came from
    a different origin, e.g. review_queue) — see the module docstring's "Gold
    exclusion and its frame bias".
    """
    if not 0 < fraction <= 1:
        raise EvalBatchError(f"fraction must be in (0, 1], got {fraction!r}")
    split_id = split_id if split_id is not None else config.DEFAULT_SPLIT_ID
    labels_csv = labels_csv if labels_csv is not None else config.LABELS_CSV
    ledger_csv = Path(ledger_csv) if ledger_csv is not None else config.EVAL_BATCH_LEDGER_CSV
    sheet_dir = Path(sheet_dir) if sheet_dir is not None else config.EVAL_BATCH_DIR

    ledger = (
        io_utils.read_csv(ledger_csv, encoding="utf-8", dtype=str, keep_default_na=False)
        if ledger_csv.is_file() else pd.DataFrame(columns=EVAL_BATCH_LEDGER_FIELDS)
    )
    if len(ledger) and batch_id in set(ledger["batch_id"]):
        raise EvalBatchError(f"batch_id {batch_id!r} already has rows in {ledger_csv}; ledgers are write-once")

    if len(ledger):
        this_call = {
            "silver_id": silver_id, "split_id": split_id,
            "target_codes_per_group": target_codes_per_group,
            "no_cancer_target": no_cancer_target, "rare_stratum_n": rare_stratum_n,
        }
        mismatches = [
            f"{column}: series has {ledger[column].iloc[0]!r}, this call passed {str(value)!r}"
            for column, value in this_call.items()
            if ledger[column].iloc[0] != str(value)
        ]
        if mismatches:
            raise EvalBatchError(
                f"batch {batch_id!r} would change series-fixed parameter(s) partway through the "
                f"{ledger['batch_id'].iloc[0]!r}-and-later series: " + "; ".join(mismatches)
            )

    split = load_split(split_id)
    tier3_cases = _tier3_batch1_cases(tier3_batch1_ledger)
    gold_path = Path(gold_csv) if gold_csv is not None else config.GOLD_STORE_CSV
    gold_cases = (
        set(io_utils.read_csv(gold_path, encoding="utf-8", dtype=str, keep_default_na=False)["case_id"])
        if gold_path.is_file() else set()
    )
    # The reference frame the series' targets (N_h, target_h) are computed
    # from — stable across every call in the series, since it depends only on
    # the split and the Tier-3 exclusion list, neither of which changes
    # between batches (WP7 fix 8). It deliberately does NOT subtract
    # gold_cases: N_h/targets must stay fixed even as gold accumulates
    # between batches (e.g. the review queue being worked in parallel, or a
    # prior batch's own gold being ingested) — see the module docstring's
    # "Pooling N_h across batches", which depends on exactly this. The gold
    # exclusion only ever narrows the per-batch *sampling pool* below.
    reference_frame = split.test - tier3_cases

    silver_df = load_silver(silver_id)
    strata = _silver_strata(silver_df, reference_frame)
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for case_id, stratum in strata.items():
        by_stratum[stratum].append(case_id)

    populations = {s: len(ids) for s, ids in by_stratum.items() if s != NO_CANCER_STRATUM}
    no_cancer_population = len(by_stratum.get(NO_CANCER_STRATUM, []))
    own_code_counts = _own_group_code_counts(silver_df, strata)
    targets = _plan_targets(
        populations, own_code_counts, no_cancer_population,
        target_codes_per_group=target_codes_per_group, big_group_share=big_group_share,
        rare_stratum_n=rare_stratum_n, no_cancer_target=no_cancer_target,
    )

    already_drawn = set(ledger["case_id"]) if len(ledger) else set()
    drawn_so_far_by_stratum = (
        ledger.groupby("stratum")["case_id"].apply(set).to_dict() if len(ledger) else {}
    )

    rng = random.Random(seed)
    drawn: dict[str, list[str]] = {}
    # sorted(targets), not targets.items(): targets' own key order traces back
    # to a set (reference_frame) somewhere upstream, which is PYTHONHASHSEED-
    # dependent. rng is shared across every stratum's shuffle() call below, so
    # processing strata in a different order draws a different sample even
    # from the same seed — this is what pins the draw across processes.
    for stratum in sorted(targets):
        target_h = targets[stratum]
        remaining = max(target_h - len(drawn_so_far_by_stratum.get(stratum, set())), 0)
        this_batch_n = math.ceil(fraction * remaining) if remaining else 0
        # The sampling pool (unlike the reference frame above) does exclude
        # gold_cases: a case someone has already coded must never be redrawn.
        pool = sorted(set(by_stratum.get(stratum, [])) - already_drawn - gold_cases)
        rng.shuffle(pool)
        drawn[stratum] = sorted(pool[:min(this_batch_n, len(pool))])

    excluded_sources = []
    if already_drawn:
        excluded_sources.append(Path(ledger_csv).name)
    if tier3_cases:
        excluded_sources.append(_TIER3_BATCH1_CASES_TXT)
    if gold_cases:
        excluded_sources.append(Path(gold_path).name)
    excluded_ledgers = ",".join(excluded_sources) if excluded_sources else "none"

    all_case_ids = sorted({c for cases in drawn.values() for c in cases})
    N_h_of = {**populations, NO_CANCER_STRATUM: no_cancer_population}
    ledger_rows = []
    for stratum in sorted(drawn):
        cases = drawn[stratum]
        if not cases:
            continue
        n_h, N_h = len(cases), N_h_of[stratum]
        weight = N_h / n_h  # this batch's local weight only — see "Pooling N_h across batches" above.
        for case_id in cases:
            ledger_rows.append({
                "case_id": case_id, "batch_id": batch_id, "split_id": split_id, "silver_id": silver_id,
                "seed": seed, "stratum": stratum, "N_h": N_h, "n_h": n_h,
                "target_codes_per_group": target_codes_per_group, "no_cancer_target": no_cancer_target,
                "rare_stratum_n": rare_stratum_n,
                "sample_weight": f"{weight:.6f}", "review_mode": REVIEW_MODE, "excluded_ledgers": excluded_ledgers,
            })

    fill_in_columns = [f"term_{i}" for i in range(1, N_TERM_COLUMNS + 1)] + ["no_cancer", "reviewer", "notes"]
    sheet_path = sheet_dir / f"{batch_id}_review.csv"
    instructions_path = sheet_dir / f"{batch_id}_instructions.md"
    taxonomy_path = sheet_dir / f"{batch_id}_taxonomy.csv"
    # record_pointer = case_id: the specialist looks the case up in the
    # registry app by case_id, which is all the pointer needs to be here.
    sheets.write_case_sheet(sheet_path, all_case_ids, {c: c for c in all_case_ids}, fill_in_columns)
    sheets.write_instructions(
        instructions_path,
        _INSTRUCTIONS.replace("__TAXONOMY_CSV__", taxonomy_path.name),
    )
    taxonomy_rows = sorted({(l.group, l.term, l.code) for l in load_labels_taxonomy(str(labels_csv))})
    sheets.write_csv(taxonomy_path, ["Group", "Term", "Code"],
                     [{"Group": g, "Term": t, "Code": c} for g, t, c in taxonomy_rows])

    # Ledger written last: if anything above raised, batch_id was never
    # recorded as used and this call can simply be retried.
    new_ledger = pd.concat([ledger, pd.DataFrame(ledger_rows, columns=EVAL_BATCH_LEDGER_FIELDS)], ignore_index=True)
    ledger_csv.parent.mkdir(parents=True, exist_ok=True)
    io_utils.write_csv(new_ledger, ledger_csv)

    return {
        "batch_id": batch_id,
        "sheet_path": sheet_path,
        "ledger_csv": ledger_csv,
        "total_cases": len(all_case_ids),
        "excluded_count": len((already_drawn | tier3_cases | gold_cases) & split.test),
        "excluded_ledgers": excluded_ledgers,
        "targets": dict(targets),
        "stratum_counts": {s: len(drawn.get(s, [])) for s in sorted(targets)},
        "stratum_populations": dict(N_h_of),
    }


def ingest_sheet(
    sheet: str | Path,
    batch_id: str,
    reviewer: str,
    *,
    split_id: str | None = None,
    labels_csv: str | Path | None = None,
    ledger_csv: str | Path | None = None,
    gold_csv: str | Path | None = None,
) -> dict:
    """Ingest a filled eval-batch sheet into the gold store, origin ``eval_batch``.

    Every sheet case must be one the ledger recorded for ``batch_id`` (and,
    defensively, still in that split's test partition), and a case_id may not
    repeat across rows. Each row must have either >=1 non-blank ``term_*`` or
    ``no_cancer`` marked, never both and never neither — a blank row aborts
    the whole ingest before anything is written, same as a bad term does.
    """
    ledger_csv = Path(ledger_csv) if ledger_csv is not None else config.EVAL_BATCH_LEDGER_CSV
    if not ledger_csv.is_file():
        raise EvalBatchError(f"no eval-batch ledger at {ledger_csv}; run generate_batch first")
    ledger = io_utils.read_csv(ledger_csv, encoding="utf-8", dtype=str, keep_default_na=False)
    batch_rows = ledger[ledger["batch_id"] == batch_id]
    if batch_rows.empty:
        raise EvalBatchError(f"batch_id {batch_id!r} has no rows in {ledger_csv}")
    batch_cases = set(batch_rows["case_id"])
    split_id = split_id if split_id is not None else batch_rows["split_id"].iloc[0]
    split = load_split(split_id)

    rows = sheets.read_csv(sheet)
    errors: list[str] = []
    gold_records: list[dict] = []
    seen_case_ids: set[str] = set()
    for idx, row in enumerate(rows, start=2):  # +1 header
        case_id = (row.get("case_id") or "").strip()
        if not case_id:
            errors.append(f"Row {idx}: case_id is blank")
            continue
        if case_id in seen_case_ids:
            errors.append(f"Row {idx}: case_id {case_id!r} appears more than once in the sheet")
            continue
        seen_case_ids.add(case_id)
        if case_id not in batch_cases:
            errors.append(f"Row {idx}: case_id {case_id!r} is not one of batch {batch_id!r}'s ledger cases")
            continue
        if case_id not in split.test:
            errors.append(f"Row {idx}: case_id {case_id!r} is not in split {split_id!r}'s test partition")
            continue

        no_cancer = (row.get("no_cancer") or "").strip().lower() in ("y", "yes", "x", "true", "1")
        terms_filled = [t for t in (
            (row.get(f"term_{i}") or "").strip() for i in range(1, N_TERM_COLUMNS + 1)
        ) if t]
        if no_cancer and terms_filled:
            errors.append(f"Row {idx} (case_id={case_id}): both no_cancer and term_* are filled in — pick one")
            continue
        if not no_cancer and not terms_filled:
            errors.append(f"Row {idx} (case_id={case_id}): neither no_cancer nor any term_* is filled in")
            continue

        if no_cancer:
            gold_records.append({"case_id": case_id, "term": gold.NO_CANCER, "origin": "eval_batch"})
        else:
            for term in terms_filled:
                gold_records.append({"case_id": case_id, "term": term, "origin": "eval_batch"})

    if errors:
        raise EvalBatchError("eval-batch sheet ingest aborted due to validation failures:\n" + "\n".join(errors))

    rows_df = pd.DataFrame(gold_records, columns=["case_id", "term", "origin"])
    return gold.ingest_gold(
        rows_df, reviewer=reviewer, source_path=sheet, labels_csv=labels_csv,
        batch_or_export_id=batch_id, out_csv=gold_csv, eval_batch_ledger_csv=ledger_csv,
    )


def pooled_weights(ledger: str | Path | pd.DataFrame) -> pd.Series:
    """The per-stratum weight pooled across every batch in an eval-batch series: ``N_h / Σ n_h``.

    ``ledger`` is the whole eval-batch ledger (a DataFrame, or a path to read
    one from) — not one batch's rows. ``N_h`` is the same for every row of a
    given stratum (fixed once, from the first batch's frame — see the module
    docstring's "Pooling N_h across batches"), so this takes it once per
    stratum and divides by the total number of cases drawn for that stratum
    across every batch in the ledger (one ledger row per case, so a stratum's
    row count *is* Σ n_h). Exists so downstream consumers (e.g. gold-eval
    scoring) don't re-derive this from the ledger themselves.
    """
    if not isinstance(ledger, pd.DataFrame):
        ledger = io_utils.read_csv(ledger, encoding="utf-8", dtype=str, keep_default_na=False)
    grouped = ledger.groupby("stratum")
    N_h = grouped["N_h"].first().astype(int)
    n_h_total = grouped.size()
    return N_h / n_h_total
