"""Case-level gold ingest: the true code set for a case, from a specialist review.

One row per (case_id, code), or exactly one ``NO_CANCER`` row for a case the
specialist confirms has no reportable cancer. Every row must carry a valid
``origin`` — ``eval_batch``, ``review_queue`` or ``random_slice`` (see
``generations.guards.GOLD_ORIGINS``) — which is the only thing that later
separates gold-eval (measures) from gold-train (trains); ingest refuses any
row without one, per icd-mapping-strategy.md ("the origin tag is mandatory").

**Gold is collected as taxonomy terms, not codes** (2026-09-26 decision). A
code alone is ambiguous for 186 of the taxonomy's 534 distinct codes — always
the group is unique, but the term is not — and a blank gold term makes
``evaluation/verdicts.score`` treat a real cancer case as non-cancer, turning
correct predictions into false positives and hiding false negatives
entirely. ``rows`` therefore carries a ``term`` column (validated and
resolved via ``manual_audit.terms.resolve_term`` — case-insensitive, and
accepting ``Group: Term`` for the one term, "Papillary adenocarcinoma", that
names two groups). A ``code`` column is still accepted with no term, but only
for ``review_queue`` rows (manual case-file corrections that already know the
precise code); ``eval_batch``/``random_slice`` rows must always resolve a
term. A code-only row backfills group always, but is refused outright — for
every origin, ``review_queue`` included — if the code maps to more than one
term: a blank term is exactly the state that reads as no-cancer in a labels
table, so there is no origin for which storing one is safe. A row that gives
both a term and a code is refused if they disagree (the term wins nothing
silently).

An ``eval_batch``-origin row must also already be present in
``config.EVAL_BATCH_LEDGER_CSV``: ingest refuses a case claiming that origin
that no eval batch ever actually drew, so a hand-built ``ingest-gold`` call
can't manufacture eval_batch gold for an unsampled case.

Re-ingesting a case replaces *all* of that case's existing rows — a
specialist correcting their own review, or a later export superseding an
earlier one, should not leave stale rows behind next to the new ones — but
**never changes a case's origin**: a case already in the store keeps whatever
origin it was first ingested under, and a re-ingest attempt with a different
origin is refused (see "Origin integrity" below).

**Review-queue gold on an eval-side case belongs to neither pool** (documented
per the WP7 brief): ``gold_eval()`` only draws origin ∈ {eval_batch,
random_slice}, so a review-queue row never counts there even if it happens to
land on a calibration/test case, and ``gold_train()`` further restricts to
train-partition cases. The review queue is deliberately skewed toward cases
where silver is weakest (see icd-mapping-strategy.md), so including its gold
in an evaluation pool would bias it; ``generations.guards`` separately refuses
to let it reach training labels for such a case.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from datetime import date
from pathlib import Path

import pandas as pd

import config
import io_utils
from generations.guards import GOLD_EVAL_ORIGINS, GOLD_ORIGINS
from generations.manifest import sha256_file
from generations.splits import load_split
from manual_audit import sheets
from manual_audit.terms import TermResolutionError, build_term_index, resolve_term
from taxonomy.taxonomy import load_labels_taxonomy

NO_CANCER = "NO_CANCER"
TERM_REQUIRED_ORIGINS = frozenset({"eval_batch", "random_slice"})

GOLD_STORE_FIELDS = [
    "case_id", "code", "term", "group", "origin",
    "batch_or_export_id", "upload_period", "slice_rate",
    # reviewer / reviewed_at: who ran THIS ingest and when (today's date at
    # ingest time) — not the specialist's own review date, which this store
    # doesn't separately track. Re-ingesting a case bumps reviewed_at to that
    # ingest's date, same as it replaces the case's other fields.
    "reviewer", "reviewed_at", "source_sha256",
]


class GoldIngestError(Exception):
    """A gold-store ingest was refused."""


def _code_taxonomy_maps(labels_csv: str | Path) -> tuple[dict[str, str], dict[str, str | None]]:
    """Return (code -> group, code -> term-or-None-if-ambiguous), for code-only rows."""
    group_of: dict[str, str] = {}
    terms_of: dict[str, set[str]] = defaultdict(set)
    for label in load_labels_taxonomy(str(labels_csv)):
        group_of[label.code] = label.group
        terms_of[label.code].add(label.term)
    term_of = {code: (next(iter(terms)) if len(terms) == 1 else None) for code, terms in terms_of.items()}
    return group_of, term_of


def _source_sha256(source_path: str | Path | None) -> str:
    return sha256_file(source_path) if source_path is not None else ""


def _clean_str_column(series: pd.Series) -> pd.Series:
    """NaN as empty, not the literal string "nan" ``.astype(str)`` alone would
    produce (WP7 fix 9) — a NaN case_id would otherwise read as the 3-char
    string "nan" and slip past the blank-case_id check below."""
    return series.fillna("").astype(str).str.strip()


def _resolve_row(record: dict, term_index, group_of: dict[str, str], term_of: dict[str, str | None]) -> dict:
    """Return {code, term, group} for one incoming row, or raise GoldIngestError."""
    term_raw = (record.get("term") or "").strip()
    code_raw = (record.get("code") or "").strip()
    case_id = record.get("case_id")

    is_no_cancer_term = term_raw.upper() == NO_CANCER
    is_no_cancer_code = code_raw.upper() == NO_CANCER
    if is_no_cancer_term or is_no_cancer_code:
        other = code_raw if is_no_cancer_term else term_raw
        if other and other.upper() != NO_CANCER:
            raise GoldIngestError(
                f"case {case_id!r}: NO_CANCER cannot be combined with a real code/term in the same "
                f"row (term={term_raw!r}, code={code_raw!r})"
            )
        return {"code": NO_CANCER, "term": "", "group": ""}

    if term_raw:
        try:
            code, term, group = resolve_term(term_raw, term_index)
        except TermResolutionError as error:
            raise GoldIngestError(f"case {case_id!r}: {error}") from None
        if code_raw and code_raw != code:
            raise GoldIngestError(
                f"case {case_id!r}: term {term_raw!r} resolves to code {code!r}, which disagrees "
                f"with the given code {code_raw!r}"
            )
        return {"code": code, "term": term, "group": group}

    if code_raw:
        if record.get("origin") not in ("review_queue",):
            raise GoldIngestError(
                f"case {case_id!r}: origin {record.get('origin')!r} rows must resolve a "
                f"term, not just a code (code-only rows are only accepted for review_queue)"
            )
        if code_raw not in group_of:
            raise GoldIngestError(f"case {case_id!r}: code {code_raw!r} is not in the taxonomy")
        term = term_of.get(code_raw)
        if not term:
            # A blank term reads as no-cancer in a labels table, so this is
            # refused for every origin — review_queue included — not just
            # backfilled blank.
            raise GoldIngestError(
                f"case {case_id!r}: code {code_raw!r} maps to more than one taxonomy term, so it "
                f"cannot be stored with a code alone — supply the term instead"
            )
        return {"code": code_raw, "term": term, "group": group_of[code_raw]}

    raise GoldIngestError(f"case {case_id!r}: row has neither a term nor a code")


def ingest_gold(
    rows: pd.DataFrame,
    *,
    reviewer: str,
    source_path: str | Path | None = None,
    labels_csv: str | Path | None = None,
    batch_or_export_id: str = "",
    upload_period: str = "",
    slice_rate: str = "",
    out_csv: str | Path | None = None,
    eval_batch_ledger_csv: str | Path | None = None,
) -> dict:
    """Validate and merge ``rows`` into the gold store.

    ``rows`` needs ``case_id, origin`` and at least one of ``term``/``code``
    per row (``term`` and/or ``code`` may be the literal ``NO_CANCER``).
    Refuses: a blank case_id; an unknown/blank origin; a case whose rows mix
    more than one origin; a case whose rows mix ``NO_CANCER`` with a real
    code/term; a duplicate (case, term-or-code) row; a term/code that doesn't
    resolve; a case-only code on a non-review_queue row; an ``eval_batch``
    row for a case that no eval batch ever actually drew (checked against
    ``config.EVAL_BATCH_LEDGER_CSV`` by default, or ``eval_batch_ledger_csv``);
    and re-ingesting a case under a different origin than it already has in
    the store (see the module docstring, "Origin integrity"). Nothing is
    written unless every row validates.
    """
    labels_csv = labels_csv if labels_csv is not None else config.LABELS_CSV
    out_csv = out_csv if out_csv is not None else config.GOLD_STORE_CSV
    rows = rows.copy()

    if "case_id" not in rows.columns or "origin" not in rows.columns:
        raise GoldIngestError("gold rows are missing required column(s) 'case_id' and/or 'origin'")
    rows["case_id"] = _clean_str_column(rows["case_id"])
    rows["origin"] = _clean_str_column(rows["origin"])
    rows["term"] = _clean_str_column(rows["term"]) if "term" in rows.columns else ""
    rows["code"] = _clean_str_column(rows["code"]) if "code" in rows.columns else ""

    errors: list[str] = []

    blank_case = rows["case_id"] == ""
    if blank_case.any():
        errors.append(f"{blank_case.sum()} row(s) have a blank case_id")

    bad_origin = sorted(set(rows.loc[~rows["origin"].isin(GOLD_ORIGINS), "case_id"]))
    if bad_origin:
        errors.append(f"{len(bad_origin)} case(s) have a gold origin outside {sorted(GOLD_ORIGINS)}: {bad_origin}")

    if (rows["origin"] == "random_slice").any():
        # WP7 fix 10 (from WP6b): evaluation.gold_eval weights a random_slice
        # case by 1/slice_rate and requires slice_rate in (0, 1] — refuse a
        # missing or out-of-range value at ingest time instead of only when a
        # much later gold-eval run discovers it.
        try:
            rate = float(slice_rate)
        except (TypeError, ValueError):
            rate = None
        if rate is None or not (0 < rate <= 1):
            errors.append(
                f"origin random_slice requires slice_rate in (0, 1] for this ingest call; got {slice_rate!r}"
            )

    eval_batch_cases = set(rows.loc[rows["origin"] == "eval_batch", "case_id"])
    if eval_batch_cases:
        ledger_path = Path(eval_batch_ledger_csv) if eval_batch_ledger_csv is not None else config.EVAL_BATCH_LEDGER_CSV
        ledgered = (
            set(io_utils.read_csv(ledger_path, encoding="utf-8", dtype=str, keep_default_na=False)["case_id"])
            if ledger_path.is_file() else set()
        )
        unledgered = sorted(eval_batch_cases - ledgered)
        if unledgered:
            errors.append(
                f"{len(unledgered)} case(s) have origin eval_batch but are not in the eval-batch "
                f"ledger {ledger_path}: {unledgered}"
            )

    mixed_origin = sorted(
        case_id for case_id, group in rows.groupby("case_id")["origin"] if group.nunique() > 1
    )
    if mixed_origin:
        errors.append(f"{len(mixed_origin)} case(s) mix more than one origin: {mixed_origin}")

    def _identifier(record: dict) -> str:
        # NO_CANCER is normalised to one sentinel value regardless of which
        # column (or casing) it arrived in, so the mixed-NO_CANCER check below
        # can compare like with like — comparing an upper-cased NO_CANCER
        # against a lower-cased one previously let this check pass silently.
        term = (record.get("term") or "").strip()
        code = (record.get("code") or "").strip()
        if term.upper() == NO_CANCER or code.upper() == NO_CANCER:
            return NO_CANCER
        return term.lower() if term else code.upper()

    mixed_no_cancer, duplicates = [], []
    for case_id, group in rows.groupby("case_id"):
        idents = [_identifier(r) for r in group.to_dict("records")]
        if NO_CANCER in idents and len(set(idents)) > 1:
            mixed_no_cancer.append(case_id)
        seen: set[str] = set()
        for ident in idents:
            if ident in seen:
                duplicates.append((case_id, ident))
            seen.add(ident)
    if mixed_no_cancer:
        errors.append(f"{len(mixed_no_cancer)} case(s) have NO_CANCER alongside a real code/term: {sorted(mixed_no_cancer)}")
    if duplicates:
        errors.append(f"{len(duplicates)} duplicate (case, term/code) row(s): {duplicates[:5]}")

    if errors:
        raise GoldIngestError("gold ingest refused:\n" + "\n".join(errors))

    labels = load_labels_taxonomy(str(labels_csv))
    term_index = build_term_index(labels)
    group_of, term_of = _code_taxonomy_maps(labels_csv)

    resolved_errors: list[str] = []
    resolved_rows: list[dict] = []
    for record in rows.to_dict("records"):
        try:
            resolved = _resolve_row(record, term_index, group_of, term_of)
        except GoldIngestError as error:
            resolved_errors.append(str(error))
            continue
        resolved_rows.append({**record, **resolved})
    if resolved_errors:
        raise GoldIngestError("gold ingest refused:\n" + "\n".join(resolved_errors))

    # A term and a code that resolve to the same taxonomy code are a duplicate
    # too, even though their raw identifiers differed before resolution.
    resolved_df = pd.DataFrame(resolved_rows)
    post_duplicates: list[tuple[str, str]] = []
    post_mixed_no_cancer: list[str] = []
    for case_id, group in resolved_df.groupby("case_id"):
        codes = list(group["code"])
        if NO_CANCER in codes and len(set(codes)) > 1:
            post_mixed_no_cancer.append(case_id)
        seen: set[str] = set()
        for code in codes:
            if code in seen:
                post_duplicates.append((case_id, code))
            seen.add(code)
    if post_mixed_no_cancer:
        # Re-checked after resolution (not just on raw term/code identifiers):
        # a resolution bug that mapped two different raw inputs to NO_CANCER
        # and a real code respectively would slip past the pre-resolution
        # check above, since that check runs on unresolved values.
        raise GoldIngestError(
            f"gold ingest refused: {len(post_mixed_no_cancer)} case(s) resolve to NO_CANCER "
            f"alongside a real code: {sorted(post_mixed_no_cancer)}"
        )
    if post_duplicates:
        raise GoldIngestError(
            f"gold ingest refused: {len(post_duplicates)} case(s) have two terms that resolve to the "
            f"same code after resolution — every (case, code) below is shared by more than one input "
            f"term/code: {post_duplicates[:5]}"
        )

    try:
        existing = sheets.read_existing_store(out_csv, GOLD_STORE_FIELDS)
    except sheets.StoreSchemaError as error:
        raise GoldIngestError(str(error)) from error

    touched_cases = set(rows["case_id"])
    if len(existing):
        existing_origin = existing.groupby("case_id")["origin"].first()
        conflicts = [
            f"{case_id} (stored origin {existing_origin[case_id]!r} vs new {new_origin!r})"
            for case_id, new_origin in rows.drop_duplicates("case_id").set_index("case_id")["origin"].items()
            if case_id in existing_origin.index and existing_origin[case_id] != new_origin
        ]
        if conflicts:
            raise GoldIngestError(
                "gold ingest refused: origin would change for case(s) already in the store "
                "(a case keeps the origin it was first ingested under): " + "; ".join(conflicts)
            )

    today = date.today().isoformat()
    source_sha256 = _source_sha256(source_path)
    new_rows = [{
        "case_id": r["case_id"], "code": r["code"], "term": r["term"], "group": r["group"], "origin": r["origin"],
        "batch_or_export_id": batch_or_export_id, "upload_period": upload_period, "slice_rate": slice_rate,
        "reviewer": reviewer, "reviewed_at": today, "source_sha256": source_sha256,
    } for r in resolved_rows]

    replaced_cases = touched_cases & set(existing["case_id"]) if len(existing) else set()
    kept = existing[~existing["case_id"].isin(touched_cases)] if len(existing) else existing
    merged = pd.concat([kept, pd.DataFrame(new_rows, columns=GOLD_STORE_FIELDS)], ignore_index=True)

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    io_utils.write_csv(merged, out_path)

    return {
        "added_cases": len(touched_cases) - len(replaced_cases),
        "replaced_cases": len(replaced_cases),
        "total_rows": len(merged),
        "total_cases": merged["case_id"].nunique(),
    }


def load_gold(gold_csv: str | Path | None = None) -> pd.DataFrame:
    """The whole gold store, as-is."""
    path = Path(gold_csv) if gold_csv is not None else config.GOLD_STORE_CSV
    if not path.is_file():
        return pd.DataFrame(columns=GOLD_STORE_FIELDS)
    return io_utils.read_csv(path, encoding="utf-8", dtype=str, keep_default_na=False)


def gold_eval(gold_csv: str | Path | None = None, split_id: str | None = None) -> pd.DataFrame:
    """Gold used only to measure: origin ∈ {eval_batch, random_slice}.

    Restricts to rows that are either in ``split_id``'s test partition
    (default ``config.DEFAULT_SPLIT_ID``), or a ``random_slice`` row for a
    case outside every historical partition entirely — a random-slice case
    can be an upload with no historical split membership at all, so it is not
    held to "must be in test" the way a historical eval_batch row is.
    """
    split_id = split_id if split_id is not None else config.DEFAULT_SPLIT_ID
    gold = load_gold(gold_csv)
    ge = gold[gold["origin"].isin(GOLD_EVAL_ORIGINS)]
    split = load_split(split_id)
    historical = split.train | split.calibration | split.test
    in_test = ge["case_id"].isin(split.test)
    upload_random_slice = (ge["origin"] == "random_slice") & ~ge["case_id"].isin(historical)
    return ge[in_test | upload_random_slice].reset_index(drop=True)


def gold_train(
    split_id: str | None = None,
    gold_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Gold used to train: origin == review_queue, restricted to the split's train cases.

    Review-queue gold for a calibration/test case is excluded here (not just
    from ``gold_eval``) — see the module docstring.
    """
    split_id = split_id if split_id is not None else config.DEFAULT_SPLIT_ID
    gold = load_gold(gold_csv)
    queue = gold[gold["origin"] == "review_queue"]
    train_cases = load_split(split_id).train
    return queue[queue["case_id"].isin(train_cases)].reset_index(drop=True)


def gold_snapshot_hash(split_id: str | None = None, gold_csv: str | Path | None = None) -> str:
    """A stable sha256 of the current gold-train rows, for a generation's manifest."""
    train = gold_train(split_id, gold_csv)
    # lineterminator pinned: pandas defaults to os.linesep, which would give Windows a different hash.
    canonical = train.sort_values(["case_id", "code"]).to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
