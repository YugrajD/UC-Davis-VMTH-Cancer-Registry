"""The L1–L4 parity comparisons. Pure functions over tables and arrays; no file I/O.

Each level returns a ``Report``: ``passed`` (True / False, or None for the
report-only L4) and the lines to print. Lines hold counts, percentages, case IDs
and taxonomy group names only — never report or diagnosis text.

A "verdict table" is the per-code table legacy ``evaluate()`` writes to
evaluation.csv (true negatives excluded); G+S and every verdict share are
fractions of its rows.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from evaluation import verdicts
from evaluation.verdicts import GOOD_PLUS_SLIGHT, VERDICTS

PREDICTION_KEY = ["case_id", "diagnosis_index"]
PREDICTION_CODE_COLUMNS = ["predicted_term", "predicted_group", "predicted_code"]
PREDICTION_PROB_COLUMNS = ["case_presence_prob", "group_prob", "confidence"]

PROB_TOLERANCE = 1e-5
MIN_COSINE = 0.9999
MIN_IDENTICAL_ROW_SHARE = 99.5  # pp
GS_TOLERANCE_REEMBEDDED = 0.2  # pp
GS_TOLERANCE_FLOOR = 1.0  # pp; L3 tolerance = max(this, 2·sd across seeds)
VERDICT_SHARE_TOLERANCE = 1.5  # pp
GROUP_MIN_CODES = 50
GROUP_LOSS_REPORT = 5.0  # pp

EXAMPLE_IDS = 5  # case IDs shown per failing check


@dataclass
class Report:
    passed: bool | None
    lines: list[str] = field(default_factory=list)


def _verdict_line(passed: bool | None) -> str:
    return {True: "PASS", False: "FAIL", None: "REPORT (no gate)"}[passed]


def _examples(case_ids) -> str:
    ids = sorted(set(case_ids))[:EXAMPLE_IDS]
    return f" e.g. {', '.join(ids)}" if ids else ""


# ---------------------------------------------------------------------------
# Verdict-table summaries
# ---------------------------------------------------------------------------


def summarize(table: pd.DataFrame) -> dict:
    """``verdicts.summarize`` in percent: total, per-verdict counts and ``<verdict>_pct``,
    and ``gs_pct`` — the exact G+S share of rows (not a sum of rounded shares)."""
    shares = verdicts.summarize(table)
    summary = {"total": shares["total"]}
    for verdict in VERDICTS:
        summary[verdict] = shares[verdict]
        summary[f"{verdict}_pct"] = 100.0 * shares[f"{verdict}_share"]
    summary["gs_pct"] = 100.0 * shares["good_plus_slight_share"]
    return summary


def group_gs(table: pd.DataFrame) -> pd.DataFrame:
    """Per expected group: number of code rows and G+S %.

    A row counts toward every group in its ``expected_group`` (legacy joins a
    case's groups with " | "). Rows with no expected group (false positives)
    have no true group and are left out. Indexed by group; columns n, gs_pct.
    """
    rows = table.loc[table["expected_group"].fillna("") != "", ["expected_group", "verdict"]].copy()
    rows["group"] = rows["expected_group"].str.split(" | ", regex=False)
    rows = rows.explode("group")
    rows["gs"] = rows["verdict"].isin(GOOD_PLUS_SLIGHT)
    grouped = rows.groupby("group")["gs"]
    return pd.DataFrame({"n": grouped.size(), "gs_pct": 100.0 * grouped.mean()})


def _summary_line(label: str, s: dict) -> str:
    return (
        f"  {label:<12} n={s['total']:>6}  G+S {s['gs_pct']:5.1f}%  "
        + "  ".join(f"{v} {s[f'{v}_pct']:.1f}%" for v in VERDICTS)
    )


# ---------------------------------------------------------------------------
# L1 — scorer
# ---------------------------------------------------------------------------


def _canonical_rows(table: pd.DataFrame, columns: list[str]) -> Counter:
    frame = table[columns].fillna("").astype(str)
    return Counter(map(tuple, frame.itertuples(index=False, name=None)))


def l1(new_table: pd.DataFrame, ref_table: pd.DataFrame) -> Report:
    """New verdict table vs the legacy one, on every legacy column (legacy dropped
    ``method`` / ``predicted_code``). Rows compared as a multiset: legacy emitted a
    case's FN rows in set-hash order, so row order is not meaningful."""
    columns = list(ref_table.columns)
    missing = [c for c in columns if c not in new_table.columns]
    if missing:
        return Report(False, [f"  new verdict table lacks columns {missing}", "L1 FAIL"])
    new_rows, ref_rows = _canonical_rows(new_table, columns), _canonical_rows(ref_table, columns)
    only_new, only_ref = new_rows - ref_rows, ref_rows - new_rows
    n_only_new, n_only_ref = sum(only_new.values()), sum(only_ref.values())
    diff_ids = [row[0] for row in list(only_new) + list(only_ref)]
    passed = n_only_new == 0 and n_only_ref == 0
    lines = [
        _summary_line("legacy", summarize(ref_table)),
        _summary_line("new", summarize(new_table)),
        f"  rows: legacy {len(ref_table)}, new {len(new_table)}; only in new {n_only_new}, "
        f"only in legacy {n_only_ref}{_examples(diff_ids)}",
        f"L1 {_verdict_line(passed)}",
    ]
    return Report(passed, lines)


# ---------------------------------------------------------------------------
# L2 — inference
# ---------------------------------------------------------------------------


def _align(new_preds: pd.DataFrame, ref_preds: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Outer-join on (case_id, diagnosis_index); ``_merge`` marks unmatched rows. Returns (merged, dup keys)."""
    dups = int(new_preds.duplicated(PREDICTION_KEY).sum() + ref_preds.duplicated(PREDICTION_KEY).sum())
    cols = PREDICTION_KEY + PREDICTION_CODE_COLUMNS + PREDICTION_PROB_COLUMNS
    merged = ref_preds[cols].merge(new_preds[cols], on=PREDICTION_KEY, how="outer",
                                   suffixes=("_ref", "_new"), indicator=True)
    return merged, dups


def _code_bad(merged: pd.DataFrame, col: str) -> pd.Series:
    return merged[f"{col}_ref"].fillna("").astype(str) != merged[f"{col}_new"].fillna("").astype(str)


def _prob_bad(merged: pd.DataFrame, col: str) -> tuple[pd.Series, pd.Series]:
    """(out-of-tolerance mask, |diff|) for one probability column; one side blank is out."""
    ref = pd.to_numeric(merged[f"{col}_ref"], errors="coerce")
    new = pd.to_numeric(merged[f"{col}_new"], errors="coerce")
    diff = (new - ref).abs()
    return (ref.isna() != new.isna()) | (diff > PROB_TOLERANCE), diff


def _case_rows(preds: pd.DataFrame, case_ids: set, prob_columns: list[str]) -> dict[str, list[tuple]]:
    """Per case: its rows as sorted (term, group, code, *probs) tuples — the multiset, order-free."""
    frame = preds[preds["case_id"].isin(case_ids)]
    rows: dict[str, list[tuple]] = {}
    for cid, group in frame.groupby("case_id"):
        codes = group[PREDICTION_CODE_COLUMNS].fillna("").astype(str).to_numpy().tolist()
        probs = (group[prob_columns].apply(pd.to_numeric, errors="coerce").to_numpy().tolist()
                 if prob_columns else [[]] * len(group))
        rows[cid] = sorted(tuple(c) + tuple(p) for c, p in zip(codes, probs))
    return rows


def _same_multiset(ref_rows: list[tuple], new_rows: list[tuple]) -> bool:
    if len(ref_rows) != len(new_rows):
        return False
    n_codes = len(PREDICTION_CODE_COLUMNS)
    for ref, new in zip(ref_rows, new_rows):
        if ref[:n_codes] != new[:n_codes]:
            return False
        for a, b in zip(ref[n_codes:], new[n_codes:]):
            if (a != a) != (b != b) or abs(a - b) > PROB_TOLERANCE:  # NaN on one side only, or too far
                return False
    return True


def _differing_cases(merged: pd.DataFrame, bad: pd.Series, new_preds: pd.DataFrame, ref_preds: pd.DataFrame,
                     uncommon_groups: frozenset[str], prob_columns: list[str]) -> tuple[set, set]:
    """Split the cases with a differing row into (uncommon-reordered, other).

    Uncommon-reordered: every differing row, on both sides, is in a group from the
    frozen uncommon list, and the case's rows are the same multiset of (term,
    group, code, *prob_columns). Legacy picked among passing merged-Uncommon labels
    in hash order; the rewrite ranks them by confidence, so only the row order
    within such a case may change.
    """
    rows = merged[bad]
    in_uncommon = ((rows["predicted_group_ref"].isna() | rows["predicted_group_ref"].isin(uncommon_groups))
                   & (rows["predicted_group_new"].isna() | rows["predicted_group_new"].isin(uncommon_groups)))
    cases = set(rows["case_id"])
    candidates = cases - set(rows.loc[~in_uncommon, "case_id"])
    ref_rows = _case_rows(ref_preds, candidates, prob_columns)
    new_rows = _case_rows(new_preds, candidates, prob_columns)
    reordered = {cid for cid in candidates if _same_multiset(ref_rows.get(cid, []), new_rows.get(cid, []))}
    return reordered, cases - reordered


def _reorder_line(merged: pd.DataFrame, bad: pd.Series, reordered: set, other: set) -> str:
    n_rows = int((bad & merged["case_id"].isin(reordered)).sum())
    return (f"  cases with a differing row: {len(reordered) + len(other)}; uncommon-reordered {len(reordered)} "
            f"({n_rows} rows, not failing); other {len(other)}{_examples(other)}")


def l2(new_preds: pd.DataFrame, ref_preds: pd.DataFrame, uncommon_groups: frozenset[str] = frozenset()) -> Report:
    """Same checkpoints + embeddings: codes identical per row, probabilities within 1e-5.

    Duplicate keys fail. A case whose only differences are uncommon-group rows
    reordered within the case (see ``_differing_cases``) is counted, not failed.
    """
    merged, dups = _align(new_preds, ref_preds)
    both = merged["_merge"] == "both"
    lines = [f"  rows: legacy {len(ref_preds)}, new {len(new_preds)}; aligned {int(both.sum())}, "
             f"only in new {int((merged['_merge'] == 'right_only').sum())}, "
             f"only in legacy {int((merged['_merge'] == 'left_only').sum())}, duplicate keys {dups}"]
    bad = ~both
    for col in PREDICTION_CODE_COLUMNS:
        col_bad = both & _code_bad(merged, col)
        bad |= col_bad
        lines.append(f"  {col:<20} mismatches {int(col_bad.sum())}{_examples(merged.loc[col_bad, 'case_id'])}")
    for col in PREDICTION_PROB_COLUMNS:
        col_bad, diff = _prob_bad(merged, col)
        col_bad &= both
        bad |= col_bad
        max_diff = float(diff[both].max()) if diff[both].notna().any() else 0.0
        lines.append(f"  {col:<20} over {PROB_TOLERANCE:g}: {int(col_bad.sum())} (max |diff| {max_diff:.2e})"
                     f"{_examples(merged.loc[col_bad, 'case_id'])}")
    reordered, other = _differing_cases(merged, bad, new_preds, ref_preds, uncommon_groups, PREDICTION_PROB_COLUMNS)
    lines.append(_reorder_line(merged, bad, reordered, other))
    passed = dups == 0 and not other
    lines.append(f"L2 {_verdict_line(passed)}")
    return Report(passed, lines)


def _cosines(new_ids, new_mat, ref_ids, ref_mat) -> tuple[list[str], np.ndarray, list[str], int]:
    """Per reference case present in new: cosine(new, ref).

    Returns (aligned case ids, their cosines, ids missing from new, count of extra ids in new).
    """
    new_index = {cid: i for i, cid in enumerate(new_ids)}
    ref_rows = [i for i, cid in enumerate(ref_ids) if cid in new_index]
    missing = [cid for cid in ref_ids if cid not in new_index]
    extra = len(set(new_ids) - set(ref_ids))
    ref = np.asarray(ref_mat[ref_rows], dtype=np.float64)
    new = np.asarray(new_mat[[new_index[ref_ids[i]] for i in ref_rows]], dtype=np.float64)
    ref_norm, new_norm = np.linalg.norm(ref, axis=1), np.linalg.norm(new, axis=1)
    denom = ref_norm * new_norm
    cos = np.where(denom > 0, (ref * new).sum(axis=1) / np.where(denom > 0, denom, 1.0), 0.0)
    # Two empty (all-zero) embeddings — a case with no text in any section — agree.
    cos[(ref_norm == 0) & (new_norm == 0)] = 1.0
    return [ref_ids[i] for i in ref_rows], cos, missing, extra


def l2_reembedded(new_emb: tuple, ref_emb: tuple, new_preds: pd.DataFrame, ref_preds: pd.DataFrame,
                  new_eval_table: pd.DataFrame, ref_eval_table: pd.DataFrame,
                  uncommon_groups: frozenset[str] = frozenset()) -> Report:
    """Re-embedded from report.csv: cosine ≥ 0.9999 per case, ≥ 99.5% rows identical,
    eval-half G+S within ±0.2 pp. ``*_emb`` are (case_ids, matrix) pairs.

    Rows are compared on term, group and code only (probabilities may drift after
    re-embedding), so an uncommon-reordered case is judged on its (term, group,
    code) multiset and its rows count as identical. Duplicate keys fail."""
    (new_ids, new_mat), (ref_ids, ref_mat) = new_emb, ref_emb
    aligned, cos, missing, extra = _cosines(list(new_ids), new_mat, list(ref_ids), ref_mat)
    low = [cid for cid, c in zip(aligned, cos) if c < MIN_COSINE]
    emb_ok = not missing and not low
    lines = [
        f"  embeddings: legacy {len(ref_ids)} cases, new {len(new_ids)}; missing from new {len(missing)}"
        f"{_examples(missing)}, extra in new {extra}",
        f"  cosine: min {cos.min() if len(cos) else float('nan'):.6f}, cases below {MIN_COSINE}: {len(low)}"
        f"{_examples(low)}",
    ]

    merged, dups = _align(new_preds, ref_preds)
    both = merged["_merge"] == "both"
    bad = ~both
    for col in PREDICTION_CODE_COLUMNS:
        bad |= both & _code_bad(merged, col)
    reordered, other = _differing_cases(merged, bad, new_preds, ref_preds, uncommon_groups, [])
    # merged has one row per aligned pair and one per unmatched row on either side.
    union = len(merged)
    identical = union - int((bad & ~merged["case_id"].isin(reordered)).sum())
    identical_pct = 100.0 * identical / union if union else 0.0
    rows_ok = identical_pct >= MIN_IDENTICAL_ROW_SHARE and dups == 0
    lines.append(f"  prediction rows identical (term, group, code): {identical}/{union} = {identical_pct:.2f}% "
                 f"(need ≥ {MIN_IDENTICAL_ROW_SHARE}%; only in new {int((merged['_merge'] == 'right_only').sum())}, "
                 f"only in legacy {int((merged['_merge'] == 'left_only').sum())}, duplicate keys {dups})")
    lines.append(_reorder_line(merged, bad, reordered, other))

    new_s, ref_s = summarize(new_eval_table), summarize(ref_eval_table)
    gs_diff = new_s["gs_pct"] - ref_s["gs_pct"]
    gs_ok = abs(gs_diff) <= GS_TOLERANCE_REEMBEDDED + 1e-9
    lines += [
        _summary_line("legacy", ref_s),
        _summary_line("new", new_s),
        f"  eval-half G+S diff {gs_diff:+.2f} pp (tolerance ±{GS_TOLERANCE_REEMBEDDED})",
    ]
    passed = emb_ok and rows_ok and gs_ok
    lines.append(f"L2 re-embedded {_verdict_line(passed)} "
                 f"(embeddings {'ok' if emb_ok else 'FAIL'}, rows {'ok' if rows_ok else 'FAIL'}, "
                 f"G+S {'ok' if gs_ok else 'FAIL'})")
    return Report(passed, lines)


# ---------------------------------------------------------------------------
# L3 / L4 — retrained heads, cold start
# ---------------------------------------------------------------------------


def l3(seed_tables: list[pd.DataFrame], ref_table: pd.DataFrame, gate: bool = True, name: str = "L3") -> Report:
    """N seed runs' eval-half verdict tables vs the legacy one.

    PASS if |mean G+S − ref| ≤ max(1.0 pp, 2·sd across seeds) and every verdict
    share's mean is within ±1.5 pp. Groups with ≥ 50 legacy codes whose mean G+S
    dropped > 5 pp are listed but never fail. ``gate=False`` (L4) reports only.
    """
    ref_s = summarize(ref_table)
    seed_s = [summarize(t) for t in seed_tables]
    lines = [_summary_line("legacy", ref_s)]
    lines += [_summary_line(f"seed {i + 1}", s) for i, s in enumerate(seed_s)]

    gs = np.array([s["gs_pct"] for s in seed_s])
    sd = float(gs.std(ddof=1)) if len(gs) > 1 else 0.0
    tolerance = max(GS_TOLERANCE_FLOOR, 2 * sd)
    gs_diff = float(gs.mean()) - ref_s["gs_pct"]
    gs_ok = abs(gs_diff) <= tolerance + 1e-9
    lines.append(f"  mean G+S {gs.mean():.2f}% (sd {sd:.2f}, n={len(gs)}) vs legacy {ref_s['gs_pct']:.2f}%: "
                 f"diff {gs_diff:+.2f} pp, tolerance ±{tolerance:.2f} {'ok' if gs_ok else 'OUT'}")

    shares_ok = True
    for verdict in VERDICTS:
        mean = float(np.mean([s[f"{verdict}_pct"] for s in seed_s]))
        diff = mean - ref_s[f"{verdict}_pct"]
        ok = abs(diff) <= VERDICT_SHARE_TOLERANCE + 1e-9
        shares_ok &= ok
        lines.append(f"  {verdict:<16} mean {mean:5.2f}% vs {ref_s[f'{verdict}_pct']:5.2f}%: "
                     f"diff {diff:+.2f} pp (±{VERDICT_SHARE_TOLERANCE}) {'ok' if ok else 'OUT'}")

    ref_groups = group_gs(ref_table)
    ref_groups = ref_groups[ref_groups["n"] >= GROUP_MIN_CODES]
    seed_groups = [group_gs(t)["gs_pct"] for t in seed_tables]
    losers = []
    for group, row in ref_groups.iterrows():
        mean = float(np.nanmean([g.get(group, np.nan) for g in seed_groups]))
        if row["gs_pct"] - mean > GROUP_LOSS_REPORT:
            losers.append(f"    {group}: n={int(row['n'])}, G+S {row['gs_pct']:.1f}% -> {mean:.1f}%")
    lines.append(f"  groups with ≥ {GROUP_MIN_CODES} codes losing > {GROUP_LOSS_REPORT:g} pp G+S "
                 f"(reported, not gating): {len(losers)} of {len(ref_groups)}")
    lines += losers

    passed = (gs_ok and shares_ok) if gate else None
    lines.append(f"{name} {_verdict_line(passed)}")
    return Report(passed, lines)
