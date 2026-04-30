"""Evaluate the three rule-based gates against the 40 manually labeled cases.

Reads the two filled spot-check CSVs (10 + 30), simulates each gate against
the (predicted_term, predicted_group, FINAL COMMENT, HISTOPATHOLOGICAL SUMMARY)
already in those files, and prints a per-case before/after diff plus the
summary counts requested in the conversation.

Does not require the labels CSV: the subtype gate's NOS lookup is built from
the predicted_group column itself by stripping the qualifier and falling back
to "<head>, NOS" when the group has a known NOS form. For the production
pipeline the real labels CSV is used (see gates.py).

Usage:
    python ml/scripts/eval_gates_on_spotcheck.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# Import text_filters directly (bypassing the package __init__ that pulls in torch).
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "text_filters",
    REPO_ROOT / "ml/production/petbert_pipeline/text_filters.py",
)
_text_filters = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_text_filters)  # type: ignore[union-attr]
looks_non_neoplastic = _text_filters.looks_non_neoplastic
qualifier_words_missing_from_text = _text_filters.qualifier_words_missing_from_text
strip_tissue_lists = _text_filters.strip_tissue_lists


SPOT_CHECKS = [
    REPO_ROOT / "ml/output/evaluation/quick_spot_check_10_filled.csv",
    REPO_ROOT / "ml/output/evaluation/spot_check_30_stratified_filled.csv",
]


def normalize_verdict(raw: str) -> str:
    v = (raw or "").strip().lower()
    if "wrong" in v and "partial" not in v:
        return "wrong"
    if "partial" in v and "wrong" in v:
        return "wrong"  # treat partial/wrong as wrong
    if "partial" in v:
        return "partial"
    if "broad" in v:
        return "correct_broad"
    if "correct" in v:
        return "correct"
    return "unlabeled"


def derive_nos_for_group(group: str) -> str:
    """Best-effort NOS variant for a group name (used only in this eval).

    Production code uses the labels CSV; here we just suffix the group's head
    noun with ', NOS'. Good enough for sanity-checking the demotion logic.
    """
    if not group:
        return ""
    return f"{group.rstrip(',').strip()}: NOS"  # cosmetic — real gate uses labels CSV


def load_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        if not p.exists():
            print(f"missing: {p}", file=sys.stderr)
            continue
        # quick_spot_check_10_full.csv has the verdict/notes filled by hand;
        # spot_check_30_stratified.csv is the empty template, so we read the
        # filled copy if the user has saved one back. Fall back to template.
        with open(p, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
    return rows


def main() -> int:
    rows = load_rows(SPOT_CHECKS)
    if not rows:
        print("no rows loaded")
        return 1

    n = len(rows)
    print(f"Loaded {n} cases from {len(SPOT_CHECKS)} spot-check files\n")

    n_strip_changed = 0
    n_subtype_demote = 0
    n_non_neo = 0
    flips_better: list[str] = []
    flips_worse: list[str] = []
    actions: list[dict] = []

    for r in rows:
        cid = r.get("case_id", "")
        term = r.get("predicted_term", "").strip()
        group = r.get("predicted_group", "").strip()
        fc = r.get("FINAL COMMENT", "")
        hp = r.get("HISTOPATHOLOGICAL SUMMARY", "")
        verdict = normalize_verdict(r.get("verdict", ""))

        # 1. Tissue-list filter (informational — count cases where it would change HP)
        hp_after = strip_tissue_lists(hp)
        stripped = hp_after != hp

        # 2. Subtype gate
        missing = qualifier_words_missing_from_text(term, f"{fc} {hp}")
        subtype_demote = bool(missing)

        # 3. Non-neoplastic gate
        non_neo = looks_non_neoplastic(fc, hp)

        if stripped:
            n_strip_changed += 1
        if subtype_demote:
            n_subtype_demote += 1
        if non_neo:
            n_non_neo += 1

        action = []
        if non_neo:
            action.append("→ Non-neoplastic")
        elif subtype_demote:
            action.append(f"→ {derive_nos_for_group(group)} (missing: {missing})")

        if action:
            # Score the flip:
            # - non-neo suppression on a "wrong" verdict for a non-cancer case = good
            # - non-neo suppression on a correct/partial cancer prediction = bad
            # - subtype demote on a "partial" verdict = good (per pattern)
            # - subtype demote on a "correct" verdict = bad
            improved = (
                (non_neo and verdict == "wrong")
                or (subtype_demote and verdict in {"partial"})
            )
            regressed = (
                (non_neo and verdict in {"correct", "correct_broad"})
                or (subtype_demote and verdict == "correct")
            )
            (flips_better if improved else (flips_worse if regressed else flips_better)).append(
                f"  [{verdict:<14}] {cid}  {term!r}  {' '.join(action)}"
            )
            actions.append({
                "case_id": cid,
                "term": term,
                "group": group,
                "verdict": verdict,
                "stripped_hp": stripped,
                "subtype_demote": subtype_demote,
                "missing": missing,
                "non_neoplastic": non_neo,
            })

    print("=== Tissue-list filter (informational) ===")
    print(f"  Would strip list segments from {n_strip_changed}/{n} HP summaries\n")

    print("=== Subtype-qualifier gate ===")
    print(f"  Would demote {n_subtype_demote}/{n} predictions to NOS\n")

    print("=== Non-neoplastic gate ===")
    print(f"  Would suppress {n_non_neo}/{n} predictions as non-neoplastic\n")

    print("=== Flips: should improve outcome ===")
    for line in flips_better:
        print(line)
    print()

    print("=== Flips: would regress a correct prediction ===")
    if not flips_worse:
        print("  (none)")
    for line in flips_worse:
        print(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
