"""Per-stratum verdict rates for the row-level Tier-3 audit.

This is Step 4 of the annotation redesign plan — the number the audit batch exists
to produce.

The audit deliberately over-samples the small strata, so a raw rate off the gold
store is not a Tier-3 population rate. Every row carries `sample_weight` (N_h/n_h),
and this module reports both:

  - the **sample** rate  (what the reviewer actually saw)
  - the **weighted** rate (what it implies for the stratum's whole population)

Deliberately *not* wired into `evaluate.py`: this is a row-level sample, and
`evaluate.py` scores per case against a case's whole annotated term set, so the
un-sampled rows of a partially-covered case would read as false positives.

The population reported is the **test-split** Tier-3 population, because that is
what `sample` drew from. Extrapolating further, to the whole corpus, is a separate
step and this tool does not do it for you.

`sample_weight` is fixed at sample time as N_h/n_h for the *whole* batch, so Σweight
only reconstructs the stratum population once that batch is fully reviewed. Run this
on a partial batch — a `pilot` slice, or a batch half filled in — and every
"represents" figure is proportionally short. That is why a pilot is a comprehension
check and not a measurement; this tool warns when it sees a partial batch, but cannot
correct for it.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

VERDICT_ORDER = ["correct", "wrong", "no_cancer", "uncertain"]

# What each stratum was drawn to answer, and which verdict is the finding.
_STRATUM_QUESTION = {
    "tier3_llm_no_match": (
        "the model refused to label it — is there a cancer it missed?", "wrong"),
    "tier3_no_candidates": (
        "no shortlist was built, so the model was never asked — recall hole?", "wrong"),
    "tier3_llm_answered": (
        "the model picked a term — is it right?", "correct"),
    "tier3_llm_uncertain": (
        "the model called it hedged — is it genuinely unclassifiable?", "uncertain"),
    "tier2_fuzzy": (
        "partial-overlap match — same clinical entity?", "correct"),
}

# The two strata that together form the false-negative reservoir: rows the
# pipeline currently drops as non-cancer without anything checking them.
_DROPPED_STRATA = ("tier3_llm_no_match", "tier3_no_candidates")


def _load(gold_csv: str) -> list[dict]:
    path = Path(gold_csv)
    if not path.exists():
        print(f"ERROR — no gold store at {gold_csv}. Run `ingest` first.", file=sys.stderr)
        sys.exit(1)
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"ERROR — gold store {gold_csv} is empty.", file=sys.stderr)
        sys.exit(1)
    for col in ("verdict", "sample_stratum", "sample_weight"):
        if col not in rows[0]:
            print(
                f"ERROR — gold store {gold_csv} has no {col!r} column; it predates the "
                f"current schema. Re-run `ingest` on the filled review CSV to rebuild it.",
                file=sys.stderr,
            )
            sys.exit(1)
    return rows


def _pct(num: float, den: float) -> str:
    return f"{100.0 * num / den:5.1f}%" if den else "    —"


def rates(gold_csv: str, provenance: str = "tier3_audit") -> None:
    """Print sample and weighted verdict rates for every audit stratum."""
    rows = [r for r in _load(gold_csv) if r.get("provenance", "") == provenance]
    if not rows:
        print(f"ERROR — no rows with provenance={provenance!r} in {gold_csv}.", file=sys.stderr)
        sys.exit(1)

    unweighted: dict[str, Counter] = defaultdict(Counter)
    weighted: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        stratum = r.get("sample_stratum", "") or "(none)"
        verdict = (r.get("verdict", "") or "(blank)").strip().lower()
        try:
            w = float(r.get("sample_weight", "") or 0.0)
        except ValueError:
            w = 0.0
        unweighted[stratum][verdict] += 1
        weighted[stratum][verdict] += w

    print(f"Tier-3 audit rates — {len(rows)} reviewed row(s), "
          f"{len({r['case_id'] for r in rows})} case(s), provenance={provenance}")
    print("'represents' = Σ sample_weight: the test-split Tier-3 rows these stand for — "
          "exact\nonly for a fully-reviewed batch.\n")

    for stratum in sorted(unweighted, key=lambda s: -sum(weighted[s].values())):
        n = sum(unweighted[stratum].values())
        pop = sum(weighted[stratum].values())
        question, headline = _STRATUM_QUESTION.get(stratum, ("", None))
        print(f"── {stratum}  (n={n} reviewed → represents {pop:,.0f} rows)")
        if question:
            print(f"   Q: {question}")
        seen = list(VERDICT_ORDER) + [v for v in unweighted[stratum] if v not in VERDICT_ORDER]
        for v in seen:
            c = unweighted[stratum].get(v, 0)
            if not c:
                continue
            w = weighted[stratum].get(v, 0.0)
            mark = " ←" if v == headline else ""
            print(f"     {v:<10} sample {c:>3}/{n:<3} {_pct(c, n)}"
                  f"   represents {w:>7,.0f}/{pop:<7,.0f} {_pct(w, pop)}{mark}")
        print()

    # The reservoir: rows currently dropped as non-cancer that a human says are cancer.
    res_n = res_pop = res_hit = res_hit_w = 0.0
    for stratum in _DROPPED_STRATA:
        if stratum not in unweighted:
            continue
        res_n += sum(unweighted[stratum].values())
        res_pop += sum(weighted[stratum].values())
        res_hit += unweighted[stratum].get("wrong", 0)
        res_hit_w += weighted[stratum].get("wrong", 0.0)
    if res_n:
        print("── FALSE-NEGATIVE RESERVOIR "
              f"({' + '.join(s for s in _DROPPED_STRATA if s in unweighted)})")
        print("   Rows the pipeline silently drops as non-cancer, that the reviewer says "
              "carry a real cancer:")
        print(f"     sample {res_hit:.0f}/{res_n:.0f} {_pct(res_hit, res_n)}"
              f"   → est. {res_hit_w:,.0f} of {res_pop:,.0f} test-split rows "
              f"({_pct(res_hit_w, res_pop).strip()})")

    _warn_if_partial(unweighted)

    blanks = sum(c["(blank)"] for c in unweighted.values())
    if blanks:
        print(f"\nWARNING: {blanks} row(s) carry a blank verdict.")


def _warn_if_partial(unweighted: dict[str, Counter]) -> None:
    """Flag a store that looks like a partial batch, where Σweight under-counts.

    A full batch draws 25-50 rows per stratum. Far fewer means the weights — fixed at
    sample time for the whole batch — are being applied to a slice, so every
    'represents' figure is proportionally short and none of them should be quoted.
    """
    thin = {s: sum(c.values()) for s, c in unweighted.items() if sum(c.values()) < 20}
    if not thin:
        return
    print("\nWARNING: this looks like a PARTIAL batch — "
          + ", ".join(f"{s} has only {n} row(s)" for s, n in sorted(thin.items())) + ".")
    print("  sample_weight was fixed for the full batch, so every 'represents' figure above")
    print("  is proportionally short. Read the sample rates as a comprehension check, not")
    print("  a measurement.")
