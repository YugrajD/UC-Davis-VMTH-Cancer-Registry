"""Intra-annotator self-consistency measurement.

Compares dup_pass=1 (first listing) vs dup_pass=2 (duplicate listing) for the
same diagnosis rows in a filled review workbook.  Computes Cohen's kappa on the
verdict column and reports per-verdict agreement counts plus Jaccard similarity
on confirmed terms.

Flags the result if kappa < 0.7 (the standard "substantial agreement" threshold).
"""

from __future__ import annotations

from collections import defaultdict

from .xlsx_io import read_review_rows


def consistency(review_xlsx: str) -> None:
    """Print Cohen's kappa and per-verdict agreement for duplicate rows."""
    # Load rows grouped by (case_id, diagnosis_number) × dup_pass.
    pass1: dict[tuple, str] = {}   # (case_id, diag_num) → verdict
    pass2: dict[tuple, str] = {}
    pass1_terms: dict[tuple, str] = {}
    pass2_terms: dict[tuple, str] = {}

    for row in read_review_rows(review_xlsx):
        dup_pass = row.get("dup_pass", "1").strip()
        key = (row.get("case_id", ""), row.get("diagnosis_number", ""))
        verdict = row.get("verdict", "").strip().lower()
        term = (row.get("confirmed_term") or row.get("cascade_matched_term") or "").strip()
        if dup_pass == "1":
            pass1[key] = verdict
            pass1_terms[key] = term
        elif dup_pass == "2":
            pass2[key] = verdict
            pass2_terms[key] = term

    # Only score keys present in both passes.
    common_keys = sorted(set(pass1) & set(pass2))
    if not common_keys:
        print("No duplicate pairs found — cannot compute consistency.")
        return

    y1 = [pass1[k] for k in common_keys]
    y2 = [pass2[k] for k in common_keys]

    from sklearn.metrics import cohen_kappa_score
    kappa = cohen_kappa_score(y1, y2)

    # Per-verdict agreement table.
    agree_by_label: dict[str, int] = defaultdict(int)
    disagree_by_label: dict[str, int] = defaultdict(int)
    for k in common_keys:
        v1, v2 = pass1[k], pass2[k]
        if v1 == v2:
            agree_by_label[v1] += 1
        else:
            disagree_by_label[v1] += 1
            disagree_by_label[v2] += 1

    all_labels = sorted(set(agree_by_label) | set(disagree_by_label))

    print(f"\nIntra-annotator consistency  (n={len(common_keys)} duplicate diagnosis rows)")
    print(f"  Cohen's kappa = {kappa:.3f}", end="  ")
    if kappa < 0.7:
        print("  *** BELOW THRESHOLD (target kappa > 0.7) ***")
    else:
        print("  [OK]")

    print(f"\n  {'Verdict':<20} {'Agree':>7} {'Disagree':>9}")
    print("  " + "-" * 38)
    for label in all_labels:
        print(f"  {label:<20} {agree_by_label[label]:>7} {disagree_by_label[label]:>9}")

    # Jaccard on confirmed terms for non-blank pairs.
    term_pairs = [
        (pass1_terms[k], pass2_terms[k])
        for k in common_keys
        if pass1_terms[k] or pass2_terms[k]
    ]
    if term_pairs:
        exact_matches = sum(1 for a, b in term_pairs if a == b)
        # Token-level Jaccard (split on whitespace).
        def _jaccard(a: str, b: str) -> float:
            sa, sb = set(a.lower().split()), set(b.lower().split())
            if not sa and not sb:
                return 1.0
            union = sa | sb
            return len(sa & sb) / len(union) if union else 0.0

        avg_jaccard = sum(_jaccard(a, b) for a, b in term_pairs) / len(term_pairs)
        print(f"\n  Term agreement (confirmed_term / cascade_matched_term):")
        print(f"    Exact match: {exact_matches} / {len(term_pairs)} "
              f"({exact_matches / len(term_pairs) * 100:.1f}%)")
        print(f"    Avg token Jaccard: {avg_jaccard:.3f}")
