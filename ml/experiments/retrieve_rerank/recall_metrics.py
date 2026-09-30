"""Recall@N of ranked candidate terms against gold terms. Pure functions, no I/O.

``ranked``: case_id -> candidate terms, best first (at least max(ns) long). ``gold``: case_id -> gold
terms, only cases that are scored. A gold term outside the retriever's vocabulary can never be
ranked, so it counts as a miss (it stays in every denominator).
"""

from __future__ import annotations

FREQ_BUCKETS = (("<5", 0, 5), ("5-19", 5, 20), ("20-49", 20, 50), (">=50", 50, 10**9))


def freq_bucket(count: int) -> str:
    return next(name for name, lo, hi in FREQ_BUCKETS if lo <= count < hi)


def _pair_recall(pairs: list[tuple[str, str]], ranked: dict, n: int) -> float | None:
    """Share of (case, term) pairs whose term is in the case's top n."""
    if not pairs:
        return None
    return sum(term in ranked[case][:n] for case, term in pairs) / len(pairs)


def _case_recall(cases: list[str], gold: dict, ranked: dict, n: int) -> float | None:
    """Share of cases whose gold terms are ALL in the top n."""
    if not cases:
        return None
    return sum(gold[c] <= set(ranked[c][:n]) for c in cases) / len(cases)


def recall_report(ranked: dict, gold: dict, term_group: dict, train_freq: dict, ns=(5, 10, 20, 30, 49)) -> dict:
    pairs = [(c, t) for c, terms in gold.items() for t in sorted(terms)]
    report: dict = {"n_cases": len(gold), "n_pairs": len(pairs),
                    "n_oov_pairs": sum(t not in train_freq for _, t in pairs)}

    report["pair_recall"] = {n: _pair_recall(pairs, ranked, n) for n in ns}
    report["case_recall"] = {n: _case_recall(list(gold), gold, ranked, n) for n in ns}

    def group_hit(case: str, term: str, n: int) -> bool:
        return term_group[term] in {term_group[t] for t in ranked[case][:n]}
    report["group_pair_recall"] = {n: sum(group_hit(c, t, n) for c, t in pairs) / len(pairs) for n in ns}

    by_bucket = {}
    for name, _, _ in FREQ_BUCKETS:
        subset = [(c, t) for c, t in pairs if freq_bucket(train_freq.get(t, 0)) == name]
        by_bucket[name] = {"n_pairs": len(subset), "pair_recall": {n: _pair_recall(subset, ranked, n) for n in ns}}
    report["pair_recall_by_train_freq"] = by_bucket

    by_size = {}
    for name, keep in (("single_term", lambda c: len(gold[c]) == 1), ("multi_term", lambda c: len(gold[c]) > 1)):
        cases = [c for c in gold if keep(c)]
        subset = [(c, t) for c, t in pairs if keep(c)]
        by_size[name] = {"n_cases": len(cases), "n_pairs": len(subset),
                         "pair_recall": {n: _pair_recall(subset, ranked, n) for n in ns},
                         "case_recall": {n: _case_recall(cases, gold, ranked, n) for n in ns}}
    report["by_case_size"] = by_size
    return report
