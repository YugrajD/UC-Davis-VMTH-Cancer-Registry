"""LLM-assisted pipeline for mapping diagnosis text to Vet-ICD-O taxonomy labels.

Flow:
  Pre-pass : normalize text + expand abbreviations & synonyms
  Tier 1   : exact match  (keyword index: full term, core term, permutations, plurals)
  Tier 2   : fuzzy match  (token-overlap on core terms)
  Tier 3   : signal fallback + LLM  (diagnoses that contain cancer-indicating terms)
  Tier 4   : no match
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from itertools import permutations as _permutations

import pandas as pd

from labels.taxonomy import TaxonomyLabel, load_labels_taxonomy
from annotation.keyword_pipeline.pipeline import (
    _normalize,
    _build_keyword_index,
    _build_oma_index,
    _OMA_RE,
    _QUALIFIER_RE,
)
from annotation.llm_pipeline.client import chat

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Abbreviation expansions and synonym mappings applied after normalization (lowercase input assumed).
_ABBREVIATIONS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bcpnet\b"), "central primitive neuroectodermal tumor"),
    (re.compile(r"\bpnet\b"), "primitive neuroectodermal tumor"),
    (re.compile(r"\bdlbcl\b"), "diffuse large b cell lymphoma"),
    (re.compile(r"\bgist\b"), "gastrointestinal stromal tumor"),
    (re.compile(r"\btvt\b"), "transmissible venereal tumor"),
    (re.compile(r"\bhsa\b"), "hemangiosarcoma"),
    (re.compile(r"\bosa\b"), "osteosarcoma"),
    (re.compile(r"\bhcc\b"), "hepatocellular carcinoma"),
    (re.compile(r"\bscc\b"), "squamous cell carcinoma"),
    (re.compile(r"\bmct\b"), "mast cell tumor"),
    # Synonym mappings: common diagnosis phrasings not in the taxonomy verbatim
    (re.compile(r"\bangiosarcoma\b"), "hemangiosarcoma"),
    (re.compile(r"\bplasmacell\b"), "plasmacytoma"),
    (re.compile(r"\bperivascular wall tumor\b"), "canine perivascular wall tumor"),
]

# Extra synonym substitution not present in keyword pipeline's _normalize().
_RE_METASTASIS = re.compile(r"\bmetastasis\b")

# Signal terms that suggest a neoplastic diagnosis and trigger Tier 3.
# Oma/emia words are detected separately via _OMA_RE.
_SIGNAL_RE = re.compile(
    r"\b(?:"
    r"tumor|tumour|leukemia|leukaemia|neoplasm|cancer|malignancy|malignant|metastatic|"
    r"carcinoid|mycosis|fungoides|polycythemia|paget|mastocytosis|"
    r"ganglioneuromatosis|gliomatosis|lipomatosis|refractory\s+anemia|"
    r"acanthomatous|fibromatosis"
    r")\b",
    re.IGNORECASE,
)

# Fuzzy Tier 2: minimum fraction of core-term tokens that must appear in diagnosis.
_FUZZY_THRESHOLD = 0.85

# LLM Tier 3: maximum number of candidate terms passed to the model.
_LLM_MAX_CANDIDATES = 30


# ---------------------------------------------------------------------------
# Config / output types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LLMConfig:
    csv_path: str
    id_col: str
    diag_num_col: str
    text_col: str
    labels_csv_path: str
    out_dir: str
    max_rows: int | None
    llm_timeout: int = 60
    llm_model: str | None = None  # None uses the default from .env
    use_claude: bool = False       # Enable Tier 4 Claude reasoning fallback
    claude_timeout: int = 30


@dataclass(frozen=True)
class LLMOutputs:
    predictions_csv: str
    summary_json: str
    summary_md: str


@dataclass(frozen=True)
class _MatchResult:
    term: str
    group: str
    code: str
    keyword: str
    method: str        # "Exact" | "Fuzzy" | "LLM" | "No Match"
    confidence: float  # 1.0=Exact/LLM, 0.0–1.0=Fuzzy, 0.0=No Match


_NO_MATCH = _MatchResult(term="", group="", code="", keyword="", method="No Match", confidence=0.0)


# ---------------------------------------------------------------------------
# Pre-pass: normalization
# ---------------------------------------------------------------------------

def _normalize_llm(text: str) -> str:
    """Normalize text using the keyword pipeline's rules, then expand abbreviations."""
    text = _normalize(text)
    text = _RE_METASTASIS.sub("metastatic neoplasm", text)
    for pattern, expansion in _ABBREVIATIONS:
        text = pattern.sub(expansion, text)
    return text


# ---------------------------------------------------------------------------
# Tier 1: Exact match (reuses keyword pipeline index)
# ---------------------------------------------------------------------------

def _tier1_exact(
    norm_text: str,
    keyword_index: list[tuple[str, re.Pattern, int]],
    taxonomy_labels: list[TaxonomyLabel],
) -> _MatchResult | None:
    for kw, pattern, label_idx in keyword_index:
        if pattern.search(norm_text):
            label = taxonomy_labels[label_idx]
            return _MatchResult(
                term=label.term, group=label.group, code=label.code,
                keyword=kw, method="Exact", confidence=1.0,
            )
    return None


# ---------------------------------------------------------------------------
# Tier 2: Fuzzy match (token overlap on core terms)
# ---------------------------------------------------------------------------

def _core_term(norm_term: str) -> str:
    """Strip qualifier words from a normalized term to get its core tokens."""
    return _QUALIFIER_RE.sub("", norm_term).strip().strip(",").strip()


def _token_overlap(core: str, norm_text: str) -> float:
    """Fraction of core tokens found in the diagnosis tokens."""
    core_tokens = core.split()
    if len(core_tokens) < 2:
        return 0.0
    text_tokens = set(norm_text.split())
    matched = sum(1 for t in core_tokens if t in text_tokens)
    return matched / len(core_tokens)


def _tier2_fuzzy(
    norm_text: str,
    taxonomy_labels: list[TaxonomyLabel],
) -> _MatchResult | None:
    best: tuple[float, _MatchResult] | None = None
    for label in taxonomy_labels:
        core = _core_term(_normalize(label.term))
        if not core:
            continue
        score = _token_overlap(core, norm_text)
        if score >= _FUZZY_THRESHOLD:
            if best is None or score > best[0]:
                best = (score, _MatchResult(
                    term=label.term, group=label.group, code=label.code,
                    keyword=core, method="Fuzzy", confidence=round(score, 2),
                ))
    return best[1] if best else None


# ---------------------------------------------------------------------------
# Tier 3: Signal fallback + LLM
# ---------------------------------------------------------------------------

def _has_signal(norm_text: str) -> bool:
    """Return True if the diagnosis contains any cancer-indicating term."""
    return bool(_OMA_RE.search(norm_text) or _SIGNAL_RE.search(norm_text))


def _build_group_index(
    taxonomy_labels: list[TaxonomyLabel],
) -> list[tuple[str, re.Pattern, str]]:
    """Build a keyword index mapping group-level keywords → group name."""
    groups: dict[str, str] = {}
    for label in taxonomy_labels:
        norm = _normalize(label.group)
        groups[norm] = label.group

    entries: list[tuple[str, re.Pattern, str]] = []
    seen: set[str] = set()
    for norm_group, original_group in groups.items():
        core = _QUALIFIER_RE.sub("", norm_group).strip()
        candidates: set[str] = {norm_group, core}
        for c in list(candidates):
            words = c.split()
            if 2 <= len(words) <= 3:
                for perm in _permutations(words):
                    candidates.add(" ".join(perm))
        for kw in candidates:
            if len(kw) < 4 or kw in seen:
                continue
            seen.add(kw)
            pat = re.compile(r"\b" + re.escape(kw) + r"s?\b")
            entries.append((kw, pat, original_group))

    entries.sort(key=lambda x: len(x[0]), reverse=True)
    return entries


def _identify_group(
    norm_text: str,
    group_index: list[tuple[str, re.Pattern, str]],
) -> str | None:
    for _kw, pattern, group in group_index:
        if pattern.search(norm_text):
            return group
    return None


_UNCERTAIN_RESULT = _MatchResult(term="", group="", code="", keyword="", method="Uncertain", confidence=0.0)


def _build_llm_prompt(original_text: str, candidates: list[TaxonomyLabel]) -> str:
    term_list = "\n".join(f"{i + 1}. {c.term}" for i, c in enumerate(candidates))
    return (
        "You are a veterinary oncology classifier. "
        "Map the diagnosis below to the best matching ICD term.\n\n"
        f'Diagnosis: "{original_text}"\n\n'
        f"Candidate ICD terms:\n{term_list}\n\n"
        "Rules:\n"
        "- Reply with ONLY the exact text of the best matching candidate "
        "(copy it character-for-character).\n"
        "- If the diagnosis is negated anywhere in the text — e.g. contains phrases like "
        '"no evidence of", "no metastasis", "not observed", "rule out", '
        '"not consistent with", "negative for", "cannot exclude" — reply with: no match\n'
        "- If the diagnosis is uncertain or hedged — e.g. contains phrases like "
        '"presumed", "suspect", "possible", "suspected", "consistent with", "compatible with", '
        '"likely", "probable", "versus", "vs." — reply with: uncertain\n'
        "- If no candidate fits the diagnosis, reply with: no match\n\n"
        "Your answer:"
    )


def _parse_llm_response(
    response: str,
    candidates: list[TaxonomyLabel],
    method: str = "LLM",
) -> _MatchResult | None:
    """Match LLM response text back to a taxonomy label. Returns None on no match."""
    response = response.strip()
    if not response or response.lower() == "no match":
        return None
    if response.lower() == "uncertain":
        return _UNCERTAIN_RESULT
    # Exact match first
    for label in candidates:
        if label.term.lower() == response.lower():
            return _MatchResult(
                term=label.term, group=label.group, code=label.code,
                keyword=response, method=method, confidence=1.0,
            )
    # Fuzzy fallback: LLM may have paraphrased slightly
    from difflib import get_close_matches
    term_names = [c.term for c in candidates]
    close = get_close_matches(response, term_names, n=1, cutoff=0.8)
    if close:
        label = next(c for c in candidates if c.term == close[0])
        return _MatchResult(
            term=label.term, group=label.group, code=label.code,
            keyword=response, method=method, confidence=0.9,
        )
    return None


def _tier3_llm(
    original_text: str,
    norm_text: str,
    group_index: list[tuple[str, re.Pattern, str]],
    taxonomy_labels: list[TaxonomyLabel],
    oma_index: dict[str, int],
    timeout: int,
    model: str | None,
    counters: dict,
) -> _MatchResult | None:
    group = _identify_group(norm_text, group_index)

    if group:
        candidates = [l for l in taxonomy_labels if l.group == group]
    else:
        # Suffix-based fallback: collect candidates via oma/emia words
        raw_words = _OMA_RE.findall(norm_text)
        indices = {oma_index[w] for w in set(raw_words) if w in oma_index}
        candidates = [taxonomy_labels[i] for i in sorted(indices)]

    if not candidates:
        return None

    candidates = candidates[:_LLM_MAX_CANDIDATES]
    prompt = _build_llm_prompt(original_text, candidates)

    counters["tier3_calls"] += 1
    try:
        response = chat(prompt, model=model, timeout=timeout)
    except Exception:
        counters["tier3_no_match"] += 1
        return None

    result = _parse_llm_response(response, candidates)
    if result is None:
        counters["tier3_no_match"] += 1
    elif result.method == "Uncertain":
        counters["tier3_uncertain"] += 1
    else:
        counters["tier3_matched"] += 1
    return result


# ---------------------------------------------------------------------------
# Tier 4: Claude free-form reasoning (full taxonomy, no pre-filtering)
# ---------------------------------------------------------------------------

def _build_tier4_prompt(original_text: str, taxonomy_labels: list[TaxonomyLabel]) -> str:
    """Build a prompt showing the full taxonomy grouped by cancer category."""
    from collections import defaultdict
    groups: dict[str, list[str]] = defaultdict(list)
    for label in taxonomy_labels:
        groups[label.group].append(label.term)

    taxonomy_block = "\n".join(
        f"[{group}]\n" + "\n".join(f"  - {term}" for term in terms)
        for group, terms in sorted(groups.items())
    )
    return (
        "You are a veterinary oncology ICD classifier with expert knowledge of "
        "veterinary pathology synonyms.\n\n"
        f'Diagnosis: "{original_text}"\n\n'
        "Map this diagnosis to the single best matching Vet-ICD-O-canine-1 term "
        "from the full taxonomy below.\n\n"
        "Apply these veterinary synonyms when matching:\n"
        "- angiosarcoma = hemangiosarcoma\n"
        "- perivascular wall tumor = canine perivascular wall tumor\n"
        "- GIST = gastrointestinal stromal tumor\n"
        "- acanthomatous epulis = acanthomatous ameloblastoma\n\n"
        f"{taxonomy_block}\n\n"
        "Rules:\n"
        "- Reply with ONLY the exact term text, copied character-for-character "
        "from the list above.\n"
        "- If the diagnosis is negated (no evidence of, rule out, negative for, "
        "not observed) → reply: no match\n"
        "- If uncertain or hedged (presumed, suspect, possible, likely, versus, "
        "consistent with, compatible with) → reply: uncertain\n"
        "- If no term fits → reply: no match\n\n"
        "Your answer:"
    )


def _tier4_claude(
    original_text: str,
    taxonomy_labels: list[TaxonomyLabel],
    timeout: int,
    counters: dict,
) -> _MatchResult | None:
    """Tier 4: call Claude with the full taxonomy for free-form synonym reasoning."""
    try:
        from annotation.llm_pipeline.client_claude import claude_classify
    except Exception:
        return None

    prompt = _build_tier4_prompt(original_text, taxonomy_labels)
    counters["tier4_calls"] += 1
    try:
        response = claude_classify(prompt, timeout=timeout)
    except Exception:
        counters["tier4_no_match"] += 1
        return None

    result = _parse_llm_response(response, taxonomy_labels, method="Claude")
    if result is None:
        counters["tier4_no_match"] += 1
    elif result.method == "Uncertain":
        counters["tier4_uncertain"] += 1
    else:
        counters["tier4_matched"] += 1
    return result


# ---------------------------------------------------------------------------
# Per-row matching
# ---------------------------------------------------------------------------

def _match_diagnosis(
    original_text: str,
    keyword_index: list[tuple[str, re.Pattern, int]],
    taxonomy_labels: list[TaxonomyLabel],
    oma_index: dict[str, int],
    group_index: list[tuple[str, re.Pattern, str]],
    llm_timeout: int,
    llm_model: str | None,
    use_claude: bool = False,
    claude_timeout: int = 30,
    counters: dict | None = None,
) -> _MatchResult:
    if counters is None:
        counters = {}
    norm_text = _normalize_llm(original_text)

    result = _tier1_exact(norm_text, keyword_index, taxonomy_labels)
    if result:
        return result

    result = _tier2_fuzzy(norm_text, taxonomy_labels)
    if result:
        return result

    if _has_signal(norm_text):
        counters["signal_rows"] = counters.get("signal_rows", 0) + 1
        result = _tier3_llm(original_text, norm_text, group_index, taxonomy_labels, oma_index, llm_timeout, llm_model, counters)
        if result:
            return result

        if use_claude:
            result = _tier4_claude(original_text, taxonomy_labels, claude_timeout, counters)
            if result:
                return result

    return _NO_MATCH


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _write_summary_md(summary: dict, path: str) -> None:
    mc = summary["method_counts"]
    total = summary["total_rows"]
    lines = [
        "# LLM Pipeline Summary",
        "",
        "## Overview",
        f"| | |",
        f"|---|---|",
        f"| Total diagnoses | {total:,} |",
        f"| Matched (confirmed) | {summary['matched_rows']:,} ({summary['match_rate_pct']}%) |",
        f"| Uncertain | {summary['uncertain_rows']:,} ({round(100*summary['uncertain_rows']/max(total,1),1)}%) |",
        f"| No match | {mc.get('No Match', mc.get('no_match', 0)):,} ({round(100*mc.get('No Match', mc.get('no_match', 0))/max(total,1),1)}%) |",
        "",
        "## Cases",
        f"| | |",
        f"|---|---|",
        f"| Total unique cases | {summary['total_cases']:,} |",
        f"| Cases with confirmed ICD match | {summary['cases_with_confirmed_match']:,} ({summary['case_match_rate_pct']}%) |",
        f"| Cases with uncertain diagnosis | {summary['cases_with_uncertain']:,} |",
        "",
        "## Method Breakdown",
        "| Method | Count | % of total |",
        "|--------|------:|----------:|",
    ]
    for method, count in sorted(mc.items(), key=lambda x: -x[1]):
        lines.append(f"| {method} | {count:,} | {round(100*count/max(total,1),1)}% |")

    ts = summary.get("tier_stats", {})
    if ts:
        claude_enabled = ts.get("claude_enabled", False)
        lines += [
            "",
            "## Tier Statistics",
            "| | |",
            "|---|---|",
            f"| Rows with cancer signal (Tier 3 eligible) | {ts.get('signal_rows', 0):,} |",
            f"| Ollama calls made | {ts.get('tier3_calls', 0):,} |",
            f"| Ollama matched | {ts.get('tier3_matched', 0):,} |",
            f"| Ollama uncertain | {ts.get('tier3_uncertain', 0):,} |",
            f"| Ollama no match | {ts.get('tier3_no_match', 0):,} |",
            f"| Claude enabled | {'Yes' if claude_enabled else 'No'} |",
            f"| Claude calls made | {ts.get('tier4_calls', 0):,} |",
            f"| Claude matched | {ts.get('tier4_matched', 0):,} |",
            f"| Claude uncertain | {ts.get('tier4_uncertain', 0):,} |",
            f"| Claude no match | {ts.get('tier4_no_match', 0):,} |",
        ]

    lines += [
        "",
        "## Taxonomy Coverage",
        f"| | |",
        f"|---|---|",
        f"| Unique terms matched | {summary['unique_terms_matched']} of 846 total |",
        f"| Unique groups matched | {summary['unique_groups_matched']} of 52 total |",
        "",
        "## Imbalance",
        f"| | |",
        f"|---|---|",
        f"| Most matched term | {next(iter(summary['top_matched_terms']))} ({next(iter(summary['top_matched_terms'].values()))} matches) |",
        f"| Terms with only 1 match | {summary['imbalance']['terms_with_1_match']} |",
        f"| Terms with < 5 matches | {summary['imbalance']['terms_with_lt5_matches']} |",
        f"| Terms with ≥ 100 matches | {summary['imbalance']['terms_with_gte100_matches']} |",
        "",
        "## Group Distribution (all 52 groups)",
        "| Group | Count | % of matched |",
        "|-------|------:|-------------:|",
    ]
    for group, vals in sorted(summary["group_distribution"].items(), key=lambda x: -x[1]["count"]):
        lines.append(f"| {group} | {vals['count']:,} | {vals['pct']}% |")

    lines += [
        "",
        "## Top 20 Matched Terms",
        "| Term | Count |",
        "|------|------:|",
    ]
    for term, count in summary["top_matched_terms"].items():
        lines.append(f"| {term} | {count:,} |")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_llm_scan(config: LLMConfig) -> LLMOutputs:
    """Execute the LLM-assisted diagnosis annotation pipeline."""
    os.makedirs(config.out_dir, exist_ok=True)
    outputs = LLMOutputs(
        predictions_csv=os.path.join(config.out_dir, "llm_annotation.csv"),
        summary_json=os.path.join(config.out_dir, "llm_summary.json"),
        summary_md=os.path.join(config.out_dir, "llm_summary.md"),
    )

    df = pd.read_csv(config.csv_path, encoding="latin-1")
    df.columns = [col.lstrip("\ufeff").lstrip("ï»¿") for col in df.columns]
    if config.max_rows is not None:
        df = df.head(config.max_rows).copy()

    taxonomy_labels = load_labels_taxonomy(config.labels_csv_path)
    keyword_index = _build_keyword_index(taxonomy_labels)
    oma_index = _build_oma_index(taxonomy_labels)
    group_index = _build_group_index(taxonomy_labels)

    total = len(df)
    counters: dict = {
        "signal_rows": 0,
        "tier3_calls": 0,
        "tier3_matched": 0,
        "tier3_uncertain": 0,
        "tier3_no_match": 0,
        "tier4_calls": 0,
        "tier4_matched": 0,
        "tier4_uncertain": 0,
        "tier4_no_match": 0,
    }
    results = []
    for i, (_, row) in enumerate(df.iterrows(), 1):
        text = str(row[config.text_col]) if pd.notna(row[config.text_col]) else ""
        match = _match_diagnosis(
            text, keyword_index, taxonomy_labels, oma_index, group_index,
            config.llm_timeout, config.llm_model,
            use_claude=config.use_claude, claude_timeout=config.claude_timeout,
            counters=counters,
        )

        if match.method in ("LLM", "Claude"):
            tier_label = match.method
            print(f"[{i}/{total}] {tier_label:<6}: {text[:60]!r}  ->  {match.term!r}")
        elif i % 500 == 0:
            print(f"[{i}/{total}] ...")

        entry: dict = {config.id_col: row[config.id_col]}
        if config.diag_num_col in df.columns:
            entry[config.diag_num_col] = row[config.diag_num_col]
        entry.update({
            config.text_col: text,
            "matched_term": match.term,
            "matched_group": match.group,
            "matched_code": match.code,
            "matched_keyword": match.keyword,
            "method": match.method,
            "confidence": match.confidence,
        })
        results.append(entry)

    out_df = pd.DataFrame(results)
    out_df.to_csv(outputs.predictions_csv, index=False)

    method_counts = out_df["method"].value_counts().to_dict()
    confirmed_df = out_df[~out_df["method"].isin(["No Match", "Uncertain"])]
    matched_df = out_df[out_df["method"] != "No Match"]  # includes Uncertain

    # Case-level stats
    total_cases = out_df[config.id_col].nunique()
    confirmed_cases = confirmed_df[config.id_col].nunique()
    uncertain_cases = out_df[out_df["method"] == "Uncertain"][config.id_col].nunique()

    # Imbalance stats (confirmed matches only)
    term_counts = confirmed_df["matched_term"].value_counts()
    group_counts = confirmed_df["matched_group"].value_counts()

    summary = {
        "csv_path": config.csv_path,
        # Row-level
        "total_rows": total,
        "matched_rows": len(confirmed_df),
        "uncertain_rows": int((out_df["method"] == "Uncertain").sum()),
        "match_rate_pct": round(100 * len(confirmed_df) / max(total, 1), 1),
        "method_counts": method_counts,
        # Tier-level call stats
        "tier_stats": {
            "claude_enabled": config.use_claude,
            **counters,
        },
        # Case-level
        "total_cases": total_cases,
        "cases_with_confirmed_match": confirmed_cases,
        "cases_with_uncertain": uncertain_cases,
        "case_match_rate_pct": round(100 * confirmed_cases / max(total_cases, 1), 1),
        # Taxonomy coverage
        "unique_terms_matched": int(term_counts.nunique()),
        "unique_groups_matched": int(group_counts.nunique()),
        # Imbalance
        "imbalance": {
            "top_term_count": int(term_counts.iloc[0]) if len(term_counts) else 0,
            "bottom_term_count": int(term_counts.iloc[-1]) if len(term_counts) else 0,
            "terms_with_1_match": int((term_counts == 1).sum()),
            "terms_with_lt5_matches": int((term_counts < 5).sum()),
            "terms_with_gte100_matches": int((term_counts >= 100).sum()),
        },
        "term_distribution": term_counts.to_dict(),
        "group_distribution": {
            g: {
                "count": int(group_counts.get(g, 0)),
                "pct": round(100 * group_counts.get(g, 0) / max(len(confirmed_df), 1), 2),
            }
            for g in sorted({label.group for label in taxonomy_labels})
        },
        # Legacy keys kept for compare-models display
        "top_matched_terms": term_counts.head(20).to_dict(),
        "top_matched_groups": group_counts.head(10).to_dict(),
    }
    with open(outputs.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    _write_summary_md(summary, outputs.summary_md)

    print(f"\nDone. {len(confirmed_df)}/{total} rows matched ({summary['match_rate_pct']}%)")
    print(f"Cases: {confirmed_cases}/{total_cases} have a confirmed ICD match ({summary['case_match_rate_pct']}%)")
    print(f"Unique terms: {summary['unique_terms_matched']} | Imbalance: top={term_counts.iloc[0] if len(term_counts) else 0}, bottom={term_counts.iloc[-1] if len(term_counts) else 0}, single-match terms={summary['imbalance']['terms_with_1_match']}")
    print(f"Method counts: {method_counts}")
    return outputs
