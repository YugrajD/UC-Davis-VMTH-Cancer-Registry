"""Ensemble verification cleanup pass over confirmed annotations.

Reads llm_annotation.csv. For every confirmed positive (Exact / Fuzzy / LLM),
sends the row to two diverse local LM-Studio / Ollama models. Each model
responds with one of: CORRECT, WRONG_should_be:<term>, WRONG_no_cancer, UNCERTAIN.
Resolution rules:
  - Both CORRECT                                  -> keep
  - Both WRONG_no_cancer                          -> set to No Match
  - Both WRONG_should_be:<X> (same X)             -> replace with X
  - Disagreement                                  -> optional tiebreaker model;
                                                     otherwise mark Uncertain

Writes:
  llm_annotation_cleaned.csv  -full row set with cleaned values
  cleanup_diff.csv            -only changed rows with before/after columns
  cleanup_summary.json        -aggregate change statistics

LLM use is training-time only. Production is unaffected.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass

import pandas as pd

from ICD_labels import TaxonomyLabel, load_labels_taxonomy
from annotation.llm_pipeline.client import chat
from annotation.llm_pipeline.pipeline import _extract_site


_CONFIRMED_METHODS = {"Exact", "Fuzzy", "LLM"}

# Match a verdict line. First try line-anchored (well-behaved responses);
# `_parse_verdict` falls back to a non-anchored scan for chain-of-thought
# preambles like "Looking at this... CORRECT".
_VERDICT_LINE_RE = re.compile(
    r"^\s*(CORRECT|WRONG_no_cancer|WRONG_should_be\s*:\s*(.+?)|UNCERTAIN)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_VERDICT_FALLBACK_RE = re.compile(
    r"\b(CORRECT|WRONG_no_cancer|WRONG_should_be\s*:\s*([^\n]+?)|UNCERTAIN)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CleanupConfig:
    input_csv: str
    output_csv: str
    diff_csv: str
    summary_json: str
    labels_csv_path: str
    models: list[str]
    tiebreaker_model: str | None
    timeout: int = 60
    max_rows: int | None = None
    methods_to_verify: frozenset[str] = frozenset(_CONFIRMED_METHODS)


@dataclass
class _Verdict:
    kind: str               # "CORRECT" | "WRONG_no_cancer" | "WRONG_should_be" | "UNCERTAIN" | "PARSE_ERROR"
    proposed_term: str = ""  # populated when kind == "WRONG_should_be"


def _build_verification_prompt(
    diagnosis: str,
    matched_term: str,
    matched_group: str,
    matched_code: str,
    candidates: list[TaxonomyLabel],
) -> str:
    site = _extract_site(diagnosis)
    site_line = f"Anatomic site context: {site}\n" if site else ""
    candidate_block = "\n".join(f"  {i + 1}. {c.term}" for i, c in enumerate(candidates))
    return (
        "You are reviewing a single veterinary pathology diagnosis annotation for accuracy.\n\n"
        f'Diagnosis text: "{diagnosis}"\n'
        f"{site_line}\n"
        "Currently annotated as:\n"
        f"  Term:  {matched_term}\n"
        f"  Group: {matched_group}\n"
        f"  Code:  {matched_code}\n\n"
        "Candidate terms in this group:\n"
        f"{candidate_block}\n\n"
        "Decide whether the current annotation is correct, wrong, non-cancer, or uncertain.\n\n"
        "Respond with EXACTLY one line, in one of these four forms:\n"
        "  CORRECT\n"
        "  WRONG_should_be: <exact candidate term, copied character-for-character>\n"
        "  WRONG_no_cancer\n"
        "  UNCERTAIN\n\n"
        "Decision rules:\n"
        "- If the current term accurately captures the diagnosis (right group, right subtype, right behavior), reply: CORRECT\n"
        "- If a different candidate is a better match (e.g. correct T-cell vs B-cell, correct anatomic-site variant, correct benign vs malignant), reply: WRONG_should_be: <candidate>\n"
        "- If the diagnosis is non-neoplastic -inflammation, infection, degeneration, hyperplasia, fibrosis without tumor language, or explicitly negated (\"no evidence of\", \"non-neoplastic\") -reply: WRONG_no_cancer\n"
        "- If the diagnosis is hedged (\"suspect\", \"presumed\", \"likely\", \"rule out\", \"versus\", \"vs.\") or you cannot tell, reply: UNCERTAIN\n\n"
        "Your decision:"
    )


def _parse_verdict(response: str, candidate_terms: set[str]) -> _Verdict:
    """Extract the first valid verdict from the LLM response.

    First scans for line-anchored verdicts (well-behaved responses); falls back
    to a non-anchored scan so chain-of-thought preambles ("Looking at this...
    CORRECT") still parse.
    """
    if not response:
        return _Verdict("PARSE_ERROR")
    for regex in (_VERDICT_LINE_RE, _VERDICT_FALLBACK_RE):
        for match in regex.finditer(response):
            head = match.group(1).strip().upper()
            if head == "CORRECT":
                return _Verdict("CORRECT")
            if head == "UNCERTAIN":
                return _Verdict("UNCERTAIN")
            if head == "WRONG_NO_CANCER":
                return _Verdict("WRONG_no_cancer")
            proposed_raw = match.group(2)
            if proposed_raw is None:
                continue
            proposed = proposed_raw.strip().strip('"').strip("'").rstrip(".")
            for term in candidate_terms:
                if term.lower() == proposed.lower():
                    return _Verdict("WRONG_should_be", proposed_term=term)
            from difflib import get_close_matches
            close = get_close_matches(proposed, list(candidate_terms), n=1, cutoff=0.8)
            if close:
                return _Verdict("WRONG_should_be", proposed_term=close[0])
            return _Verdict("UNCERTAIN")
    return _Verdict("PARSE_ERROR")


def _query_model(
    model: str,
    prompt: str,
    candidate_terms: set[str],
    timeout: int,
) -> _Verdict:
    try:
        response = chat(prompt, model=model, timeout=timeout)
    except Exception:
        return _Verdict("PARSE_ERROR")
    return _parse_verdict(response, candidate_terms)


def _resolve(verdicts: list[_Verdict]) -> _Verdict:
    """Apply resolution rules to a list of N verdicts.

    Unanimous agreement (all CORRECT, all WRONG_no_cancer, or all
    WRONG_should_be with same proposed term) wins; otherwise UNCERTAIN.
    PARSE_ERROR rows are excluded from the agreement check -if any model
    parse-errored and the rest don't unanimously agree, the row goes to
    UNCERTAIN.
    """
    valid = [v for v in verdicts if v.kind != "PARSE_ERROR"]
    if len(valid) < len(verdicts):
        # At least one parse error -require remaining to unanimously agree
        if not valid:
            return _Verdict("UNCERTAIN")
    kinds = {v.kind for v in valid}
    if len(kinds) != 1:
        return _Verdict("UNCERTAIN")
    kind = kinds.pop()
    if kind == "CORRECT":
        return _Verdict("CORRECT")
    if kind == "WRONG_no_cancer":
        return _Verdict("WRONG_no_cancer")
    if kind == "WRONG_should_be":
        proposed_terms = {v.proposed_term for v in valid}
        if len(proposed_terms) == 1:
            return _Verdict("WRONG_should_be", proposed_term=proposed_terms.pop())
        return _Verdict("UNCERTAIN")
    if kind == "UNCERTAIN":
        return _Verdict("UNCERTAIN")
    return _Verdict("UNCERTAIN")


def _build_group_index(taxonomy: list[TaxonomyLabel]) -> dict[str, list[TaxonomyLabel]]:
    idx: dict[str, list[TaxonomyLabel]] = {}
    for label in taxonomy:
        idx.setdefault(label.group, []).append(label)
    return idx


def run_cleanup(config: CleanupConfig) -> dict:
    os.makedirs(os.path.dirname(config.output_csv) or ".", exist_ok=True)

    df = pd.read_csv(config.input_csv)
    if config.max_rows is not None:
        df = df.head(config.max_rows).copy()
    print(f"Loaded {len(df):,} rows from {config.input_csv}")

    taxonomy = load_labels_taxonomy(config.labels_csv_path)
    group_to_labels = _build_group_index(taxonomy)
    term_to_label = {l.term: l for l in taxonomy}

    confirmed_mask = df["method"].isin(config.methods_to_verify)
    confirmed_idx = df.index[confirmed_mask].tolist()
    print(f"Verifying {len(confirmed_idx):,} rows (methods: {sorted(config.methods_to_verify)})")
    print(f"Verifier models: {config.models}"
          + (f" + tiebreaker {config.tiebreaker_model}" if config.tiebreaker_model else ""))

    out_df = df.copy()
    diff_rows: list[dict] = []
    counters = {
        "kept": 0,
        "term_changed": 0,
        "group_changed": 0,
        "set_no_match": 0,
        "flagged_uncertain": 0,
        "parse_errors": 0,
        "total_verified": 0,
        "tiebreaker_used": 0,
    }
    t0 = time.time()

    for n, idx in enumerate(confirmed_idx, 1):
        row = df.loc[idx]
        diagnosis = str(row.get("diagnosis", ""))
        matched_term = str(row.get("matched_term", ""))
        matched_group = str(row.get("matched_group", ""))
        matched_code = str(row.get("matched_code", ""))
        method = str(row.get("method", ""))

        candidates = group_to_labels.get(matched_group, [])
        if not candidates:
            # Group not in taxonomy -should not happen; flag as uncertain
            _apply_uncertain(out_df, idx)
            counters["flagged_uncertain"] += 1
            diff_rows.append(_diff_row(row, "flagged_uncertain", reason="group not in taxonomy"))
            continue

        candidate_terms = {c.term for c in candidates}
        prompt = _build_verification_prompt(diagnosis, matched_term, matched_group, matched_code, candidates)

        verdicts = [_query_model(m, prompt, candidate_terms, config.timeout) for m in config.models]
        resolved = _resolve(verdicts)

        if resolved.kind == "UNCERTAIN" and config.tiebreaker_model:
            tb = _query_model(config.tiebreaker_model, prompt, candidate_terms, config.timeout)
            counters["tiebreaker_used"] += 1
            resolved = _resolve(verdicts + [tb])

        counters["total_verified"] += 1
        change_type = _apply_resolution(out_df, idx, resolved, term_to_label)
        counters[change_type] = counters.get(change_type, 0) + 1
        if change_type != "kept":
            diff_rows.append(_diff_row(row, change_type, resolved=resolved))

        if n % 50 == 0 or n == len(confirmed_idx):
            elapsed = time.time() - t0
            rate = n / elapsed if elapsed > 0 else 0
            eta = (len(confirmed_idx) - n) / rate if rate > 0 else 0
            print(
                f"[{n}/{len(confirmed_idx)}] "
                f"kept={counters['kept']} term_changed={counters['term_changed']} "
                f"set_no_match={counters['set_no_match']} uncertain={counters['flagged_uncertain']} "
                f"({rate:.1f} rows/s, ETA {eta/60:.1f} min)"
            )

    out_df.to_csv(config.output_csv, index=False)
    print(f"\nWrote cleaned annotations: {config.output_csv}")

    diff_df = pd.DataFrame(diff_rows)
    if not diff_df.empty:
        diff_df.to_csv(config.diff_csv, index=False)
        print(f"Wrote diff rows: {config.diff_csv}  ({len(diff_df):,} rows)")

    summary = {
        "input_csv": config.input_csv,
        "output_csv": config.output_csv,
        "models": config.models,
        "tiebreaker_model": config.tiebreaker_model,
        "counters": counters,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    with open(config.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary: {config.summary_json}")
    print(f"Counters: {counters}")
    return summary


def _apply_resolution(
    out_df: pd.DataFrame,
    idx: int,
    resolved: _Verdict,
    term_to_label: dict[str, TaxonomyLabel],
) -> str:
    """Apply the resolved verdict to out_df row idx in place. Returns change_type."""
    if resolved.kind == "CORRECT":
        return "kept"
    if resolved.kind == "WRONG_no_cancer":
        out_df.at[idx, "matched_term"] = ""
        out_df.at[idx, "matched_group"] = ""
        out_df.at[idx, "matched_code"] = ""
        out_df.at[idx, "matched_keyword"] = ""
        out_df.at[idx, "method"] = "No Match"
        out_df.at[idx, "confidence"] = 0.0
        return "set_no_match"
    if resolved.kind == "WRONG_should_be":
        new_label = term_to_label.get(resolved.proposed_term)
        if new_label is None:
            _apply_uncertain(out_df, idx)
            return "flagged_uncertain"
        prior_group = out_df.at[idx, "matched_group"]
        out_df.at[idx, "matched_term"] = new_label.term
        out_df.at[idx, "matched_group"] = new_label.group
        out_df.at[idx, "matched_code"] = new_label.code
        out_df.at[idx, "matched_keyword"] = new_label.term
        # Keep original method label so the cleanup origin can be traced.
        return "group_changed" if new_label.group != prior_group else "term_changed"
    # UNCERTAIN or PARSE_ERROR
    _apply_uncertain(out_df, idx)
    return "flagged_uncertain"


def _apply_uncertain(out_df: pd.DataFrame, idx: int) -> None:
    out_df.at[idx, "matched_term"] = ""
    out_df.at[idx, "matched_group"] = ""
    out_df.at[idx, "matched_code"] = ""
    out_df.at[idx, "matched_keyword"] = ""
    out_df.at[idx, "method"] = "Uncertain"
    out_df.at[idx, "confidence"] = 0.0


def _diff_row(
    row: pd.Series,
    change_type: str,
    resolved: _Verdict | None = None,
    reason: str = "",
) -> dict:
    d = {
        "case_id": row.get("case_id", ""),
        "diagnosis_number": row.get("diagnosis_number", ""),
        "diagnosis": row.get("diagnosis", ""),
        "before_term": row.get("matched_term", ""),
        "before_group": row.get("matched_group", ""),
        "before_code": row.get("matched_code", ""),
        "before_method": row.get("method", ""),
        "change_type": change_type,
    }
    if resolved is not None:
        d["after_term"] = resolved.proposed_term if resolved.kind == "WRONG_should_be" else ""
        d["resolved_kind"] = resolved.kind
    if reason:
        d["reason"] = reason
    return d
