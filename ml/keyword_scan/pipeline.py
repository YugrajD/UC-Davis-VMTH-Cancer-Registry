"""Keyword-only pipeline for mapping diagnosis text to Vet-ICD-O taxonomy labels.

No ML model is required. Each diagnosis is scanned for taxonomy term keywords
using word-boundary regex patterns. The longest matching term wins.

Flow:
  1. Load diagnoses CSV (case_id, diagnosis_number, diagnosis).
  2. Load Vet-ICD-O taxonomy from labels.csv.
  3. Build a keyword index: normalized term strings → (pattern, taxonomy index).
  4. For each diagnosis row, find the first (longest) keyword match.
  5. Write keyword_predictions.csv and keyword_summary.json.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

import pandas as pd

from labels.taxonomy import TaxonomyLabel, load_labels_taxonomy

# Trailing qualifiers in taxonomy terms that are stripped when building the
# core keyword (e.g. "Hemangioma, NOS" → core keyword "hemangioma").
_QUALIFIER_RE = re.compile(
    r",?\s*\b("
    r"nos|nec|conventional|well differentiated|spindle cell|kaposiform|"
    r"epithelioid|inflammatory lobular capillary|mixed capillary cavernous|"
    r"retiform|malignant|benign|presumptive|adult type|juvenile type|atypical"
    r")\b.*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class KeywordConfig:
    csv_path: str
    id_col: str
    diag_num_col: str
    text_col: str
    labels_csv_path: str
    out_dir: str
    max_rows: int | None


@dataclass(frozen=True)
class KeywordOutputs:
    predictions_csv: str
    summary_json: str


def _normalize(text: str) -> str:
    """Lowercase, collapse hyphens/underscores to spaces, normalize whitespace."""
    text = text.lower()
    text = re.sub(r"[-_]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _build_keyword_index(
    taxonomy_labels: list[TaxonomyLabel],
) -> list[tuple[str, re.Pattern, int]]:
    """Build (core_keyword, pattern, taxonomy_index) sorted by keyword length descending.

    Two keyword candidates are generated per label:
      - Full normalized term  (e.g. "hemangioma, nos")
      - Core term with qualifiers stripped  (e.g. "hemangioma")

    Longer keywords are tried first so more specific terms take priority.
    Duplicate keyword strings are skipped — the first label to define a keyword
    wins, which corresponds to the Preferred term in the taxonomy CSV.
    """
    entries: list[tuple[str, re.Pattern, int]] = []
    seen: set[str] = set()

    for i, label in enumerate(taxonomy_labels):
        norm = _normalize(label.term)
        core = _QUALIFIER_RE.sub("", norm).strip().strip(",").strip()

        for kw in {norm, core}:
            kw = kw.strip()
            if len(kw) < 6 or kw in seen:
                continue
            seen.add(kw)
            pat = re.compile(r"\b" + re.escape(kw) + r"\b")
            entries.append((kw, pat, i))

    entries.sort(key=lambda x: len(x[0]), reverse=True)
    return entries


def _match_diagnosis(
    text: str,
    keyword_index: list[tuple[str, re.Pattern, int]],
    taxonomy_labels: list[TaxonomyLabel],
) -> tuple[str, str, str, str, str]:
    """Return (matched_term, matched_group, matched_code, matched_keyword, method).

    method is "keyword" on a match, "no_match" otherwise.
    """
    norm_text = _normalize(text)
    for kw, pattern, label_idx in keyword_index:
        if pattern.search(norm_text):
            label = taxonomy_labels[label_idx]
            return label.term, label.group, label.code, kw, "keyword"
    return "", "", "", "", "no_match"


def run_keyword_scan(config: KeywordConfig) -> KeywordOutputs:
    """Execute the keyword-only diagnosis categorization pipeline."""
    os.makedirs(config.out_dir, exist_ok=True)
    predictions_csv = os.path.join(config.out_dir, "keyword_predictions.csv")
    summary_json = os.path.join(config.out_dir, "keyword_summary.json")

    # Load diagnoses
    df = pd.read_csv(config.csv_path, encoding="latin-1")
    df.columns = [col.lstrip("\ufeff").lstrip("ï»¿") for col in df.columns]
    if config.max_rows is not None:
        df = df.head(config.max_rows).copy()

    for col in [config.id_col, config.text_col]:
        if col not in df.columns:
            raise ValueError(f"Column {col!r} not found. Available: {df.columns.tolist()}")

    # Load taxonomy and build keyword index
    taxonomy_labels = load_labels_taxonomy(config.labels_csv_path)
    keyword_index = _build_keyword_index(taxonomy_labels)

    # Match each row
    results = []
    for _, row in df.iterrows():
        text = str(row[config.text_col]) if pd.notna(row[config.text_col]) else ""
        matched_term, matched_group, matched_code, matched_keyword, method = (
            _match_diagnosis(text, keyword_index, taxonomy_labels)
        )
        result = {config.id_col: row[config.id_col]}
        if config.diag_num_col in df.columns:
            result[config.diag_num_col] = row[config.diag_num_col]
        result.update({
            config.text_col: text,
            "matched_term": matched_term,
            "matched_group": matched_group,
            "matched_code": matched_code,
            "matched_keyword": matched_keyword,
            "method": method,
        })
        results.append(result)

    out_df = pd.DataFrame(results)
    out_df.to_csv(predictions_csv, index=False)

    # Summary
    method_counts = out_df["method"].value_counts().to_dict()
    matched_df = out_df[out_df["method"] == "keyword"]
    summary = {
        "csv_path": config.csv_path,
        "total_rows": len(out_df),
        "method_counts": method_counts,
        "match_rate_pct": round(100 * method_counts.get("keyword", 0) / max(len(out_df), 1), 1),
        "top_matched_terms": matched_df["matched_term"].value_counts().head(20).to_dict(),
        "top_matched_groups": matched_df["matched_group"].value_counts().head(10).to_dict(),
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return KeywordOutputs(predictions_csv=predictions_csv, summary_json=summary_json)
