"""Run the diagnosis-to-taxonomy cascade and write a versioned silver generation.

A silver generation is immutable once written, mirroring the split
generations in ``generations/splits.py``:
``config.SILVER_DIR/<silver_id>/annotation.csv`` + a ``manifest.json``
(``generations.manifest.write_manifest``). Every write function here refuses
to write into an existing ``<silver_id>/`` directory, and ``load_silver``
verifies the manifest before reading. The manifest is always written last, once
``annotation.csv`` is in its final form, since a manifest may not list files
added after it was written.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

import config
import io_utils
from generations.manifest import read_manifest, sha256_file, verify_manifest, write_manifest
from taxonomy.taxonomy import TaxonomyLabel, load_labels_taxonomy
from diagnosis_mapping import cleanup as cleanup_mod
from diagnosis_mapping import llm_tier
from diagnosis_mapping.keyword_tiers import (
    NO_MATCH,
    build_keyword_index,
    cascade_fingerprint,
    has_signal,
    mask_negation,
    normalize_llm,
    tier1_exact,
    tier2_fuzzy,
)

ANNOTATION_COLUMNS = [
    "case_id", "diagnosis_number", "diagnosis", "matched_term", "matched_group",
    "matched_code", "matched_keyword", "method", "confidence", "decision_stage",
]

ID_COL = "case_id"
DIAG_NUM_COL = "diagnosis_number"
TEXT_COL = "diagnosis"


@dataclass(frozen=True)
class CascadeConfig:
    """Controls whether Tier 3 actually calls the LLM (see llm_tier.run_tier3).

    ``llm_enabled=False`` runs every tier (including candidate assembly) but
    never calls the model - used for ``--no-llm`` runs and for the keyword-only
    legacy replay, where tier3-eligible rows are recorded exactly like a
    declined LLM call rather than skipped.
    """
    llm_enabled: bool = True
    llm_model: str | None = None
    llm_timeout: int = 60


def match_diagnosis(
    original_text: str,
    keyword_index,
    taxonomy_labels: list[TaxonomyLabel],
    oma_index: dict[str, int],
    group_token_index: dict[str, set[str]],
    cascade_config: CascadeConfig,
    counters: dict,
):
    """Run one diagnosis through Tier 1 -> Tier 2 -> Tier 3.

    Ported from ``ml/annotation/llm_pipeline/pipeline.py::_match_diagnosis``.
    """
    norm_text = normalize_llm(original_text)
    masked_text = mask_negation(norm_text)

    result = tier1_exact(masked_text, keyword_index, taxonomy_labels)
    if result:
        return result

    result = tier2_fuzzy(masked_text, taxonomy_labels)
    if result:
        return result

    if has_signal(masked_text):
        counters["signal_rows"] = counters.get("signal_rows", 0) + 1
        return llm_tier.run_tier3(
            original_text, masked_text, group_token_index, taxonomy_labels, oma_index,
            llm_enabled=cascade_config.llm_enabled, model=cascade_config.llm_model,
            timeout=cascade_config.llm_timeout, counters=counters,
        )

    return NO_MATCH


def run_cascade(
    df: pd.DataFrame,
    taxonomy_labels: list[TaxonomyLabel],
    cascade_config: CascadeConfig = CascadeConfig(),
) -> tuple[pd.DataFrame, dict]:
    """Run the cascade over every row of a diagnoses-shaped dataframe.

    ``df`` must have ``case_id`` and ``diagnosis`` columns (``diagnosis_number``
    is carried through if present). Returns (annotation-shaped dataframe, tier
    counters).
    """
    keyword_index = build_keyword_index(taxonomy_labels)
    oma_index = llm_tier.build_oma_index(taxonomy_labels)
    group_token_index = llm_tier.build_group_token_index(taxonomy_labels)

    counters: dict = {
        "signal_rows": 0, "tier3_calls": 0, "tier3_matched": 0,
        "tier3_uncertain": 0, "tier3_no_match": 0,
    }
    rows: list[dict] = []
    for _, row in df.iterrows():
        text = str(row[TEXT_COL]) if pd.notna(row[TEXT_COL]) else ""
        match = match_diagnosis(
            text, keyword_index, taxonomy_labels, oma_index, group_token_index,
            cascade_config, counters,
        )
        entry: dict = {ID_COL: row[ID_COL]}
        if DIAG_NUM_COL in df.columns:
            entry[DIAG_NUM_COL] = row[DIAG_NUM_COL]
        entry.update({
            TEXT_COL: text,
            "matched_term": match.term,
            "matched_group": match.group,
            "matched_code": match.code,
            "matched_keyword": match.keyword,
            "method": match.method,
            "confidence": match.confidence,
            "decision_stage": match.stage,
        })
        rows.append(entry)

    return pd.DataFrame(rows, columns=ANNOTATION_COLUMNS), counters


# ---------------------------------------------------------------------------
# Silver generation directories (write-once)
# ---------------------------------------------------------------------------


def silver_dir(silver_id: str) -> Path:
    return config.SILVER_DIR / silver_id


def _new_silver_dir(silver_id: str) -> Path:
    directory = silver_dir(silver_id)
    if directory.exists():
        raise FileExistsError(f"silver generation {silver_id!r} already exists at {directory}; silver generations are immutable")
    directory.mkdir(parents=True)
    return directory


class NoLLMGenerationError(Exception):
    """A --no-llm silver generation was loaded without allow_no_llm=True."""


def load_silver(silver_id: str, *, allow_no_llm: bool = False) -> pd.DataFrame:
    """Verify the silver generation's manifest, then load its annotation.csv.

    Refuses a generation whose manifest records ``llm_enabled: false`` unless
    ``allow_no_llm=True`` is passed explicitly. A --no-llm run records every
    Tier-3-eligible row as a declined LLM match (tier3_llm/"No Match"), which
    the coding rule treats as decisive non-cancer - consumers such as the
    coding rule and the Tier-3 sampler must not pick one up by accident.
    """
    directory = silver_dir(silver_id)
    verify_manifest(directory)
    manifest = read_manifest(directory)
    if manifest.get("llm_enabled") is False and not allow_no_llm:
        raise NoLLMGenerationError(
            f"silver generation {silver_id!r} was produced with --no-llm: every Tier-3-eligible "
            "row is recorded as a declined LLM match (tier3_llm/\"No Match\"), which the coding "
            "rule treats as decisive non-cancer. Pass allow_no_llm=True to load it anyway "
            "(e.g. for inspection or a keyword-only replay)."
        )
    return io_utils.read_csv(directory / "annotation.csv", encoding="utf-8", dtype=str, keep_default_na=False)


def run(
    silver_id: str,
    *,
    diagnoses_csv: Path | None = None,
    labels_csv: Path | None = None,
    llm_enabled: bool = True,
    llm_model: str | None = None,
    llm_timeout: int = 60,
    cleanup_enabled: bool = False,
    cleanup_models: list[str] | None = None,
    cleanup_tiebreaker: str | None = None,
    cleanup_timeout: int = 60,
) -> dict:
    """Run the cascade (and optionally the cleanup pass) over ``diagnoses_csv``
    and write the result as a new silver generation. Returns the manifest."""
    diagnoses_csv = Path(diagnoses_csv) if diagnoses_csv is not None else config.DIAGNOSES_CSV
    labels_csv = Path(labels_csv) if labels_csv is not None else config.LABELS_CSV

    diagnoses_df = io_utils.read_csv(diagnoses_csv)
    taxonomy_labels = load_labels_taxonomy(str(labels_csv))

    cascade_config = CascadeConfig(llm_enabled=llm_enabled, llm_model=llm_model, llm_timeout=llm_timeout)
    out_df, _counters = run_cascade(diagnoses_df, taxonomy_labels, cascade_config)

    if cleanup_enabled:
        cleanup_config = cleanup_mod.CleanupConfig(
            models=cleanup_models or [], tiebreaker_model=cleanup_tiebreaker, timeout=cleanup_timeout,
        )
        out_df, _diff_df, _cleanup_counters = cleanup_mod.clean(out_df, taxonomy_labels, cleanup_config)

    out_df = out_df.copy()
    out_df["silver_generation"] = silver_id

    directory = _new_silver_dir(silver_id)
    io_utils.write_csv(out_df, directory / "annotation.csv")

    counts = out_df["decision_stage"].value_counts().to_dict()
    return write_manifest(directory, {
        "silver_id": silver_id,
        "parent": None,
        "source": "cascade",
        "cascade_constants_sha256": cascade_fingerprint(taxonomy_labels),
        "llm_enabled": llm_enabled,
        "llm_model": llm_model,
        "cleanup_enabled": cleanup_enabled,
        "cleanup_models": cleanup_models if cleanup_enabled else None,
        "cleanup_tiebreaker": cleanup_tiebreaker if cleanup_enabled else None,
        "input_csv_sha256": sha256_file(diagnoses_csv),
        "taxonomy_sha256": sha256_file(labels_csv),
        "decision_stage_counts": counts,
    })
