"""The bundle contract shared between ML (which builds it) and ml-worker (which
consumes it), plus the worker's upload and response formats.

A bundle is just a report-mapping generation directory (``current/``) —
``report_mapping.model.generation``'s docstring already says so: "This layout
*is* the cloud bundle layout ml-worker downloads." This module adds nothing to
that layout; it names the subset of it ml-worker actually needs to serve
predictions, and the one check both sides must run before trusting a bundle:
manifest file-hashes + embedding fingerprint, reusing
``generations.manifest.verify_manifest`` and
``report_mapping.model.generation.verify_fingerprint`` rather than
re-implementing either (per CLAUDE.md, "call the public interface").

ml-rewrite-plan.md's Decisions: "The worker refuses to start if a file listed
in its manifest is missing" — ``verify_worker_bundle`` below is that check,
run by ``handoff.exports.export_bundle``; the worker gets the same check from
``load_generation`` at startup.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

from generations.manifest import ManifestError, verify_manifest
from report_mapping import sections
from report_mapping.model.generation import GenerationError, generation_paths, verify_fingerprint

# ml-worker's env-var contract (ml-worker/app.py, batch_predict.py), mapped to
# the path each variable must name relative to the bundle root
# (``report_mapping.model.generation.generation_paths``); ``resolve_bundle_root``
# refuses any that points elsewhere. Per-group label_presence checkpoints have
# no variable: the worker loads the whole label_presence dir.
WORKER_ENV_VAR_TO_BUNDLE_PATH = {
    "PETBERT_MODEL_PATH": "petbert",  # app.py's name for the model path
    "MODEL_PATH": "petbert",  # batch_predict.py's name for it
    "LABELS_CSV_PATH": "labels/labels.csv",
    "GROUP_CLASSIFIER_PATH": "checkpoints/group_classifier_best.pt",
    "CASE_PRESENCE_CLASSIFIER_PATH": "checkpoints/case_presence_classifier.pt",
    "LP_THRESHOLDS_JSON_PATH": "checkpoints/label_presence/lp_thresholds.json",
    "UNCOMMON_GROUPS_PATH": "checkpoints/uncommon_groups.txt",
}


UPLOAD_ID_COL = "anon_id"
UPLOAD_TEXT_COL = "Text"


class BundleError(Exception):
    """The worker's env vars do not describe one bundle."""


def resolve_bundle_root(env: Mapping[str, str], model_var: str) -> Path:
    """The bundle root from the worker's env vars: the parent of the model path (``<root>/petbert``).

    Every other path var that is set must point at its own place in that bundle, so a deployment
    that mixes files from two generations refuses to start rather than predicting with them."""
    root = Path(env[model_var]).resolve().parent
    wrong = sorted(var for var, rel in WORKER_ENV_VAR_TO_BUNDLE_PATH.items()
                   if env.get(var) and Path(env[var]).resolve() != root / rel)
    if wrong:
        raise BundleError(f"{', '.join(wrong)} must point inside the bundle at {root} "
                          f"({', '.join(WORKER_ENV_VAR_TO_BUNDLE_PATH[v] for v in wrong)})")
    return root


def upload_to_reports(upload: pd.DataFrame) -> pd.DataFrame:
    """An upload (Dataset A: ``anon_id`` + one ``Text`` column) as report columns for
    ``sections.build_section_frame``. As the legacy worker did, Text fills every section: the first
    source column of each section gets Text and the rest stay empty, so section 1 (FINAL COMMENT +
    COMMENT) is Text alone. Source columns the upload already has are kept."""
    reports = upload.copy()
    text = reports[UPLOAD_TEXT_COL] if UPLOAD_TEXT_COL in reports.columns else ""
    for group in sections.CONCAT_3_SECTIONS:
        for i, column in enumerate(group):
            if column not in reports.columns:
                reports[column] = text if i == 0 else ""
    return reports


def response_rows(rows: list[dict], text_by_id: dict[str, str]) -> list[dict]:
    """One dict per case in the payload ``ingestion_service.parse_predictions`` reads: a case with
    several prediction rows gets "1) a 2) b" strings per field, as the legacy worker sent. Each
    carries ``source_version``, the generation_id that made it."""
    by_case: dict[str, list[dict]] = {}
    for row in rows:
        by_case.setdefault(row["case_id"], []).append(row)
    fields = ("predicted_term", "predicted_group", "predicted_code", "confidence", "method")
    out = []
    for case_id, case_rows in by_case.items():
        case_rows.sort(key=lambda r: int(r["diagnosis_index"]))
        if len(case_rows) == 1:
            values = {f: case_rows[0][f] for f in fields}
        else:
            values = {f: " ".join(f"{i}) {r[f]}" for i, r in enumerate(case_rows, start=1)) for f in fields}
        out.append({UPLOAD_ID_COL: case_id, "original_text": text_by_id.get(case_id, ""), **values,
                    "source_version": case_rows[0]["generation_id"]})
    return out


def verify_worker_bundle(bundle_root: str | Path) -> dict:
    """Verify ``bundle_root``'s manifest (file hashes) and embedding fingerprint —
    the same refusal ml-worker must perform at startup before loading anything.
    Returns the manifest on success; raises ``ManifestError``/``GenerationError``
    naming what's missing or mismatched."""
    bundle_root = Path(bundle_root)
    verify_manifest(bundle_root)
    return verify_fingerprint(bundle_root)


def required_paths(bundle_root: str | Path) -> list[Path]:
    """The files/dirs ml-worker must find under ``bundle_root`` (the same set
    ``report_mapping.model.generation.load_generation`` loads), for a quick
    existence check without loading any model weights."""
    paths = generation_paths(bundle_root)
    return [
        paths.petbert_dir, paths.labels_csv, paths.case_presence_pt, paths.group_pt,
        paths.label_presence_dir, paths.lp_thresholds_json, paths.thresholds_json,
    ]


__all__ = [
    "ManifestError", "GenerationError", "BundleError", "WORKER_ENV_VAR_TO_BUNDLE_PATH",
    "UPLOAD_ID_COL", "UPLOAD_TEXT_COL", "resolve_bundle_root", "upload_to_reports", "response_rows",
    "verify_worker_bundle", "required_paths",
]
