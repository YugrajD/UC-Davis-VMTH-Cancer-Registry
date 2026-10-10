"""What ml-worker shares with the ml suite: where the bundle is, the upload -> text mapping, the response payload.

A bundle is a report-mapping generation directory (``current/``, see ``report_mapping.generation``). The
worker loads it with ``report_mapping.inference.predict.load_generation``, which runs
``report_mapping.generation.verify_spec`` first (file hashes, architecture incl. text spec, calibration bound
to the weights) and refuses a bad bundle; this module adds nothing to that check.

Only the model-path variable locates the bundle. The backend and docker-compose keep naming it
``<bundle>/petbert`` (the PetBERT-era layout); the generation has no ``petbert/`` directory any more
(``encoder/`` replaced it), so that path need not exist and only its parent is used.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

import pandas as pd

from report_mapping.text import clean_text

# The model-path variables of ml-worker (app.py reads the first, batch_predict.py the second). Both
# are the bundle root plus this leaf name, a name kept so the backend's contract does not change.
MODEL_PATH_VARS = ("PETBERT_MODEL_PATH", "MODEL_PATH")
MODEL_PATH_LEAF = "petbert"

UPLOAD_ID_COL = "anon_id"
UPLOAD_TEXT_COL = "Text"

THREADS_VAR = "TORCH_NUM_THREADS"
# 8 is what the CPU timings were measured with; fewer cores use all they have.
DEFAULT_THREADS = 8


class BundleError(Exception):
    """The worker's env vars do not describe one bundle."""


def resolve_bundle_root(env: Mapping[str, str], model_var: str) -> Path:
    """The bundle root from the worker's env vars: the parent of the model path (``<root>/petbert``).

    Both model-path variables, if set, must name that same ``<root>/petbert``, so a deployment that points
    the two at different bundles refuses to start. The path itself need not exist. The other path variables
    of the PetBERT-era worker (``LABELS_CSV_PATH``, ``GROUP_CLASSIFIER_PATH``, ``CASE_PRESENCE_CLASSIFIER_PATH``,
    ``LP_THRESHOLDS_JSON_PATH``, ``UNCOMMON_GROUPS_PATH``) are not read any more and are ignored wherever
    they point."""
    root = Path(env[model_var]).resolve().parent
    wrong = sorted(var for var in MODEL_PATH_VARS if env.get(var) and Path(env[var]).resolve() != root / MODEL_PATH_LEAF)
    if wrong:
        raise BundleError(f"{', '.join(wrong)} must be {root / MODEL_PATH_LEAF} (the bundle root is the parent of {model_var})")
    return root


def torch_threads(env: Mapping[str, str]) -> int:
    """CPU threads for torch: ``TORCH_NUM_THREADS`` if set, else ``min(DEFAULT_THREADS, cpu count)``."""
    raw = env.get(THREADS_VAR)
    if raw:
        try:
            threads = int(raw)
        except ValueError:
            threads = 0
        if threads < 1:
            raise BundleError(f"{THREADS_VAR} must be a positive integer, got {raw!r}")
        return threads
    return min(DEFAULT_THREADS, os.cpu_count() or 1)


def upload_to_texts(upload: pd.DataFrame) -> tuple[list[str], list[str]]:
    """An upload (Dataset A: ``anon_id`` + one ``Text`` column) as (case ids, report texts), row by row.
    The text is ``Text`` cleaned and flat: no heading, no section columns (measured on calibration, restructure
    plan WP4). A missing or blank ``Text`` is the empty text, which gets no prediction rows. Other columns are ignored."""
    ids = upload[UPLOAD_ID_COL].map(clean_text).tolist()
    texts = upload[UPLOAD_TEXT_COL].map(clean_text).tolist() if UPLOAD_TEXT_COL in upload.columns else [""] * len(ids)
    return ids, texts


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


__all__ = [
    "BundleError", "MODEL_PATH_VARS", "MODEL_PATH_LEAF", "THREADS_VAR", "DEFAULT_THREADS", "UPLOAD_ID_COL",
    "UPLOAD_TEXT_COL", "resolve_bundle_root", "torch_threads", "upload_to_texts", "response_rows",
]
