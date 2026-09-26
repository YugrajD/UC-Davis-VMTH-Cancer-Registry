"""The bundle contract shared between ML (which builds it) and ml-worker (WP12,
which will consume it).

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
exercised now by ``handoff.exports.export_bundle`` and importable unchanged by
WP12 for the worker's own startup gate.
"""

from __future__ import annotations

from pathlib import Path

from generations.manifest import ManifestError, verify_manifest
from report_mapping.model.generation import GenerationError, generation_paths, verify_fingerprint

# Bundle-relative paths ml-worker's current env-var contract (ml-worker/app.py,
# batch_predict.py) points at today, mapped to what WP12 should resolve each
# one from instead: a path relative to the bundle root, via
# ``report_mapping.model.generation.generation_paths``. Per-group label_presence
# checkpoints aren't listed individually — WP12 lists label_presence_dir itself.
WORKER_ENV_VAR_TO_BUNDLE_PATH = {
    "PETBERT_MODEL_PATH": "petbert",
    "LABELS_CSV_PATH": "labels/labels.csv",
    "GROUP_CLASSIFIER_PATH": "checkpoints/group_classifier_best.pt",
    "CASE_PRESENCE_CLASSIFIER_PATH": "checkpoints/case_presence_classifier.pt",
    "LP_THRESHOLDS_JSON_PATH": "checkpoints/label_presence/lp_thresholds.json",
    "UNCOMMON_GROUPS_PATH": "checkpoints/uncommon_groups.txt",
}


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
    "ManifestError", "GenerationError", "WORKER_ENV_VAR_TO_BUNDLE_PATH",
    "verify_worker_bundle", "required_paths",
]
