"""A report-mapping generation directory (``current/``, ``candidate/``): layout names, manifest, spec check.

No torch here: promote, export and the S3 publish read generations through this module without loading a model.

    encoder/          HF ModernBERT weights (model.safetensors) + tokenizer files + config.json
    head.pt           term head (fp32 state_dict)
    gate_head.pt      gate head (fp32 state_dict)
    terms.json        term vocabulary, head order
    term_info.json    term -> [group, code], a snapshot of the taxonomy (inference never reads the taxonomy)
    thresholds.json   gate_t, tau, rho, review_cut, max_fp_share, review_target_share (written by calibration)
    manifest.json

``verify_spec`` is the one check every loader runs: file hashes match the manifest, the generation's
architecture (including the text spec) equals the code's, and the thresholds were calibrated for these weights.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import config
from generations.manifest import ManifestError, read_manifest, sha256_file, update_manifest, verify_manifest, write_manifest
from report_mapping.text import TEXT_SPEC_VERSION

ENCODER_DIR = "encoder"
ENCODER_WEIGHTS = "encoder/model.safetensors"
HEAD_FILE = "head.pt"
GATE_HEAD_FILE = "gate_head.pt"
TERMS_FILE = "terms.json"
TERM_INFO_FILE = "term_info.json"
THRESHOLDS_FILE = "thresholds.json"
# What weights_sha256 covers: everything a retrain changes and a recalibration does not.
WEIGHT_FILES = (ENCODER_WEIGHTS, HEAD_FILE, GATE_HEAD_FILE, TERMS_FILE)

KIND = "report_mapping_generation"
SCHEMA_VERSION = 2
# Changing any of these is an architectural change (promote refuses it; adopt takes it with a reason).
ARCHITECTURE = {
    "name": "joint-modernbert-v1",
    "text_spec_version": TEXT_SPEC_VERSION,
    "pooling": "mean",
    "heads": ["term", "gate"],
    "decision_rule": "gate-top1-rank2-v1",
}
THRESHOLD_KEYS = ("gate_t", "tau", "rho", "review_cut", "max_fp_share", "review_target_share")


class GenerationError(Exception):
    """A generation directory is missing, malformed, or fails the spec check; refuse rather than load it."""


def new_generation_id(now: datetime | None = None) -> str:
    """A unique generation_id from the UTC training time, e.g. ``gen-20261009T031500Z``. It is also the
    cloud's ``source_version``, so it needs no state shared between machines."""
    return f"gen-{(now or datetime.now(timezone.utc)):%Y%m%dT%H%M%SZ}"


def resolve_generation_dir(name: str | Path) -> Path:
    """"current" / "candidate" -> config paths; anything else is used as a literal directory."""
    if name == "current":
        return config.REPORT_MAPPING_CURRENT_DIR
    if name == "candidate":
        return config.REPORT_MAPPING_CANDIDATE_DIR
    return Path(name)


def weights_digest(file_hashes: dict[str, str]) -> str:
    """sha256 over the sha256 of each weight file, in ``WEIGHT_FILES`` order."""
    digest = hashlib.sha256()
    for name in WEIGHT_FILES:
        digest.update(f"{name}:{file_hashes[name]}\n".encode())
    return digest.hexdigest()


def weights_sha256(directory: Path) -> str:
    return weights_digest({name: sha256_file(directory / name) for name in WEIGHT_FILES})


def read_terms(directory: Path) -> list[str]:
    return json.loads((directory / TERMS_FILE).read_text(encoding="utf-8"))


def read_term_info(directory: Path) -> dict[str, tuple[str, str]]:
    info = json.loads((directory / TERM_INFO_FILE).read_text(encoding="utf-8"))
    return {term: (group, code) for term, (group, code) in info.items()}


def read_thresholds(directory: Path) -> dict:
    values = json.loads((directory / THRESHOLDS_FILE).read_text(encoding="utf-8"))
    missing = [k for k in THRESHOLD_KEYS if k not in values]
    if missing:
        raise GenerationError(f"{directory / THRESHOLDS_FILE} lacks {missing}")
    return values


def write_pending_manifest(directory: Path, *, encoder_id: str, max_length: int, recipe: dict, parents: dict,
                           training: dict) -> dict:
    """The manifest of a freshly trained generation: calibration "pending", no thresholds.json yet.
    ``directory`` already holds the encoder, both heads, terms.json and term_info.json."""
    return write_manifest(directory, {
        "kind": KIND, "schema_version": SCHEMA_VERSION, "generation_id": new_generation_id(), "status": "candidate",
        "architecture": ARCHITECTURE,
        "model": {"encoder_id": encoder_id, "weights_sha256": weights_sha256(directory),
                  "n_terms": len(read_terms(directory)), "max_length": max_length},
        "recipe": recipe, "parents": parents, "training": training,
        "calibration": {"status": "pending"},
    })


def write_calibration(directory: Path, thresholds: dict, block: dict) -> dict:
    """Write thresholds.json and the manifest's calibrated block (bound to the generation's weights).
    The manifest is rewritten last: until then the new file fails verification and nothing loads it."""
    (directory / THRESHOLDS_FILE).write_text(json.dumps(thresholds, indent=2) + "\n", encoding="utf-8", newline="\n")
    old = read_manifest(directory)
    fields = {k: v for k, v in old.items() if k not in ("files", "created_at", "git_sha")}
    fields["calibration"] = {**block, "status": "calibrated", "weights_sha256": old["model"]["weights_sha256"],
                             "calibrated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    write_manifest(directory, fields)
    return update_manifest(directory, {"created_at": old["created_at"], "git_sha": old["git_sha"]})


def copy_for_recalibration(source: Path, dest: Path) -> dict:
    """Copy the calibrated generation ``source`` to ``dest`` as a new generation with the same weights: a new
    generation_id, calibration "pending", no thresholds.json; weights_sha256, architecture and parents unchanged.
    New thresholds for the same weights are a new generation, so ``calibrate`` can then fit ``dest``."""
    source, dest = Path(source), Path(dest)
    old = verify_spec(source)
    if dest.exists():
        raise GenerationError(f"{dest} already exists")
    shutil.copytree(source, dest, ignore=shutil.ignore_patterns(THRESHOLDS_FILE, "manifest.json"))
    fields = {k: v for k, v in old.items() if k not in ("files", "created_at", "git_sha", "adoption")}
    return write_manifest(dest, {**fields, "generation_id": new_generation_id(), "status": "candidate",
                                 "recalibrated_from": old["generation_id"], "calibration": {"status": "pending"}})


def verify_spec(directory: Path, *, require_calibrated: bool = True) -> dict:
    """Refuse a generation that is not whole, not built for this code, or (unless ``require_calibrated`` is
    False, calibrate's own load) not calibrated for its weights. Returns the manifest."""
    directory = Path(directory)
    try:
        verify_manifest(directory)
        manifest = read_manifest(directory)
    except ManifestError as error:
        raise GenerationError(f"{directory}: {error}") from error
    if manifest.get("kind") != KIND or manifest.get("schema_version") != SCHEMA_VERSION:
        raise GenerationError(f"{directory}: not a {KIND} generation of schema {SCHEMA_VERSION}")
    if manifest.get("architecture") != ARCHITECTURE:
        raise GenerationError(f"{directory}: architecture {manifest.get('architecture')} is not the code's "
                              f"{ARCHITECTURE} (text spec, heads or decision rule changed); refusing")
    model = manifest["model"]
    missing = [name for name in WEIGHT_FILES if name not in manifest["files"]]
    if missing:
        raise GenerationError(f"{directory}: weight file(s) {missing} are not in the manifest")
    if model["weights_sha256"] != weights_digest(manifest["files"]):
        raise GenerationError(f"{directory}: model.weights_sha256 does not match the weight files")
    if model["n_terms"] != len(read_terms(directory)):
        raise GenerationError(f"{directory}: model.n_terms {model['n_terms']} != len(terms.json)")
    calibration = manifest.get("calibration") or {}
    if require_calibrated:
        if calibration.get("status") != "calibrated":
            raise GenerationError(f"{directory}: calibration.status is {calibration.get('status')!r}, not "
                                  "'calibrated'; run calibrate.py first")
        if calibration.get("weights_sha256") != model["weights_sha256"]:
            raise GenerationError(f"{directory}: the thresholds were calibrated for other weights; refusing")
    return manifest
