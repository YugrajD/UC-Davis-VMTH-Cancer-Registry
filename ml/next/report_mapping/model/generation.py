"""Load/save a report-mapping generation directory (``current/``, ``candidate/``).

This layout *is* the cloud bundle layout ml-worker downloads (ml-rewrite-plan.md,
Artefacts: "Report-mapping generation")::

    petbert/                                  HF backbone checkpoint dir
    labels/labels.csv                         Vet-ICD-O taxonomy snapshot
    checkpoints/case_presence_classifier.pt
    checkpoints/group_classifier_best.pt
    checkpoints/label_presence/<safe_group>.pt
    checkpoints/label_presence/lp_thresholds.json
    checkpoints/thresholds.json               gate / group / tail / LP-fallback
    checkpoints/uncommon_groups.txt
    manifest.json

``load_generation`` verifies the directory's manifest (file hashes) and its
embedding fingerprint (backbone sha256, section-spec version, inference
max_length) before loading anything — the code-level fix for "stale
classifiers load silently and produce wrong results" (CLAUDE.md, "Embedding &
Classifier Versioning"). A mismatch raises ``GenerationError`` rather than
returning a usable-looking but wrong generation.

Also owns ``safe_filename`` (per-group ``.pt`` / training-pair filenames) and
``npz_col_key`` (embedding-cache npz key), moved here from the legacy
``ml/utils/encoding.py`` per the plan's Old -> new table.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import config
from generations import manifest as manifest_mod
from report_mapping import sections
from report_mapping.model import backbone as backbone_mod
from report_mapping.model import heads
from taxonomy.taxonomy import TaxonomyLabel, load_labels_taxonomy

CHECKPOINTS_DIRNAME = "checkpoints"
LABEL_PRESENCE_DIRNAME = "label_presence"
LP_THRESHOLDS_NAME = "lp_thresholds.json"
THRESHOLDS_NAME = "thresholds.json"
UNCOMMON_GROUPS_NAME = "uncommon_groups.txt"
CASE_PRESENCE_NAME = "case_presence_classifier.pt"
GROUP_NAME = "group_classifier_best.pt"
PETBERT_DIRNAME = "petbert"
LABELS_DIRNAME = "labels"

# Written by import_legacy_gen0(); every other threshold key stages.py reads.
GEN0_THRESHOLDS = {
    "case_presence_gate": 0.80,
    "group": 0.85,
    "tail_max_predictions": 2,
    "tail_max_group_prob_gap": 0.08,
    "label_presence_fallback": 0.5,
}


class GenerationError(Exception):
    """A generation directory is missing, malformed, or its embedding fingerprint
    doesn't match the manifest — refuse rather than load silently-wrong classifiers."""


def safe_filename(name: str) -> str:
    """Aggressive filename-safety transform: per-group LP checkpoint / training-pair names."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def npz_col_key(col: str) -> str:
    """Mild transform for embedding-cache npz array keys: only characters np.savez rejects."""
    return col.replace(" ", "_").replace(",", "").replace("/", "_")


@dataclass(frozen=True)
class GenerationPaths:
    root: Path
    petbert_dir: Path
    labels_csv: Path
    case_presence_pt: Path
    group_pt: Path
    label_presence_dir: Path
    lp_thresholds_json: Path
    thresholds_json: Path
    uncommon_groups_txt: Path


def generation_paths(directory: str | Path) -> GenerationPaths:
    directory = Path(directory)
    checkpoints = directory / CHECKPOINTS_DIRNAME
    return GenerationPaths(
        root=directory,
        petbert_dir=directory / PETBERT_DIRNAME,
        labels_csv=directory / LABELS_DIRNAME / "labels.csv",
        case_presence_pt=checkpoints / CASE_PRESENCE_NAME,
        group_pt=checkpoints / GROUP_NAME,
        label_presence_dir=checkpoints / LABEL_PRESENCE_DIRNAME,
        lp_thresholds_json=checkpoints / LABEL_PRESENCE_DIRNAME / LP_THRESHOLDS_NAME,
        thresholds_json=checkpoints / THRESHOLDS_NAME,
        uncommon_groups_txt=checkpoints / UNCOMMON_GROUPS_NAME,
    )


def resolve_generation_dir(name: str) -> Path:
    """"current" / "candidate" -> config paths; anything else is used as a literal directory."""
    if name == "current":
        return config.REPORT_MAPPING_CURRENT_DIR
    if name == "candidate":
        return config.REPORT_MAPPING_CANDIDATE_DIR
    return Path(name)


def compute_embedding_fingerprint(petbert_dir: str | Path) -> dict:
    """{backbone_sha256, section_spec_version, max_length} — what a generation's
    classifiers were trained against. Compared against the *current* code + the
    generation's own bundled backbone at load time."""
    return {
        "backbone_sha256": backbone_mod.model_fingerprint(str(petbert_dir)),
        "section_spec_version": sections.SECTION_SPEC_VERSION,
        "max_length": backbone_mod.INFERENCE_MAX_LENGTH,
    }


def _read_lines(path: Path) -> frozenset[str]:
    if not path.exists():
        return frozenset()
    return frozenset(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class LoadedGeneration:
    generation_id: str
    root: Path
    petbert_dir: Path
    taxonomy_labels: list[TaxonomyLabel]
    case_presence: heads.CasePresenceClassifier
    group_head: heads.GroupClassifier
    group_names: list[str]
    label_presence_heads: dict[str, heads.LabelPresenceClassifier]
    lp_thresholds: dict[str, float]
    thresholds: dict
    uncommon_groups: frozenset[str]
    manifest: dict


def verify_fingerprint(directory: str | Path) -> dict:
    """Verify the manifest's embedding_fingerprint against the current code +
    the generation's own bundled backbone. Returns the manifest on success."""
    directory = Path(directory)
    manifest = manifest_mod.read_manifest(directory)
    recorded = manifest.get("embedding_fingerprint")
    if not recorded:
        raise GenerationError(f"{directory}: manifest has no embedding_fingerprint")
    current = compute_embedding_fingerprint(generation_paths(directory).petbert_dir)
    if current != recorded:
        raise GenerationError(
            f"{directory}: embedding fingerprint mismatch (manifest {recorded}, current {current}). "
            "Stale classifiers would silently load against differently-built embeddings; refusing."
        )
    return manifest


def _check_calibrated(directory: Path, manifest: dict, allow_uncalibrated: bool) -> None:
    """Refuse a generation whose ``calibration.status`` isn't "calibrated" —
    a freshly trained candidate's thresholds are only a placeholder copied
    from its parent (scripts/train.py), so predicting against it would score
    with thresholds fitted for a different set of embeddings/heads. Pass
    ``allow_uncalibrated=True`` for calibrate.py itself, which is the one
    caller that must load a pending candidate in order to calibrate it."""
    if allow_uncalibrated:
        return
    status = (manifest.get("calibration") or {}).get("status")
    if status != "calibrated":
        raise GenerationError(
            f"{directory}: calibration.status is {status!r} (expected 'calibrated'). "
            "This generation's thresholds are a placeholder pending calibration; predicting against "
            "it would silently score with thresholds fitted for a different generation. Run "
            "calibrate.py first, or pass allow_uncalibrated=True to load it anyway."
        )


def load_generation(generation_dir: str | Path | None = None, *, allow_uncalibrated: bool = False) -> LoadedGeneration:
    """Verify manifest + embedding fingerprint + calibration status, then load
    every checkpoint and threshold. Raises ``GenerationError`` on an
    uncalibrated (``calibration.status != "calibrated"``) generation unless
    ``allow_uncalibrated=True`` (calibrate.py's own load)."""
    directory = Path(generation_dir) if generation_dir is not None else config.REPORT_MAPPING_CURRENT_DIR
    manifest_mod.verify_manifest(directory)
    manifest = verify_fingerprint(directory)
    _check_calibrated(directory, manifest, allow_uncalibrated)
    paths = generation_paths(directory)

    taxonomy_labels = load_labels_taxonomy(str(paths.labels_csv))
    case_presence = heads.CasePresenceClassifier.load(paths.case_presence_pt)
    group_head, group_names = heads.GroupClassifier.load(paths.group_pt)

    lp_heads: dict[str, heads.LabelPresenceClassifier] = {}
    missing_lp = []
    for group_name in group_names:
        pt_path = paths.label_presence_dir / f"{safe_filename(group_name)}.pt"
        if pt_path.exists():
            model = heads.LabelPresenceClassifier.load(pt_path)
            model.eval()
            lp_heads[group_name] = model
        else:
            missing_lp.append(group_name)
    if missing_lp:
        print(f"Warning: no LP checkpoint for {len(missing_lp)} group(s); these produce no winner: {missing_lp}")

    return LoadedGeneration(
        generation_id=manifest.get("generation_id", directory.name),
        root=directory,
        petbert_dir=paths.petbert_dir,
        taxonomy_labels=taxonomy_labels,
        case_presence=case_presence,
        group_head=group_head,
        group_names=group_names,
        label_presence_heads=lp_heads,
        lp_thresholds=_read_json(paths.lp_thresholds_json),
        thresholds=_read_json(paths.thresholds_json),
        uncommon_groups=_read_lines(paths.uncommon_groups_txt),
        manifest=manifest,
    )


def import_legacy_gen0() -> dict:
    """Copy (never move) the legacy backbone + three head kinds + lp_thresholds +
    uncommon groups into ``config.REPORT_MAPPING_CURRENT_DIR``, write thresholds.json
    with the legacy production defaults, and backfill a manifest.

    Refuses if the target already has any content (write-once, like split
    generations). Ends by calling ``load_generation`` on the freshly-written
    directory — the "load the real legacy checkpoints" smoke check the plan asks
    for, exercised naturally by running this import.
    """
    directory = config.REPORT_MAPPING_CURRENT_DIR
    if directory.exists() and any(directory.iterdir()):
        raise GenerationError(f"{directory} already has content; refusing to overwrite gen-0")
    paths = generation_paths(directory)

    shutil.copytree(config.LEGACY_CHECKPOINT_CONTRASTIVE_DIR, paths.petbert_dir)
    paths.labels_csv.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config.LABELS_CSV, paths.labels_csv)
    paths.case_presence_pt.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config.LEGACY_CHECKPOINT_CASE_PRESENCE_PT, paths.case_presence_pt)
    shutil.copyfile(config.LEGACY_CHECKPOINT_GROUP_BEST_PT, paths.group_pt)
    shutil.copytree(config.LEGACY_CHECKPOINT_LABEL_PRESENCE_DIR, paths.label_presence_dir)
    shutil.copyfile(config.LEGACY_UNCOMMON_GROUPS_TXT, paths.uncommon_groups_txt)
    paths.thresholds_json.write_text(json.dumps(GEN0_THRESHOLDS, indent=2) + "\n", encoding="utf-8")

    fingerprint = compute_embedding_fingerprint(paths.petbert_dir)
    manifest = manifest_mod.write_manifest(directory, {
        "kind": "report_mapping_generation",
        "generation_id": "gen-0-legacy",
        "parents": {"silver_id": "silver-0-legacy", "split_id": "legacy-80-20", "gold_train_snapshot": None},
        "recipe": "legacy (see ml-rewrite-plan.md Findings)",
        "embedding_fingerprint": fingerprint,
        "calibration": {
            "status": "calibrated",
            "partition": "all of test",
            "objective": "gate (0.80) and tail gate (K=2, gap=0.08) fitted on all of test; "
                         "per-LP thresholds fitted on the md5 sweep half of test only",
            "values": GEN0_THRESHOLDS,
        },
        "status": "current",
    })
    load_generation(directory)  # smoke check: the real legacy checkpoints must load.
    return manifest
