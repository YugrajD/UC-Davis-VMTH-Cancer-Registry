"""A frozen bronze_rr generation: ``config.RR_PHASE1_DIR/generation/<generation_id>/``.

  retriever/           encoder/, head.pt, terms.json (copied from the source retriever dir), train_log.json
  gate/                encoder/, head.pt, train_log.json (new gate)   OR   gate_source.json (old gate: a reference
                       to the report_mapping predictions CSV whose case_presence_prob is the gate)   OR   head.pt,
                       source.json, train_log.json (shared gate: a head on the retriever's pooled vector; the
                       encoder is the retriever's, so one pass gives both the gate and the term probabilities)
                       OR   gate_source.json pointing at retriever/gate_head.pt (joint model: the retriever/ dir is the
                       whole jointly trained model, one encoder and both heads)
  thresholds.json      {gate_t, tau, rho}
  uncommon_groups.txt  the list the scoring ruler uses
  manifest.json        generation_id, sources, seeds, split/labels ids, calibration block, sha256 per file

Not a report_mapping generation: it has no embedding fingerprint and cannot be loaded by ml-worker.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from bronze_rr.decide import Thresholds
from generations.manifest import read_manifest, sha256_file, verify_manifest, write_manifest

COPIED_FILES = ("head.pt", "terms.json", "train_log.json", "gate_head.pt")
SHARED_GATE_FILES = ("head.pt", "source.json", "train_log.json")


def new_generation_id() -> str:
    return "rr-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _copy_model(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True)
    shutil.copytree(source / "encoder", destination / "encoder")
    for name in COPIED_FILES:
        if (source / name).is_file():
            shutil.copy2(source / name, destination / name)


def _seed_of(model_dir: Path) -> str | None:
    log = model_dir / "train_log.json"
    return json.loads(log.read_text(encoding="utf-8"))["summary"]["args"].get("seed") if log.is_file() else None


def write_generation(root: Path, *, generation_id: str, retriever_dir: Path, gate_dir: Path | None,
                     old_gate_csv: Path | None, thresholds: Thresholds, calibration: dict, split_id: str,
                     silver_id: str, uncommon_groups_txt: Path, max_length: int, shared_gate_dir: Path | None = None,
                     joint_gate: bool = False) -> Path:
    """Write the generation directory and its manifest; returns the directory. Exactly one of ``gate_dir`` (new
    gate), ``old_gate_csv`` (old gate), ``shared_gate_dir`` (head on the retriever's encoder) and ``joint_gate``
    (``retriever_dir`` is a jointly trained model that holds the gate head, gate_head.pt) is given."""
    if sum(x is not None for x in (gate_dir, old_gate_csv, shared_gate_dir)) + joint_gate != 1:
        raise ValueError("give exactly one of gate_dir, old_gate_csv, shared_gate_dir and joint_gate")
    directory = root / generation_id
    directory.mkdir(parents=True)
    _copy_model(retriever_dir, directory / "retriever")
    if joint_gate:
        (directory / "gate").mkdir()
        source = {"kind": "joint_head", "path": "retriever/gate_head.pt",
                  "sha256": sha256_file(directory / "retriever" / "gate_head.pt")}
        (directory / "gate" / "gate_source.json").write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
        gate = {"kind": "joint_head", "source": str(retriever_dir), "seed": _seed_of(retriever_dir)}
    elif shared_gate_dir is not None:
        (directory / "gate").mkdir()
        for name in SHARED_GATE_FILES:
            shutil.copy2(shared_gate_dir / name, directory / "gate" / name)
        gate = {"kind": "shared_head", "source": str(shared_gate_dir), "seed": _seed_of(shared_gate_dir)}
    elif gate_dir is not None:
        _copy_model(gate_dir, directory / "gate")
        gate = {"kind": "modernbert", "source": str(gate_dir), "seed": _seed_of(gate_dir)}
    else:
        (directory / "gate").mkdir()
        source = {"kind": "wp14_predictions_csv", "path": str(old_gate_csv), "sha256": sha256_file(old_gate_csv)}
        (directory / "gate" / "gate_source.json").write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
        gate = {"kind": "wp14_predictions_csv", "source": str(old_gate_csv)}
    (directory / "thresholds.json").write_text(json.dumps(thresholds.as_dict(), indent=2) + "\n", encoding="utf-8")
    shutil.copy2(uncommon_groups_txt, directory / "uncommon_groups.txt")
    write_manifest(directory, {
        "kind": "bronze_rr_generation", "generation_id": generation_id,
        "retriever": {"source": str(retriever_dir), "seed": _seed_of(retriever_dir)}, "gate": gate,
        "parents": {"split_id": split_id, "silver_id": silver_id}, "max_length": max_length,
        "calibration": {**calibration, "values": thresholds.as_dict(),
                        "calibrated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")},
    })
    return directory


def load_generation_info(directory: Path) -> tuple[dict, Thresholds]:
    """(manifest, thresholds) of a generation whose files all match the manifest's sha256s."""
    verify_manifest(directory)
    values = json.loads((directory / "thresholds.json").read_text(encoding="utf-8"))
    return read_manifest(directory), Thresholds(**values)
