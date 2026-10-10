"""The serving image's file set is enough, and nothing else is imported: lay out exactly the files
ml-worker/Dockerfile.batch COPYs (``/ml`` and ``/worker`` as siblings, like the image) and run the worker's entry
point there with the training-side packages blocked, as they are absent from the image."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import io_utils

_REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ml-worker").is_dir())
_WORKER = _REPO_ROOT / "ml-worker"

# scipy/sklearn: transformers imports them when installed; the image has neither. The rest is ml/ code the
# image does not carry.
_BLOCKED = ("scipy", "sklearn", "diagnosis_mapping", "evaluation", "generations.guards", "generations.promote",
            "generations.splits", "generations.triggers", "taxonomy", "coding", "s3sync", "manual_audit")

# Run in the laid-out image: block the packages, load the entry point as a module, then (optionally) run main().
_RUN = """
import importlib.util, sys
for name in {blocked!r}:
    sys.modules[name] = None
spec = importlib.util.spec_from_file_location('entry', {entry!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
if {run_main!r}:
    module.main()
loaded = sorted(m for m, v in sys.modules.items() if v is not None and (m in {blocked!r} or m.split('.')[0] in {blocked!r}))
import config
print('CONFIG', config.__file__)
sys.exit(1 if loaded else 0)
"""


def _lay_out_image(dockerfile: str, image_root: Path) -> None:
    """Copy what the Dockerfile's COPY lines name into ``image_root`` (its /ml and /worker)."""
    for line in (_WORKER / dockerfile).read_text(encoding="utf-8").splitlines():
        if not line.startswith("COPY ") or "requirements.txt" in line:
            continue
        *sources, dest = line.split()[1:]
        target = image_root / dest.lstrip("/") if dest.startswith("/") else image_root / "worker" / dest
        for source in sources:
            if dest.endswith("/") or dest == ".":
                target.mkdir(parents=True, exist_ok=True)
                shutil.copy(_REPO_ROOT / source, target / Path(source).name)
            else:
                shutil.copytree(_REPO_ROOT / source, target, ignore=shutil.ignore_patterns("__pycache__"))


def _run_in_image(image: Path, entry: Path, run_main: bool, env: dict) -> subprocess.CompletedProcess:
    code = _RUN.format(blocked=_BLOCKED, entry=str(entry), run_main=run_main)
    return subprocess.run([sys.executable, "-c", code], cwd=image / "worker", capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": "", **env})


def test_dockerfile_batch_and_ecs_copy_the_same_ml_files():
    def ml_copies(name):
        return [line for line in (_WORKER / name).read_text(encoding="utf-8").splitlines() if re.match(r"COPY ml/", line)]
    assert ml_copies("Dockerfile.batch") == ml_copies("Dockerfile.ecs") != []


def test_batch_predict_runs_end_to_end_on_the_image_file_set(tmp_path, tiny_generation_dir):
    image = tmp_path / "image"
    _lay_out_image("Dockerfile.batch", image)
    assert not (image / "ml" / "taxonomy").exists() and not (image / "ml" / "diagnosis_mapping").exists()
    bundle = shutil.copytree(tiny_generation_dir, tmp_path / "models" / "production")
    io_utils.write_csv(pd.DataFrame({"anon_id": ["A", "B"], "Text": ["mast cell tumor present", ""]}), tmp_path / "in.csv")
    env = {"JOB_ID": "1", "INPUT_CSV_PATH": str(tmp_path / "in.csv"), "OUTPUT_DIR": str(tmp_path / "out"),
           "MODEL_PATH": str(bundle / "petbert"), "TORCH_NUM_THREADS": "1"}

    result = _run_in_image(image, image / "worker" / "batch_predict.py", True, env)

    assert result.returncode == 0, result.stderr[-2000:]
    config_line = next(line for line in result.stdout.splitlines() if line.startswith("CONFIG "))
    assert Path(config_line.removeprefix("CONFIG ")).resolve() == (image / "ml" / "config.py").resolve()
    predictions = json.loads((tmp_path / "out" / "predictions.json").read_text(encoding="utf-8"))
    assert [p["anon_id"] for p in predictions] == ["A"]


def test_app_imports_nothing_outside_the_image_file_set(tmp_path, tiny_generation_dir):
    if importlib.util.find_spec("fastapi") is None:
        pytest.skip("fastapi is a worker dependency, not installed in the ml venv")
    image = tmp_path / "image"
    _lay_out_image("Dockerfile.batch", image)
    shutil.copy(_WORKER / "app.py", image / "worker" / "app.py")
    env = {"PETBERT_MODEL_PATH": str(tiny_generation_dir / "petbert"), "TORCH_NUM_THREADS": "1",
           # the image has /ml; here it is a sibling of the worker dir
           "PYTHONPATH": os.pathsep.join([str(image / "ml"), os.environ.get("PYTHONPATH", "")])}
    assert _run_in_image(image, image / "worker" / "app.py", False, env).returncode == 0
