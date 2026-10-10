"""ml-worker serves the model ml/ trains: every package pinned in both requirements files has the same pin."""

from __future__ import annotations

import config


def _pins(path) -> dict[str, str]:
    pins = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if "==" in line:
            name, version = line.split("==", 1)
            pins[name.strip().lower()] = version.strip()
    return pins


def test_shared_pins_match():
    ml = _pins(config.ML_ROOT / "requirements.txt")
    worker = _pins(config.REPO_ROOT / "ml-worker" / "requirements.txt")
    shared = ml.keys() & worker.keys()
    assert {"torch", "transformers", "tokenizers", "huggingface_hub", "safetensors"} <= shared
    assert {name: (ml[name], worker[name]) for name in shared if ml[name] != worker[name]} == {}
