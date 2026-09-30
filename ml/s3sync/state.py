"""Per-machine sync state (``config.S3_SYNC_STATE_JSON``): what this machine last pulled or pushed, per remote prefix.

Keyed by prefix so a scratch ``--prefix`` run never touches production state. Set names and the reserved
``MODELS`` key share the second level; set names never start with an underscore.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import config
from s3sync.files import PARTIAL_SUFFIX
from s3sync.remote import Remote

MODELS = "_generations"


def _load() -> dict:
    path = Path(config.S3_SYNC_STATE_JSON)
    return json.loads(path.read_text()) if path.exists() else {}


def synced(remote: Remote, name: str) -> dict | None:
    return _load().get(remote.prefix, {}).get(name)


def save(remote: Remote, name: str, entry: dict) -> None:
    state = _load()
    state.setdefault(remote.prefix, {})[name] = entry
    path = Path(config.S3_SYNC_STATE_JSON)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + PARTIAL_SUFFIX)
    temp.write_text(json.dumps(state, indent=1, sort_keys=True))
    os.replace(temp, path)
