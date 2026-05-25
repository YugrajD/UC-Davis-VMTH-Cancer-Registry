"""Name-safety transforms shared across modules.

Two distinct transforms are needed:

  ``safe_filename``  — used for filenames (per-group LabelPresenceClassifier
                       .pt files, training pair CSVs). Aggressive: lowercases
                       and collapses any non-alphanumeric run to a single
                       underscore.

  ``npz_col_key``    — used for npz keys for the column-name suffixed arrays
                       in the embedding cache (``col_<key>``, ``has_<key>``).
                       Mild: only the three characters that np.savez rejects
                       in keys are replaced. Preserves case so existing caches
                       remain readable.

These are intentionally *not* the same function — see embedding_cache.py for
why the column-key transform must stay narrow.
"""

from __future__ import annotations

import re


def safe_filename(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def npz_col_key(col: str) -> str:
    return col.replace(" ", "_").replace(",", "").replace("/", "_")
