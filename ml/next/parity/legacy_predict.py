"""Regenerate the legacy predictions with the OLD production code, read-only.

LEGACY COMPARISON — deleted at cutover.

Run as a subprocess by ``reference.freeze`` (never imported by ml/next: the old
tree's ``config`` / ``evaluation`` modules would collide with ours):

    python legacy_predict.py <old run_production.py> <out_dir> <device>

It runs the old ``run_production.main`` with its production defaults and only
``--out-dir`` / ``--device`` overridden, with three in-process patches and no
edits to old code:

- The embedding cache is loaded by the old ``load_cache`` with only the
  labels.csv mtime check bypassed (its content is unchanged since 2026-03 but
  its mtime moved). Any other cache miss raises instead of re-embedding.
- ``save_cache`` raises, so the legacy cache can never be written.
- Every output writer except predictions and summary is a no-op, so no
  report text (provenance, similarity, embeddings npz) is written anywhere.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import numpy as np


def main(run_production_py: str, out_dir: str, device: str) -> None:
    ml_root = Path(run_production_py).resolve().parents[1]
    sys.path.insert(0, str(ml_root))
    from production.petbert_pipeline import embedding_cache, pipeline

    original_load = embedding_cache.load_cache

    def load_cache_ignoring_labels_mtime(path, *, labels_csv_path, **kwargs):
        cached_mtime = float(np.load(path, allow_pickle=True)["labels_mtime"][0])
        target = os.path.abspath(labels_csv_path)
        real_getmtime = os.path.getmtime
        os.path.getmtime = lambda p: cached_mtime if os.path.abspath(p) == target else real_getmtime(p)
        try:
            data = original_load(path, labels_csv_path=labels_csv_path, **kwargs)
        finally:
            os.path.getmtime = real_getmtime
        if data is None:
            raise RuntimeError("legacy embedding cache rejected; refusing to re-embed")
        return data

    def refuse_save(*args, **kwargs):
        raise RuntimeError("legacy embedding cache must not be written")

    embedding_cache.load_cache = load_cache_ignoring_labels_mtime
    embedding_cache.save_cache = refuse_save
    for writer in ("write_provenance_csv", "write_similarity_csv", "write_visualization_csv",
                   "write_neighbors_csv", "write_embeddings_npz"):
        setattr(pipeline, writer, lambda *args, **kwargs: None)

    legacy = runpy.run_path(run_production_py, run_name="legacy_run_production")
    sys.argv = [run_production_py, "--out-dir", out_dir, "--device", device]
    legacy["main"]()


if __name__ == "__main__":
    main(*sys.argv[1:])
