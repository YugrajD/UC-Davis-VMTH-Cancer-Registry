"""Report-mapping generation management.

  ml/.venv/bin/python ml/next/scripts/generations.py import-gen0
  ml/.venv/bin/python ml/next/scripts/generations.py import-cache [--generation current|candidate|<dir>]

``import-gen0`` and ``import-cache`` are WP4's contribution (one-time legacy
imports for gen-0). WP10 adds ``status`` / ``promote`` subcommands here later.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from report_mapping.inference.embedding_cache import import_legacy_cache
from report_mapping.model.generation import import_legacy_gen0, resolve_generation_dir


def _import_gen0(_args: argparse.Namespace) -> None:
    manifest = import_legacy_gen0()
    print(f"Imported gen-0: generation_id={manifest['generation_id']!r}")
    print(f"  files: {len(manifest['files'])}")
    print(f"  embedding_fingerprint: {manifest['embedding_fingerprint']}")


def _import_cache(args: argparse.Namespace) -> None:
    key = import_legacy_cache(resolve_generation_dir(args.generation))
    print(f"Imported legacy embedding cache -> key {key}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen0 = sub.add_parser("import-gen0", help="Copy the legacy checkpoints into config.REPORT_MAPPING_CURRENT_DIR.")
    p_gen0.set_defaults(func=_import_gen0)

    p_cache = sub.add_parser("import-cache", help="Copy the legacy embedding cache under the new content-hash key.")
    p_cache.add_argument("--generation", default="current")
    p_cache.set_defaults(func=_import_cache)

    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
