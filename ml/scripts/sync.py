"""Thin entry point: sync mutable file sets with S3 (dry run by default; ``--apply`` executes).

All the work lives in ``s3sync``. Sets are listed in ``config.S3_SYNC_SETS``; SET defaults to ``all``.

  ml/.venv/bin/python ml/scripts/sync.py status [SET]
  ml/.venv/bin/python ml/scripts/sync.py push [SET] [--apply]
  ml/.venv/bin/python ml/scripts/sync.py pull [SET] [--apply]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from s3sync import sets
from s3sync.client import make_client
from s3sync.remote import Remote, S3SyncError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prefix", default=None,
                        help="Narrow to a sub-prefix of config.S3_PREFIX (e.g. a scratch run); validated by the guard.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("status", "push", "pull"):
        command = commands.add_parser(name)
        command.add_argument("set", nargs="?", default="all")
        if name != "status":
            command.add_argument("--apply", action="store_true", help="Execute instead of printing the plan.")
    return parser


def _print_status(result: dict) -> None:
    local = result["local"]
    print(f"{result['set']}: remote {result['remote_manifest'] or 'empty'}"
          f"{' (MOVED since last sync — pull first)' if result['remote_moved'] else ''}"
          f"{' (never synced)' if result['never_synced'] else ''}; local vs last sync: "
          f"{len(local['added'])} added, {len(local['changed'])} changed, {len(local['removed'])} removed")


def _print_push(result: dict) -> None:
    change = result["change"]
    if result["nothing_to_do"]:
        print(f"{result['set']}: nothing to push")
        return
    print(f"{result['set']}: {'pushed' if result['applied'] else 'would push'} "
          f"{len(change['added'])} added, {len(change['changed'])} changed, {len(change['removed'])} removed")


def _print_pull(result: dict) -> None:
    verb = "pulled" if result["applied"] else "would pull"
    print(f"{result['set']}: {verb} {len(result['fetch'])} files, remove {len(result['remove'])} "
          f"(deleted remotely), {len(result['conflicts'])} conflicts (remote wins, local copy backed up), "
          f"{len(result['kept_modified'])} locally modified files kept, "
          f"{len(result['skipped_excluded'])} excluded skipped, {len(result['backed_up'])} backed up")
    for label in ("conflicts", "kept_modified", "remove"):
        for rel in result[label]:
            print(f"  {label}: {rel}")


def main() -> int:
    args = build_parser().parse_args()
    apply = getattr(args, "apply", False)
    try:
        remote = Remote(make_client(), args.prefix)
        for name in sets.set_names(args.set):
            if args.command == "status":
                _print_status(sets.status(remote, name))
            elif args.command == "push":
                _print_push(sets.push(remote, name, apply))
            else:
                _print_pull(sets.pull(remote, name, apply))
    except S3SyncError as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 1
    if args.command != "status" and not apply:
        print("DRY RUN — pass --apply")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
