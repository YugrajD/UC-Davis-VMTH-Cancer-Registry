"""Thin entry point: sync mutable file sets with S3 (dry run by default; ``--apply`` executes).

All the work lives in ``s3sync``. Sets are listed in ``config.S3_SYNC_SETS``; SET defaults to ``all``.

  ml/.venv/bin/python ml/scripts/sync.py status [SET]
  ml/.venv/bin/python ml/scripts/sync.py push [SET] [--apply]
  ml/.venv/bin/python ml/scripts/sync.py pull [SET] [--apply]
  ml/.venv/bin/python ml/scripts/sync.py publish-model [--apply]   # config.REPORT_MAPPING_CURRENT_DIR only
  ml/.venv/bin/python ml/scripts/sync.py pull-model [--apply]      # fetch, verify, then promote.adopt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from botocore.exceptions import BotoCoreError, ClientError

from s3sync import models, sets
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
    for name in ("publish-model", "pull-model"):
        commands.add_parser(name).add_argument("--apply", action="store_true",
                                               help="Execute instead of printing the plan.")
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
    for label in ("added", "removed"):
        for rel in change[label]:
            print(f"  {label}: {rel}")


def _print_pull(result: dict) -> None:
    verb = "pulled" if result["applied"] else "would pull"
    print(f"{result['set']}: {verb} {len(result['fetch'])} files, remove {len(result['remove'])} "
          f"(deleted remotely), {len(result['conflicts'])} conflicts (remote wins, local copy backed up), "
          f"{len(result['kept_modified'])} locally modified files kept, "
          f"{len(result['skipped_excluded'])} excluded skipped, {len(result['backed_up'])} backed up")
    for label in ("conflicts", "kept_modified", "remove"):
        for rel in result[label]:
            print(f"  {label}: {rel}")


def _print_model_status(result: dict) -> None:
    print(f"model: remote {result['remote_generation'] or 'none'}, local current {result['local_generation'] or 'none'}"
          f"{' (MOVED since last sync — pull-model first)' if result['remote_moved'] else ''}"
          f"{' (never synced)' if result['never_synced'] else ''}")


def _print_publish_model(result: dict) -> None:
    if result["nothing_to_do"]:
        print(f"model: {result['generation_id']} is already published")
        return
    print(f"model: {'published' if result['applied'] else 'would publish'} {result['generation_id']} "
          f"({result['files']} files, {result['bytes']} bytes"
          f"{', files already uploaded' if result['already_uploaded'] else ''}) and move CURRENT to it")


def _print_pull_model(result: dict) -> None:
    files = f" ({result['files']} files)" if result["files"] else ""
    print(f"model: remote {result['remote_generation'] or 'none'}, local current "
          f"{result['local_generation'] or 'none'}: {'DONE' if result['applied'] else 'plan'}: {result['action']}{files}")
    if result["archive"]:
        print(f"  current is archived to {result['archive']}")
    if result["warning"]:
        print(f"  WARNING: {result['warning']}")


def main() -> int:
    args = build_parser().parse_args()
    apply = getattr(args, "apply", False)
    try:
        remote = Remote(make_client(), args.prefix)
        if args.command == "publish-model":
            _print_publish_model(models.publish(remote, apply))
        elif args.command == "pull-model":
            _print_pull_model(models.pull_model(remote, apply))
        else:
            for name in sets.set_names(args.set):
                if args.command == "status":
                    _print_status(sets.status(remote, name))
                elif args.command == "push":
                    _print_push(sets.push(remote, name, apply))
                else:
                    _print_pull(sets.pull(remote, name, apply))
            if args.command == "status" and args.set == "all":
                _print_model_status(models.status(remote))
    except (S3SyncError, BotoCoreError, ClientError) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 1
    if args.command != "status" and not apply:
        print("DRY RUN — pass --apply")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
