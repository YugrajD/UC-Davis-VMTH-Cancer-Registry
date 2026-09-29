"""The opt-in S3 hooks behind ``promote.py --publish`` and ``handoff.py --push``.

Each returns an exit code: summary lines go to stdout, problems to stderr. The local work has already
succeeded by the time a hook runs, so every failure message says so and names the command that finishes
the job. The entry points import this module only when their flag is given.
"""

from __future__ import annotations

import sys

from botocore.exceptions import BotoCoreError, ClientError

from s3sync import client, models, sets
from s3sync.remote import ConflictError, Remote, S3SyncError

_ERRORS = (S3SyncError, BotoCoreError, ClientError)


def push_sets(command: str, set_names) -> int:
    """Push ``set_names`` in order after ``command`` succeeded locally; stop at the first failure."""
    set_names, pushed, remote = list(set_names), [], None
    for i, set_name in enumerate(set_names):
        try:
            remote = remote or Remote(client.make_client())
            change = sets.push(remote, set_name, apply=True)["change"]
        except _ERRORS as error:
            if isinstance(error, ConflictError):
                hint = (f"run ml/scripts/sync.py pull {set_name} --apply, "
                        f"then ml/scripts/sync.py push {set_name} --apply")
            else:
                hint = f"re-run ml/scripts/sync.py push {set_name} --apply"
            rest = set_names[i + 1:]
            print(f"{command} done locally; pushed: {', '.join(pushed) or '-'}; FAILED: {set_name} ({error}); "
                  f"not attempted: {', '.join(rest) or '-'} (stopped at the first failure); {hint}"
                  f"{' (likewise for the not-attempted sets)' if rest else ''}", file=sys.stderr)
            return 1
        pushed.append(set_name)
        print(f"pushed {set_name}: {len(change['added'])} added, {len(change['changed'])} changed, "
              f"{len(change['removed'])} removed")
    return 0


def publish_current() -> int:
    """Publish ``current/`` after a local promotion."""
    try:
        result = models.publish(Remote(client.make_client()), apply=True)
    except ConflictError as error:
        # Fast-forward only: this promotion was scored against a current/ that is no longer the shared one.
        print(f"PROMOTED locally, but the remote model moved ({error}); another machine published a newer "
              "generation. Run ml/scripts/sync.py pull-model --apply (archives this promotion locally), then "
              "re-run promotion against the new current/ if this candidate should still win", file=sys.stderr)
        return 1
    except _ERRORS as error:
        print(f"PROMOTED locally, but publish failed: {error}; run ml/scripts/sync.py publish-model --apply",
              file=sys.stderr)
        return 1
    print(f"Published {result['generation_id']} to S3 ({result['files']} files, {result['bytes']} bytes)")
    return 0
