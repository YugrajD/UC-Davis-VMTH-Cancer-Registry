# 0006 Move private data between machines with an S3 sync, not Syncthing or Box/rclone

**Status:** accepted · **Date:** 2026-09-28

## Context
Data, stores and model generations hold patient records and cannot go through git. The first mechanism was Syncthing with a shared allowlist (`.stignore`, added 2026-08-21, widened to all of `ml/data` and `ml/output` by 2026-09-15). A Box + rclone proposal (per-person pipeline directories, pull-only shared data and bisync for outputs) was written earlier and never implemented. In September a project S3 bucket became available. The repository does not record why S3 was preferred over Syncthing beyond the hazards below.

## Decision
Use a guarded boto3 sync (`ml/s3sync/`, entry point `ml/scripts/sync.py`).
- Content-addressed blobs and immutable manifests; each file set has one mutable pointer (`HEAD.json`) moved only by a conditional write (`If-Match`), so sync is fast-forward only. Remote wins conflicts, and the local copy is backed up first.
- Model generations are published as immutable directories with one mutable `CURRENT.json` pointer (`publish-model`, `pull-model`).
- Nothing relies on S3 versioning: the project's access was probed on 2026-09-28 and could read, write, list and do conditional writes, but not list or read old object versions.
- Every command is a dry run until `--apply`. Opt-in hooks: `promote.py --publish`, `handoff.py --push`.
- Syncthing was disabled on every machine, then `.stignore` and its `.gitignore` block were removed (commit `0c06b0a`).

## Consequences
- Do not re-enable Syncthing on this repo: without `.stignore` it would sync `.git/` and the virtual environment, and it would also sync each machine's `s3sync_state.json` and `s3sync_backup/`, corrupting the three-way base. Leftover `.stfolder/`, `.stversions/` and `*.sync-conflict-*` files can be deleted.
- Rollback is a manual repoint of a pointer; there is no rollback command yet. One `put_object` per file means a single file over 5 GB cannot be uploaded.
- The bucket's `database/` and `ml/` prefixes belong to other people; the key builder refuses them and everything outside `ml-Revised-ICD-Mapping/`.
- Root `CLAUDE.md` is gitignored and is copied between machines by hand. The embedding cache, archive and `candidate/` are not synced.
- The Box/rclone proposal ([kept for history](../plans/box-rclone-sync-proposal.md)) was never built. Its useful idea of one config-derived path root was not needed once one tree and one sync tool existed.

## Evidence
No experiment; an operations decision. Commits `582e3a7` (core), `219d74b` (model generations), `eb8c33f` (hooks), `b75fff9` (docs, Box proposal archived), `0c06b0a` (Syncthing retired). How to use it: [../how-to/sync-with-s3.md](../../how-to/sync-with-s3.md).
