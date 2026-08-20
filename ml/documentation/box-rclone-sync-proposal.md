# Proposal: Box + rclone sync for per-person ML pipelines

**Status:** Planning / for team review — nothing implemented.

## Goal

Let teammates run each other's models and compare results, without committing
private patient data to git and without clobbering divergent NLP architectures.

## Repo structure (monorepo, same branch — extends existing `ml-xxx/` pattern)

- `ml/` — production default (promoted *into*, not worked in directly)
- `ml-chunho/`, `ml-yugi/` — each person's pipeline; code shared via git

## Box layout (UC Davis SSO — the client's existing upload space)

```
VMTH-Cancer-Registry/
├── data/              ← single shared raw input (client uploads)
├── shared/gold/       ← single shared eval ruler
├── ml/output/         ← production weights
├── ml-chunho/output/  ← Chun Ho writes / Yugi reads  ┐ bisync
└── ml-yugi/output/    ← Yugi writes / Chun Ho reads   ┘
```

## Sync rules

| Path | Mode | Why |
|------|------|-----|
| `data/`, `shared/gold/` | **pull-only** (`rclone copy` down) | client/canonical owns it — bisync could propagate a delete *up* and destroy it |
| `*/output/` | **bisync** | write-ownership partitioned per person → no real conflicts |
| embedding caches (`*.npz`) | **not synced** | backbone-specific, auto-rebuild; syncing causes silent rebuilds |
| code | **not synced** (git only) | rclone + git fighting over the same files |

## Prerequisite refactor (the part that makes it safe)

Each `config.py` derives every path from one `PKG` constant:

```python
PKG = "ml-yugi"   # only line that changes per dir
REPORTS_CSV = f"{PKG}/data/report.csv"  # → repoint to shared root data/
```

This kills the existing footgun where `ml-experiment/config.py` half-points at
`ml/` (e.g. `CHECKPOINT_GROUP_DIR = "ml/output/checkpoints/group"` while
`REPORTS_CSV = "ml-experiment/data/report.csv"`) and would silently load the
wrong checkpoints — this project's signature failure mode (stale checkpoint
loads silently, no error). Shared `data/` means a `ROOT_DATA` constant kept
separate from `PKG`.

## Why this works across different architectures

Code (git) + weights (Box) + same branch → `git pull` gets their code,
`rclone` pull gets their weights, same commit = guaranteed match. No
separate-repo commit-SHA pinning needed.

## Open questions for the team

1. **Environment reproducibility** — if stacks have different deps (torch
   versions, GPUs), does each `ml-xxx/` ship its own `requirements.txt` /
   lockfile, or share one venv? (The RTX 5070 Ti needs torch 2.6 / CU12.8;
   a teammate on different hardware won't match.)
2. **Shared I/O contract** — agree on a common input→output CSV shape so
   results in `shared/` are comparable / ensemble-able across architectures.
3. **Who promotes to `ml/`** — production is the one shared collision point.
   Promotion via PR/review, not direct edits?
4. **Git coordination** — same branch, two committers: agree on
   pull-before-push / rebase discipline (file conflicts rare since dirs are
   namespaced, but history intertwines).
5. **rclone bisync hygiene** — needs `--resync` on first run and a
   conflict-handling policy; worth a dry-run before trusting it.

## Risks

- **Privacy:** acceptable only because this is UC Davis SSO Box already
  approved for this data class. Personal Box would not be.
- **Silent cross-dir reads** if the `config.py` prefix refactor is skipped.
- **bisync data loss** if `data/` is ever bisynced instead of pull-only.
- **Cache confusion** if `*.npz` embedding caches are synced across backbones.
