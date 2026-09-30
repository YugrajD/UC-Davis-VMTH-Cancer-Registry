# 0004 Keep every trained model in a generation directory (candidate, current, archive) with a fingerprint check

**Status:** accepted · **Date:** 2026-09-25 (rewrite plan approved; earlier rule in `CLAUDE.md` versioning section)

## Context
Embeddings change when the text fed to PetBERT changes or the PetBERT weights change. Any classifier trained on the old embeddings is then silently wrong. Before the rewrite this was guarded only by a written rule (archive before retrain) and an embedding cache keyed on the CSV's modified time, which an incident in August 2026 showed to be fragile. Stale checkpoints loading without error was the project's signature failure.

## Decision
- A **generation** is one directory holding the backbone, the label table, all head checkpoints, thresholds and a `manifest.json` (sha256 per file, parents, recipe, seed, embedding fingerprint, calibration status).
- Training writes only to `candidate/`. `current/` is production. `promote.py --apply` archives the old generation whole (with the cache entries the new one cannot use) under `ml/output/archive/YYYY-MM-DD_<description>/` and swaps the candidate in; a losing candidate is deleted. The archive is written only by the generations code and never loaded from.
- Loading verifies the manifest, then the fingerprint (backbone hash, section spec version, max length), and refuses a generation that is not calibrated.
- The embedding cache is keyed by a sha256 of report bytes, label bytes and the fingerprint, and lives outside any generation.
- Promotion is a rule, not a judgment: guards pass, then a paired stratified case-cluster bootstrap on gold-eval must show the lower bound of the good-share difference at or above -2 pp, and at least one retraining trigger (new silver lineage, random-slice drop, or 200 or more new gold-train codes) must be met.

## Consequences
- A mismatched or partial generation fails loudly instead of predicting quietly.
- Promotion cannot run for real until gold exists (none does yet), so production remains `gen-0-legacy`.
- There is no automatic restore step: the incumbent is archived only when the candidate wins.

## Evidence
No experiment measures this; it is an infrastructure decision made in the [rewrite](0005-ml-rewrite-and-cutover.md). Current description: [../../concepts/generations.md](../../concepts/generations.md).
