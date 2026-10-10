> **Archived** — historical record, not maintained. Paths, flags and numbers may not match today's code. Current docs: [README](../../README.md).

# Archive — legacy tree docs

Seven files document trees that no longer exist:

- Five describe the pre-rewrite `ml/` tree (`production/petbert_pipeline/`, `training/`,
  `annotation/llm_pipeline/`, `evaluation/evaluate.py`, ...), deleted in the WP13 cutover. See
  [ml-rewrite-plan.md](../plans/ml-rewrite-plan.md) for the rewrite: `classifiers.md`,
  `label-annotation.md`, `model-training.md`, `production-pipeline.md`, `training-guide.md`.
- Two describe the PetBERT four-stage report-mapping pipeline of the rewritten tree (case-presence
  gate, group classifier, per-group label-presence heads, keyword correction), retired by the
  2026-10-09 restructure ([ADR 0008](../decisions/0008-joint-modernbert-report-mapping.md),
  [ADR 0009](../decisions/0009-bronze-restructure-and-lifecycle.md)):
  [four-stage-pipeline.md](four-stage-pipeline.md) (the old `concepts/report-mapping.md`) and
  [petbert.md](petbert.md) (the old `reference/petbert.md`).

The current design is in [report-mapping.md](../../concepts/report-mapping.md).
