# 0001 Represent each report as three per-section embeddings, concatenated (concat-3)

**Status:** accepted · **Date:** 2026-05-12 (promoted 2026-05-13)

## Context
Reports have structured sections. The earlier input merged several columns into one string and, when it overflowed 512 tokens, cut it with TF-IDF sentence scoring. In 26.7% of cases the merged text overflowed; the merge also flattened section structure so no head could weight one section over another.

## Decision
Every report is split into three sections (histopathological summary; final comment plus comment; ancillary tests). Each section is embedded on its own with the full token budget and mean-pooled to 768 dims; the three vectors are concatenated into one 2304-dim case vector. The section specification is defined once (`ml/report_mapping/sections.py`, `SECTION_SPEC_VERSION = 1`) and shared by training and inference.

## Consequences
- Any change to the section definition changes the embeddings, invalidates every downstream head, and must bump `SECTION_SPEC_VERSION`. The generation's embedding fingerprint (backbone hash, section spec version, max length) records this and loading refuses a mismatch ([0004](0004-generations-candidate-current-archive.md)).
- The three heads are sized for 2304 dims (the per-group LabelPresence head uses three section pairs and a learned combiner).
- A small share of cases (about 0.25%) have all three sections empty and get a near-zero vector.
- The old merged-text path, TF-IDF selector and cache key `tfidf_selected` no longer exist.

## Evidence
[concat-3](../experiments/2026-05-12-concat-3-text-representation.md), [TF-IDF selection it replaced](../experiments/2026-04-29-tfidf-text-selection.md), [per-column embeddings that led to it](../experiments/2026-03-05-per-column-embeddings-and-wider-hidden-layer.md). Current description: [../../concepts/report-mapping.md](../../concepts/report-mapping.md).
