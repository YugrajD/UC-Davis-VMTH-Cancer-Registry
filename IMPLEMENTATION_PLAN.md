# Implementation Plan: Closing Requirements Gaps

**Project:** UC Davis VMTH Cancer Registry
**Date:** 2026-02-08
**Reference:** `Requirements_Doc.md`

This document is the overview of the implementation plan. Each workstream and architectural section has its own detailed document in the `docs/` directory.

---

## Table of Contents

- [Gap Summary](#gap-summary)
- [Architecture](#architecture)
- [Workstreams](#workstreams)
- [Implementation Order](#implementation-order)
- [File Change Summary](#file-change-summary)

---

## Gap Summary

| # | Gap | Severity | User Stories |
|---|-----|----------|--------------|
| 1 | No CSV upload / data ingestion pipeline | Critical | US #2, #9 |
| 2 | No real BERT integration (mock keyword matcher only) | Critical | US #3 |
| 3 | No Vet-ICD-O-canine-1 coding system | Critical | US #1 |
| 4 | No authentication / access control | Critical | Security req |
| 5 | No trend line visualization in the frontend | Major | US #6 |
| 6 | No ambiguous diagnosis flagging / review workflow | Major | US #8 |
| 7 | Frontend tabs rendering fake data instead of real API data | Major | US #11 |
| 8 | NLP worker is inline, not async/separate | Moderate | Architecture |
| 9 | No tests | Moderate | Maintainability |

---

## Architecture

Detailed architecture diagrams, schemas, API surfaces, and component trees:

- **[Current Architecture](docs/current-architecture.md)** — Snapshot of the existing system (Docker Compose, DB schema, API endpoints, frontend component tree)
- **[Target Architecture](docs/target-architecture.md)** — Where we need to get to (Redis, Celery NLP worker, new routers, new DB tables, expanded frontend)

---

## Workstreams

Each workstream has a dedicated document with full implementation details including database migrations, API contracts, code patterns, component wireframes, and file change summaries.

### Phase 1 — Foundation (no cross-dependencies, can be parallelized)

| Workstream | Document | Effort | Developer |
|------------|----------|--------|-----------|
| WS7: Fix Frontend Tabs (Real Data) | [docs/workstream-7-frontend-tabs.md](docs/workstream-7-frontend-tabs.md) | Small | Frontend |
| WS3: Vet-ICD-O-canine-1 Codes | [docs/workstream-3-icd-codes.md](docs/workstream-3-icd-codes.md) | Small-Medium | Backend |
| WS5: Trend Line Visualization | [docs/workstream-5-trend-visualization.md](docs/workstream-5-trend-visualization.md) | Medium | Frontend |

### Phase 2 — Core Features (build on Phase 1)

| Workstream | Document | Effort | Dependencies |
|------------|----------|--------|--------------|
| WS4: Authentication & Access Control | [docs/workstream-4-authentication.md](docs/workstream-4-authentication.md) | Medium | None (but needed by WS1, WS6) |
| WS1: CSV Upload & Data Ingestion | [docs/workstream-1-csv-upload.md](docs/workstream-1-csv-upload.md) | Large | WS4 (auth on upload endpoints) |
| WS2: Real BERT Integration | [docs/workstream-2-bert-integration.md](docs/workstream-2-bert-integration.md) | Large | None directly, integrates with WS1 |

### Phase 3 — Advanced Features (depend on Phase 2)

| Workstream | Document | Effort | Dependencies |
|------------|----------|--------|--------------|
| WS6: Ambiguous Diagnosis Flagging & Review | [docs/workstream-6-review-workflow.md](docs/workstream-6-review-workflow.md) | Medium | WS2 (BERT confidence), WS4 (auth) |

### Phase 4 — Quality

| Workstream | Document | Effort | Dependencies |
|------------|----------|--------|--------------|
| WS8: Tests | [docs/workstream-8-tests.md](docs/workstream-8-tests.md) | Medium-Large | All other workstreams |

---

## Implementation Order

Full dependency graph, execution order rationale, and Gantt timeline:

- **[Implementation Order](docs/implementation-order.md)**

---

## File Change Summary

Complete list of all 26+ new files to create and 18 existing files to modify:

- **[File Change Summary](docs/file-change-summary.md)**
