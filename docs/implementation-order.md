# Implementation Order

[Back to Overview](../IMPLEMENTATION_PLAN.md)

---

The workstreams have dependencies. Here is the recommended execution order with rationale:

```
Phase 1 — Foundation (no cross-dependencies, can be parallelized)
│
├── Workstream 7: Fix frontend tabs (real data)
│   Rationale: Quick win. Only frontend changes. No backend work.
│   Effort: Small (3 new components, refactor App.tsx)
│   Can be done by: Frontend developer
│
├── Workstream 3: Vet-ICD-O-canine-1 codes
│   Rationale: Database schema change — do early before other migrations.
│   Effort: Small-Medium (1 migration, model + schema + router updates)
│   Can be done by: Backend developer
│
└── Workstream 5: Trend line visualization
    Rationale: No backend changes needed (endpoints exist). Frontend only.
    Effort: Medium (install recharts, build TrendChart, wire to API)
    Can be done by: Frontend developer


Phase 2 — Core Features (build on Phase 1)
│
├── Workstream 4: Authentication
│   Rationale: Must exist before upload & review (they require auth).
│   Effort: Medium (JWT service, auth router, login page, auth context)
│   Depends on: Nothing (but needed by WS1 and WS6)
│
├── Workstream 1: CSV upload & data ingestion
│   Rationale: Core feature — data entry into the registry.
│   Effort: Large (ingestion service, upload router, upload page, validation)
│   Depends on: WS4 for auth on upload endpoints
│
└── Workstream 2: Real BERT integration
    Rationale: Needed for free-text processing in upload pipeline + review.
    Effort: Large (model integration, Celery worker, Redis, Docker changes)
    Depends on: Nothing directly, but integrates with WS1


Phase 3 — Advanced Features (depend on Phase 2)
│
└── Workstream 6: Ambiguous diagnosis flagging & review
    Rationale: Requires BERT (for confidence scores) + auth (for reviewers).
    Effort: Medium (review router, flagging logic, review queue UI)
    Depends on: WS2 (BERT confidence), WS4 (authenticated reviewers)


Phase 4 — Quality
│
└── Workstream 8: Tests
    Rationale: Test all implemented features. Write tests after code is stable.
    Effort: Medium-Large (backend + frontend test suites)
    Depends on: All other workstreams (to test their functionality)
```

**Gantt-style timeline (if 2 developers working in parallel):**

```
                   Week 1        Week 2        Week 3        Week 4        Week 5
Dev A (Frontend):  [WS7: Tabs]   [WS5: Trends] [WS1.5: Upload UI] [WS6.5: Review UI] [WS8: FE Tests]
Dev B (Backend):   [WS3: ICD-O]  [WS4: Auth]   [WS1: Upload API]  [WS2: BERT+Worker]  [WS6: Review]
                                                                     [WS8: BE Tests]
```
