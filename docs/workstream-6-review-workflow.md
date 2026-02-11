# Workstream 6: Ambiguous Diagnosis Flagging & Review

[Back to Overview](../IMPLEMENTATION_PLAN.md)

---

**Gaps addressed:** #6 (US #8)

## 6.1 Database Migration: `database/migrations/010_review_status.sql`

```sql
-- 010_review_status.sql
-- Add review workflow to pathology reports

ALTER TABLE pathology_reports
    ADD COLUMN IF NOT EXISTS review_status VARCHAR(20)
        DEFAULT 'pending'
        CHECK (review_status IN ('pending', 'auto_accepted', 'flagged', 'manually_reviewed', 'rejected')),
    ADD COLUMN IF NOT EXISTS reviewed_by INTEGER REFERENCES users(id),
    ADD COLUMN IF NOT EXISTS reviewed_at TIMESTAMP;

-- Set existing reports with classification to auto_accepted
UPDATE pathology_reports SET review_status = 'auto_accepted' WHERE classification IS NOT NULL;
UPDATE pathology_reports SET review_status = 'pending' WHERE classification IS NULL;

CREATE INDEX IF NOT EXISTS idx_reports_review_status ON pathology_reports (review_status);
```

## 6.2 Backend Model Update

**`backend/app/models/models.py`** — update `PathologyReport`:

```python
class PathologyReport(Base):
    __tablename__ = "pathology_reports"

    id = Column(Integer, primary_key=True)
    case_id = Column(Integer, ForeignKey("cancer_cases.id"), nullable=False)
    report_text = Column(Text, nullable=False)
    classification = Column(String(100))
    confidence_score = Column(Numeric(5, 4))
    report_date = Column(Date, nullable=False)
    review_status = Column(String(20), default="pending")       # NEW
    reviewed_by = Column(Integer, ForeignKey("users.id"))       # NEW
    reviewed_at = Column(DateTime)                              # NEW

    case = relationship("CancerCase", back_populates="reports")
    reviewer = relationship("User")                             # NEW
```

## 6.3 Flagging Logic

The confidence threshold is configured in `backend/app/config.py` as `CONFIDENCE_THRESHOLD = 0.7`.

**In `backend/app/services/nlp_worker.py` (Workstream 2) or `bert_service.py`:**

```python
def determine_review_status(confidence: float) -> str:
    if confidence >= settings.CONFIDENCE_THRESHOLD:
        return "auto_accepted"
    return "flagged"
```

## 6.4 Review Router: `backend/app/routers/review.py`

**Endpoints:**

### `GET /api/v1/review/queue`

```
Query params: ?status=flagged&limit=20&offset=0&cancer_type=Lymphoma&min_confidence=0.3&max_confidence=0.7

Response 200:
{
  "reports": [
    {
      "id": 456,
      "case_id": 123,
      "report_text": "Histopathology reveals...",
      "classification": "Lymphoma",
      "confidence_score": 0.55,
      "report_date": "2024-03-15",
      "review_status": "flagged",
      "patient_info": {            // Join data for context
        "species": "Dog",
        "breed": "Golden Retriever",
        "age_years": 8.5,
        "county": "Sacramento"
      }
    }
  ],
  "total": 42,
  "stats": {
    "flagged": 42,
    "pending": 5,
    "auto_accepted": 893,
    "manually_reviewed": 31,
    "rejected": 3
  }
}
```

### `PUT /api/v1/review/{report_id}`

```
Request:
{
  "action": "approve",                    // approve | reclassify | reject
  "new_classification": "Mast Cell Tumor" // required if action=reclassify
}

Response 200:
{
  "id": 456,
  "review_status": "manually_reviewed",
  "classification": "Mast Cell Tumor",
  "reviewed_by": "admin",
  "reviewed_at": "2026-02-08T15:30:00Z"
}
```

- `approve`: Keep the BERT classification, set `review_status = 'manually_reviewed'`
- `reclassify`: Override classification with `new_classification`, set `review_status = 'manually_reviewed'`
- `reject`: Set `review_status = 'rejected'`, mark case as needing further investigation

### `GET /api/v1/review/stats`

```
Response 200:
{
  "pending": 5,
  "flagged": 42,
  "auto_accepted": 893,
  "manually_reviewed": 31,
  "rejected": 3,
  "total": 974,
  "average_confidence": 0.78
}
```

## 6.5 Frontend — Review Queue Component

**File:** `frontend/src/components/ReviewQueue/ReviewQueue.tsx`

```
┌────────────────────────────────────────────────────────────────┐
│  Review Queue                                  Stats: 42 flagged│
│                                                                 │
│  Filters: [Confidence: 0.3-0.7] [Cancer Type: All] [Search...] │
│                                                                 │
│  ┌─────┬──────────────────┬──────────┬──────┬─────────────────┐│
│  │  ID │ Report Excerpt    │ Predicted│ Conf │ Actions         ││
│  ├─────┼──────────────────┼──────────┼──────┼─────────────────┤│
│  │ 456 │ Histopathology   │ Lymphoma │ 55%  │ [Ap] [Re] [Rj]  ││
│  │     │ reveals diffuse  │          │      │                 ││
│  │     │ large B-cell...  │          │      │                 ││
│  ├─────┼──────────────────┼──────────┼──────┼─────────────────┤│
│  │ 457 │ Excisional biopsy│ MCT      │ 48%  │ [Ap] [Re] [Rj]  ││
│  │     │ reveals dermal...│          │      │                 ││
│  └─────┴──────────────────┴──────────┴──────┴─────────────────┘│
│                                                                 │
│  Showing 1-20 of 42  [< Prev] [Next >]                         │
│                                                                 │
│  Expanded report view (on row click):                           │
│  ┌──────────────────────────────────────────────────┐          │
│  │ Full report text...                               │          │
│  │ Patient: Dog, Golden Retriever, 8.5y, Sacramento  │          │
│  │ BERT prediction: Lymphoma (55%)                   │          │
│  │ Alternatives: MCT (20%), Hemangiosarcoma (15%)    │          │
│  │                                                    │          │
│  │ Reclassify as: [dropdown of cancer types]         │          │
│  │ [Approve] [Reclassify] [Reject]                   │          │
│  └──────────────────────────────────────────────────┘          │
└────────────────────────────────────────────────────────────────┘
```

## 6.6 Files Summary

**Files to create:**
| File | Purpose |
|------|---------|
| `database/migrations/010_review_status.sql` | Review workflow columns |
| `backend/app/routers/review.py` | Review queue endpoints |
| `frontend/src/components/ReviewQueue/ReviewQueue.tsx` | Review UI |

**Files to modify:**
| File | Change |
|------|--------|
| `backend/app/models/models.py` | Add review columns to PathologyReport |
| `backend/app/schemas/schemas.py` | Add Review* schemas, update ReportOut |
| `backend/app/services/bert_service.py` | Add `determine_review_status()` |
| `backend/app/config.py` | `CONFIDENCE_THRESHOLD` (already added in WS2) |
| `backend/app/main.py` | Register review router |
| `frontend/src/api/client.ts` | Add `fetchReviewQueue`, `updateReview`, `fetchReviewStats` |
| `frontend/src/types/index.ts` | Add ReviewItem, ReviewQueueResponse types |
| `frontend/src/components/Navigation/Navigation.tsx` | Add Review tab (auth-only) |
| `frontend/src/components/index.ts` | Export ReviewQueue |
| `frontend/src/App.tsx` | Render ReviewQueue for review tab |
| `docker-compose.yml` | Mount migration 010 |
