# Workstream 3: Vet-ICD-O-canine-1 Coding System

[Back to Overview](../IMPLEMENTATION_PLAN.md)

---

**Gaps addressed:** #3 (US #1)

## 3.1 ICD-O Code Reference

The Vet-ICD-O-canine-1 system uses the same morphology code structure as human ICD-O-3 (format: `XXXX/B` where XXXX = morphology, B = behavior).

**Mapping for existing cancer types:**

| Cancer Type | ICD-O Morphology Code | ICD-O Label |
|---|---|---|
| Lymphoma | 9590/3 | Malignant lymphoma, NOS |
| Mast Cell Tumor | 9740/3 | Mast cell sarcoma |
| Osteosarcoma | 9180/3 | Osteosarcoma, NOS |
| Hemangiosarcoma | 9120/3 | Hemangiosarcoma |
| Melanoma | 8720/3 | Malignant melanoma, NOS |
| Squamous Cell Carcinoma | 8070/3 | Squamous cell carcinoma, NOS |
| Fibrosarcoma | 8810/3 | Fibrosarcoma, NOS |
| Transitional Cell Carcinoma | 8120/3 | Transitional cell carcinoma, NOS |

## 3.2 Database Migration: `database/migrations/008_icd_codes.sql`

```sql
-- 008_icd_codes.sql
-- Add Vet-ICD-O-canine-1 coding to cancer types

ALTER TABLE cancer_types
    ADD COLUMN IF NOT EXISTS icd_o_morphology_code VARCHAR(10),
    ADD COLUMN IF NOT EXISTS icd_o_topography_code VARCHAR(10),
    ADD COLUMN IF NOT EXISTS icd_o_label VARCHAR(200);

UPDATE cancer_types SET icd_o_morphology_code = '9590/3', icd_o_label = 'Malignant lymphoma, NOS' WHERE name = 'Lymphoma';
UPDATE cancer_types SET icd_o_morphology_code = '9740/3', icd_o_label = 'Mast cell sarcoma' WHERE name = 'Mast Cell Tumor';
UPDATE cancer_types SET icd_o_morphology_code = '9180/3', icd_o_label = 'Osteosarcoma, NOS' WHERE name = 'Osteosarcoma';
UPDATE cancer_types SET icd_o_morphology_code = '9120/3', icd_o_label = 'Hemangiosarcoma' WHERE name = 'Hemangiosarcoma';
UPDATE cancer_types SET icd_o_morphology_code = '8720/3', icd_o_label = 'Malignant melanoma, NOS' WHERE name = 'Melanoma';
UPDATE cancer_types SET icd_o_morphology_code = '8070/3', icd_o_label = 'Squamous cell carcinoma, NOS' WHERE name = 'Squamous Cell Carcinoma';
UPDATE cancer_types SET icd_o_morphology_code = '8810/3', icd_o_label = 'Fibrosarcoma, NOS' WHERE name = 'Fibrosarcoma';
UPDATE cancer_types SET icd_o_morphology_code = '8120/3', icd_o_label = 'Transitional cell carcinoma, NOS' WHERE name = 'Transitional Cell Carcinoma';
```

## 3.3 Backend Model Update

**`backend/app/models/models.py`** — add to `CancerType`:

```python
class CancerType(Base):
    __tablename__ = "cancer_types"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, unique=True)
    description = Column(Text)
    icd_o_morphology_code = Column(String(10))      # NEW
    icd_o_topography_code = Column(String(10))       # NEW
    icd_o_label = Column(String(200))                # NEW

    cases = relationship("CancerCase", back_populates="cancer_type")
```

## 3.4 Schema Updates

**`backend/app/schemas/schemas.py`** — update `CancerTypeOut`:

```python
class CancerTypeOut(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    icd_o_morphology_code: Optional[str] = None    # NEW
    icd_o_topography_code: Optional[str] = None    # NEW
    icd_o_label: Optional[str] = None              # NEW
    model_config = {"from_attributes": True}
```

Also update `IncidenceRecord` to include `icd_o_code`:
```python
class IncidenceRecord(BaseModel):
    cancer_type: str
    icd_o_code: Optional[str] = None   # NEW
    county: Optional[str] = None
    species: Optional[str] = None
    breed: Optional[str] = None
    year: Optional[int] = None
    count: int
```

And update `ClassifyResult`:
```python
class ClassifyResult(BaseModel):
    predicted_cancer_type: str
    icd_o_code: Optional[str] = None   # NEW
    confidence: float
    top_predictions: List[dict]
```

## 3.5 Router Updates

**`backend/app/routers/incidence.py`** — add ICD-O code to the SELECT clause wherever `CancerType.name` is selected. Example for `/by-cancer-type`:

```python
# Before:
select(CancerType.name.label("cancer_type"), func.count(CancerCase.id).label("count"))

# After:
select(
    CancerType.name.label("cancer_type"),
    CancerType.icd_o_morphology_code.label("icd_o_code"),
    func.count(CancerCase.id).label("count"),
)
```

**`backend/app/routers/search.py`** — after classification, look up the ICD-O code:

```python
# After getting classification result:
if result.predicted_cancer_type != "Unknown":
    ct = await db.execute(
        select(CancerType.icd_o_morphology_code)
        .where(CancerType.name == result.predicted_cancer_type)
    )
    code = ct.scalar_one_or_none()
    result.icd_o_code = code
```

## 3.6 Frontend Display

**`frontend/src/types/index.ts`** — update `CANCER_TYPES` to include codes:

```typescript
export interface CancerTypeOption {
  name: string;
  icd_o_code?: string;
}
```

**`frontend/src/components/Filters/Filters.tsx`** — display ICD-O codes in dropdown:
```tsx
// Show: "Lymphoma (9590/3)" in the cancer type dropdown
```

**`frontend/src/components/CountyTable/CountyTable.tsx`** and **`SummaryTable/SummaryTable.tsx`** — add ICD-O code column where cancer types are displayed.

## 3.7 Files Summary

**Files to create:**
| File | Purpose |
|------|---------|
| `database/migrations/008_icd_codes.sql` | Add ICD-O columns and data |

**Files to modify:**
| File | Change |
|------|--------|
| `backend/app/models/models.py` | Add 3 columns to CancerType |
| `backend/app/schemas/schemas.py` | Add ICD-O fields to CancerTypeOut, IncidenceRecord, ClassifyResult |
| `backend/app/routers/dashboard.py` | Include ICD-O in filter options response |
| `backend/app/routers/incidence.py` | Include ICD-O code in all incidence queries |
| `backend/app/routers/search.py` | Look up and return ICD-O code for predictions |
| `frontend/src/types/index.ts` | Add CancerTypeOption interface |
| `frontend/src/components/Filters/Filters.tsx` | Display ICD-O codes in dropdown |
| `frontend/src/components/CountyTable/CountyTable.tsx` | Add ICD-O column |
| `frontend/src/components/SummaryTable/SummaryTable.tsx` | Add ICD-O column |
| `docker-compose.yml` | Mount migration 008 |
