# Workstream 8: Tests

[Back to Overview](../IMPLEMENTATION_PLAN.md)

---

**Gaps addressed:** #9 (Maintainability requirement)

## 8.1 Backend Test Architecture

**Directory structure:**
```
backend/
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # Shared fixtures and mock helpers
│   ├── test_admin_users.py      # Admin role management (GET/PUT roles, self-demotion, env seed)
│   ├── test_dashboard.py        # Dashboard summary and filters endpoints
│   ├── test_incidence.py        # Incidence endpoints (by-cancer-type, by-species, by-breed, breed-detail)
│   ├── test_security.py         # Security hardening (body size, auth gates, input validation, LIKE escaping, error sanitization)
│   ├── test_trends.py           # (planned) Yearly aggregation, by-cancer-type series
│   ├── test_geo.py              # (planned) GeoJSON structure, county geometry
│   ├── test_search.py           # (planned) Search/classify endpoint tests
│   ├── test_upload.py           # (planned) CSV upload validation tests
│   ├── test_auth.py             # (planned) Authentication flow tests
│   ├── test_review.py           # (planned) Diagnosis review queue tests
│   └── test_ingestion.py        # (planned) Ingestion service unit tests
```

**`conftest.py` architecture:**

```python
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from app.main import app
from app.database import get_db, Base

# Use a test database
TEST_DB_URL = "postgresql+asyncpg://postgres:postgres@db:5432/vmth_cancer_test"

@pytest_asyncio.fixture
async def db_session():
    """Create a fresh test database session."""
    engine = create_async_engine(TEST_DB_URL)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, class_=AsyncSession)
    async with session_factory() as session:
        yield session
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

@pytest_asyncio.fixture
async def client(db_session):
    """Create test HTTP client with DB override."""
    app.dependency_overrides[get_db] = lambda: db_session
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as ac:
        yield ac
    app.dependency_overrides.clear()

@pytest.fixture
def sample_csv():
    """Generate a valid test CSV file."""
    return b"species,breed,sex,age_years,county,registered_date,cancer_type,diagnosis_date\n" \
           b"Dog,Golden Retriever,Male,7.5,Sacramento,2024-01-15,Lymphoma,2024-02-01\n"
```

**Key test cases per file:**

| File | Test Cases | Status |
|------|-----------|--------|
| `test_admin_users.py` | GET/PUT role management, env fallback, admin implication, self-demotion block, email normalization, auth gating, role_seed logic | Implemented (15 tests) |
| `test_dashboard.py` | Summary returns correct totals, species percentages, year range, empty DB defaults, top cancers ordering, filters endpoint | Implemented (10 tests) |
| `test_incidence.py` | Incidence with no filters, schema validation, total counts, filter echo, by-cancer-type, by-species, by-breed, breed-detail, SEX_MAP | Implemented (15 tests) |
| `test_security.py` | Body size middleware (413), upload path exemption, search auth gates, ClassifyRequest validation, IngestionJobReview Literal/max_length, LIKE escaping, error message sanitization | Implemented (17 tests) |
| `test_trends.py` | Yearly aggregation correct, by-cancer-type returns multiple series, filter narrows results | Planned |
| `test_geo.py` | GeoJSON has valid structure, counties have geometry, filter reduces case counts | Planned |
| `test_search.py` | Classify returns valid cancer type, confidence between 0-1 | Planned |
| `test_upload.py` | Valid CSV accepted, missing columns return 422, invalid breed rejected with error, file type validation | Planned |
| `test_auth.py` | Login returns token, invalid password returns 401, protected endpoint returns 403 without token | Planned |
| `test_review.py` | Queue returns flagged reports, approve updates status, reclassify changes classification | Planned |
| `test_ingestion.py` | Sex aliases mapped correctly, case-insensitive breed matching, date parsing | Planned |

**Dependencies to add to `backend/requirements.txt`:**
```
pytest>=8.0.0
pytest-asyncio>=0.23.0
```

## 8.2 Frontend Test Architecture

**Directory structure:**
```
frontend/
├── src/
│   ├── components/
│   │   └── __tests__/
│   │       ├── Filters.test.tsx
│   │       ├── ChoroplethMap.test.tsx
│   │       ├── TrendChart.test.tsx
│   │       ├── UploadPage.test.tsx
│   │       └── ReviewQueue.test.tsx
│   ├── hooks/
│   │   └── __tests__/
│   │       ├── useFilteredData.test.ts
│   │       └── useAuth.test.ts
│   └── api/
│       └── __tests__/
│           └── client.test.ts
```

**Dependencies to add to `frontend/package.json` (devDependencies):**
```json
"vitest": "^3.0.0",
"@testing-library/react": "^16.0.0",
"@testing-library/jest-dom": "^6.0.0",
"jsdom": "^25.0.0"
```

**Add test script to `frontend/package.json`:**
```json
"test": "vitest",
"test:coverage": "vitest --coverage"
```

**Vitest configuration in `frontend/vite.config.ts`:**
```typescript
export default defineConfig({
  // ... existing config ...
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './src/test-setup.ts',
  },
});
```

## 8.3 Files Summary

**Files to create:**
| File | Purpose |
|------|---------|
| `backend/tests/__init__.py` | Package init |
| `backend/tests/conftest.py` | Shared fixtures |
| `backend/tests/test_dashboard.py` | Dashboard tests |
| `backend/tests/test_incidence.py` | Incidence tests |
| `backend/tests/test_trends.py` | Trends tests |
| `backend/tests/test_geo.py` | Geo tests |
| `backend/tests/test_search.py` | Search tests |
| `backend/tests/test_upload.py` | Upload tests |
| `backend/tests/test_auth.py` | Auth tests |
| `backend/tests/test_review.py` | Review tests |
| `backend/tests/test_ingestion.py` | Ingestion unit tests |
| `frontend/src/test-setup.ts` | Test environment setup |
| `frontend/src/components/__tests__/*.test.tsx` | Component tests |
| `frontend/src/hooks/__tests__/*.test.ts` | Hook tests |
| `frontend/src/api/__tests__/client.test.ts` | API client tests |

**Files to modify:**
| File | Change |
|------|--------|
| `backend/requirements.txt` | Add pytest, pytest-asyncio |
| `frontend/package.json` | Add vitest, testing-library, jsdom |
| `frontend/vite.config.ts` | Add test configuration |
