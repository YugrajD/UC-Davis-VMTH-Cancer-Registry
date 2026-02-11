# File Change Summary

[Back to Overview](../IMPLEMENTATION_PLAN.md)

---

## New Files to Create (26+ files)

| File | Workstream | Layer |
|------|------------|-------|
| `database/migrations/007_raw_uploads.sql` | 1 | DB |
| `database/migrations/008_icd_codes.sql` | 3 | DB |
| `database/migrations/009_users.sql` | 4 | DB |
| `database/migrations/010_review_status.sql` | 6 | DB |
| `database/migrations/011_nlp_jobs.sql` | 2 | DB |
| `backend/app/routers/upload.py` | 1 | Backend |
| `backend/app/routers/auth.py` | 4 | Backend |
| `backend/app/routers/review.py` | 6 | Backend |
| `backend/app/services/ingestion_service.py` | 1 | Backend |
| `backend/app/services/auth_service.py` | 4 | Backend |
| `backend/app/services/nlp_worker.py` | 2 | Backend |
| `backend/tests/` (11 test files) | 8 | Backend |
| `frontend/src/components/UploadPage/UploadPage.tsx` | 1 | Frontend |
| `frontend/src/components/LoginPage/LoginPage.tsx` | 4 | Frontend |
| `frontend/src/components/TrendChart/TrendChart.tsx` | 5 | Frontend |
| `frontend/src/components/ReviewQueue/ReviewQueue.tsx` | 6 | Frontend |
| `frontend/src/components/BreedDisparities/BreedDisparities.tsx` | 7 | Frontend |
| `frontend/src/components/CancerTypesChart/CancerTypesChart.tsx` | 7 | Frontend |
| `frontend/src/components/RegionalComparison/RegionalComparison.tsx` | 7 | Frontend |
| `frontend/src/hooks/useAuth.ts` | 4 | Frontend |
| `frontend/src/components/__tests__/` (5+ test files) | 8 | Frontend |

## Existing Files to Modify (18 files)

| File | Workstreams | Changes |
|------|-------------|---------|
| `backend/app/main.py` | 1, 4, 6 | Register upload, auth, review routers |
| `backend/app/config.py` | 2, 4, 6 | Add USE_REAL_BERT, SECRET_KEY, REDIS_URL, CONFIDENCE_THRESHOLD |
| `backend/app/models/models.py` | 3, 4, 6 | Add ICD-O cols to CancerType, add User model, add review cols to PathologyReport |
| `backend/app/schemas/schemas.py` | 1, 3, 4, 6 | Add Upload*, ICD-O, User/Token, Review* schemas |
| `backend/app/services/bert_service.py` | 2, 6 | Rewrite to use ml/ module, add flagging |
| `backend/app/routers/dashboard.py` | 3 | Include ICD-O codes in filter options |
| `backend/app/routers/incidence.py` | 3 | Include ICD-O code in incidence queries |
| `backend/app/routers/search.py` | 2, 3 | Use updated bert_service, return ICD-O code |
| `backend/requirements.txt` | 2, 4, 8 | Add transformers, torch, celery, redis, jose, passlib, pytest |
| `backend/Dockerfile` | 2 | Install ml/ dependencies |
| `ml/model/classifier.py` | 2 | Add real BERT mode alongside keyword fallback |
| `docker-compose.yml` | 1, 2, 3, 4, 6 | Mount new migrations, add redis + nlp_worker services, add env vars |
| `frontend/package.json` | 5, 8 | Add recharts, vitest, testing-library |
| `frontend/vite.config.ts` | 8 | Add test configuration |
| `frontend/src/App.tsx` | 1, 4, 5, 6, 7 | AuthProvider wrapper, replace hardcoded tabs, add TrendChart + new tabs |
| `frontend/src/api/client.ts` | 1, 4, 5, 6, 7 | Auth headers, upload functions, trends functions, review functions |
| `frontend/src/types/index.ts` | 1, 3, 4, 5, 6, 7 | Extend TabType, add Upload/Trend/Review/Auth types, CancerTypeOption |
| `frontend/src/components/Navigation/Navigation.tsx` | 1, 4, 6 | Conditional auth tabs, sign in/out button |
| `frontend/src/components/index.ts` | 1, 4, 5, 6, 7 | Export all new components |
