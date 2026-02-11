# Workstream 2: Real BERT Integration & Async NLP Worker

[Back to Overview](../IMPLEMENTATION_PLAN.md)

---

**Gaps addressed:** #2, #8 (US #3, Architecture)

## 2.1 Consolidate Classifier Code

**Problem:** Two separate keyword matchers exist:
- `ml/model/classifier.py` → `VetBERTClassifier` with weighted keywords
- `backend/app/services/bert_service.py` → `BertClassifier` with unweighted keywords

**Solution:** Delete `backend/app/services/bert_service.py` inline matcher. Make the backend import from `ml/model/classifier.py` (already volume-mounted in Docker).

**Update `backend/app/services/bert_service.py`:**

```python
"""
BERT classification service.
Wraps the ml/model/classifier.py module, which provides either
a real BERT model (production) or keyword-based fallback (development).
"""

import sys
import os

# ml/ is mounted at /ml in the Docker container
sys.path.insert(0, "/ml")
from model.classifier import VetBERTClassifier

from app.config import settings
from app.schemas.schemas import ClassifyResult


class BertService:
    def __init__(self):
        self.classifier = VetBERTClassifier(
            use_real_model=settings.USE_REAL_BERT,
            model_path=settings.BERT_MODEL_PATH,
        )

    def classify(self, text: str) -> ClassifyResult:
        result = self.classifier.predict(text)
        return ClassifyResult(
            predicted_cancer_type=result["predicted_label"],
            confidence=result["confidence"],
            top_predictions=[
                {"cancer_type": ct, "confidence": conf}
                for ct, conf in list(result["all_probabilities"].items())[:5]
            ],
        )

# Singleton instance
bert_service = BertService()
```

## 2.2 Upgrade `ml/model/classifier.py` to Support Real BERT

**Architecture:**

```python
class VetBERTClassifier:
    def __init__(self, use_real_model: bool = False, model_path: str = "./vetbert-finetuned"):
        self.use_real_model = use_real_model
        if use_real_model:
            self._load_bert(model_path)
        else:
            self._load_keyword_fallback()

    def _load_bert(self, model_path: str):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _load_keyword_fallback(self):
        # Existing weighted keyword matching (current code)
        ...

    def predict(self, text: str) -> dict:
        if self.use_real_model:
            return self._predict_bert(text)
        return self._predict_keywords(text)

    def _predict_bert(self, text: str) -> dict:
        import torch

        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=512, padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]

        sorted_indices = torch.argsort(probs, descending=True)
        all_probs = {
            CANCER_LABELS[i.item()]: round(probs[i.item()].item(), 4)
            for i in sorted_indices
        }
        predicted_idx = sorted_indices[0].item()

        return {
            "predicted_label": CANCER_LABELS[predicted_idx],
            "confidence": round(probs[predicted_idx].item(), 4),
            "all_probabilities": all_probs,
        }

    def _predict_keywords(self, text: str) -> dict:
        # ... existing weighted keyword code from current classifier.py ...
```

## 2.3 Configuration Additions

**`backend/app/config.py`** — add:

```python
USE_REAL_BERT: bool = False   # Toggle real BERT vs keyword fallback
BERT_MODEL_PATH: str = "/ml/model/weights/vetbert-finetuned"
CONFIDENCE_THRESHOLD: float = 0.7   # For flagging (Workstream 6)
REDIS_URL: str = "redis://redis:6379/0"
```

## 2.4 Async NLP Worker with Celery + Redis

**Architecture decision:** Use Celery + Redis for a decoupled worker that can process pathology reports without blocking the API, matching the architecture diagram's "separate NLP processing worker."

**New file: `backend/app/services/nlp_worker.py`**

```python
"""
Async NLP worker using Celery for background BERT classification.
Processes pathology reports queued by the upload pipeline.
"""

from celery import Celery
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import settings

celery_app = Celery("nlp_worker", broker=settings.REDIS_URL)

# Sync DB session (Celery tasks are synchronous)
sync_engine = create_engine(settings.DATABASE_URL_SYNC)
SyncSession = sessionmaker(bind=sync_engine)


@celery_app.task(name="classify_report")
def classify_report(report_id: int):
    """
    1. Load pathology_report by ID
    2. Run BERT classification
    3. Write classification + confidence_score back to DB
    4. Set review_status based on confidence threshold
    5. Update nlp_jobs status
    """
    from ml.model.classifier import VetBERTClassifier
    classifier = VetBERTClassifier(
        use_real_model=settings.USE_REAL_BERT,
        model_path=settings.BERT_MODEL_PATH,
    )

    with SyncSession() as db:
        report = db.query(PathologyReport).get(report_id)
        if not report:
            return

        result = classifier.predict(report.report_text)

        report.classification = result["predicted_label"]
        report.confidence_score = result["confidence"]
        report.review_status = (
            "auto_accepted" if result["confidence"] >= settings.CONFIDENCE_THRESHOLD
            else "flagged"
        )
        db.commit()


@celery_app.task(name="classify_batch")
def classify_batch(report_ids: list[int]):
    """Classify multiple reports in sequence."""
    for report_id in report_ids:
        classify_report(report_id)
```

## 2.5 Docker Compose Additions

Add to `docker-compose.yml`:

```yaml
  redis:
    image: redis:7-alpine
    container_name: vmth_cancer_redis
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 5s
      retries: 5

  nlp_worker:
    build:
      context: ./backend
      dockerfile: Dockerfile
    container_name: vmth_cancer_nlp_worker
    environment:
      DATABASE_URL: postgresql+asyncpg://postgres:postgres@db:5432/vmth_cancer
      DATABASE_URL_SYNC: postgresql://postgres:postgres@db:5432/vmth_cancer
      REDIS_URL: redis://redis:6379/0
      USE_REAL_BERT: "false"
      BERT_MODEL_PATH: /ml/model/weights/vetbert-finetuned
    volumes:
      - ./backend:/app
      - ./ml:/ml
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
    command: celery -A app.services.nlp_worker.celery_app worker --loglevel=info --concurrency=2
```

Also add `REDIS_URL` to the backend service environment.

## 2.6 Backend Requirements Additions

Add to `backend/requirements.txt`:

```
# NLP / ML
transformers>=4.38.0
torch>=2.2.0
# Task queue
celery>=5.3.0
redis>=5.0.0
```

## 2.7 NLP Jobs Table

**New migration: `database/migrations/011_nlp_jobs.sql`**

```sql
CREATE TABLE IF NOT EXISTS nlp_jobs (
    id SERIAL PRIMARY KEY,
    report_id INTEGER REFERENCES pathology_reports(id),
    status VARCHAR(20) NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    queued_at TIMESTAMP NOT NULL DEFAULT NOW(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT
);

CREATE INDEX IF NOT EXISTS idx_nlp_jobs_status ON nlp_jobs (status);
CREATE INDEX IF NOT EXISTS idx_nlp_jobs_report ON nlp_jobs (report_id);
```

## 2.8 Integration with Upload Pipeline

When `ingestion_service.py` encounters a `pathology_notes` column:
1. Create a `pathology_reports` row with `classification=NULL`, `confidence_score=NULL`.
2. Create an `nlp_jobs` row with `status='pending'`.
3. Dispatch `classify_report.delay(report_id)` to the Celery queue.

The upload endpoint returns immediately; the NLP worker processes asynchronously.

## 2.9 Files Summary

**Files to create:**
| File | Purpose |
|------|---------|
| `backend/app/services/nlp_worker.py` | Celery task definitions |
| `database/migrations/011_nlp_jobs.sql` | Job tracking table |

**Files to modify:**
| File | Change |
|------|--------|
| `ml/model/classifier.py` | Add real BERT mode, keep keyword fallback |
| `backend/app/services/bert_service.py` | Rewrite to import from ml/, use config toggle |
| `backend/app/routers/search.py` | Use updated bert_service |
| `backend/app/config.py` | Add `USE_REAL_BERT`, `BERT_MODEL_PATH`, `REDIS_URL`, `CONFIDENCE_THRESHOLD` |
| `backend/requirements.txt` | Add transformers, torch, celery, redis |
| `backend/Dockerfile` | Install ml dependencies |
| `docker-compose.yml` | Add redis + nlp_worker services, add env vars to backend |
| `backend/app/services/ingestion_service.py` | Queue NLP jobs after inserting reports |
