"""BERT search and classification endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Request

from app.auth import CurrentUser, get_current_user
from app.config import settings
from app.rate_limit import limiter
from app.schemas.schemas import ClassifyRequest, ClassifyResult
from app.services.bert_service import BertClassifier

router = APIRouter(prefix="/api/v1/search", tags=["search"])

classifier = BertClassifier()


def _escape_like(value: str) -> str:
    """Escape SQL LIKE wildcard characters in user input."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


@router.post("/classify", response_model=ClassifyResult)
@limiter.limit(settings.RATE_LIMIT_EXPENSIVE)
async def classify_report(
    body: ClassifyRequest,
    request: Request,
    _user: CurrentUser = Depends(get_current_user),
):
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="Report text is required")

    result = classifier.classify(body.text)
    return result
