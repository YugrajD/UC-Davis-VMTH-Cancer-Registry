"""Single source of truth for which case_diagnoses rows are visible to
public dashboard aggregations.

The review queue (`backend/app/routers/diagnoses_review.py`) routes
low-confidence and ambiguous predictions to `pending` status; admins can
move them to `confirmed`, `corrected`, or `rejected`. Aggregate stats
must exclude `pending` and `rejected` so rejected predictions don't
pollute counts and pending ones don't bias numbers before sign-off.

Use `VISIBLE_REVIEW_STATUSES` for raw SQL and `apply_review_filter()` for
SQLAlchemy queries. Pass `include_pending=True` in the rare case where a
caller deliberately wants the pre-review picture (e.g. an admin debug
view).
"""

from typing import Iterable

from sqlalchemy import or_
from sqlalchemy.sql import Select

from app.models.models import CaseDiagnosis, Patient


# Exclude patients whose zip code was resolved as non-California.
# A patient with zip_code IS NULL has no geographic data (acceptable — include them).
# A patient with zip_code IS NOT NULL but county_id IS NULL had a non-CA zip → exclude.
CALIFORNIA_PATIENT_FILTER = or_(
    Patient.zip_code.is_(None),
    Patient.county_id.is_not(None),
)

# Tuple form (immutable) for the WHERE-IN clause.
VISIBLE_REVIEW_STATUSES: tuple[str, ...] = ("confirmed", "corrected")
VISIBLE_REVIEW_STATUSES_WITH_PENDING: tuple[str, ...] = (
    "confirmed",
    "corrected",
    "pending",
)


def apply_review_filter(
    query: Select,
    *,
    include_pending: bool = False,
    column=CaseDiagnosis.review_status,
) -> Select:
    """Add `review_status IN (...)` to a SQLAlchemy Select."""
    statuses: Iterable[str] = (
        VISIBLE_REVIEW_STATUSES_WITH_PENDING if include_pending else VISIBLE_REVIEW_STATUSES
    )
    return query.where(column.in_(statuses))


def review_status_sql_in(*, include_pending: bool = False) -> str:
    """Return the literal `IN (...)` clause string for raw SQL.

    e.g. ``cd.review_status IN ('confirmed', 'corrected')``
    """
    statuses = (
        VISIBLE_REVIEW_STATUSES_WITH_PENDING if include_pending else VISIBLE_REVIEW_STATUSES
    )
    quoted = ", ".join(f"'{s}'" for s in statuses)
    return f"({quoted})"
