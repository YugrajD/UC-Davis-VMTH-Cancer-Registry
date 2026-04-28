"""Seed user_roles from env vars on startup.

Reads ADMIN_EMAILS / UPLOADER_EMAILS / REVIEWER_EMAILS, unions them by
email, and inserts a row for any email not already present in
user_roles. Existing rows are NEVER overwritten — once a user is
managed via the admin UI, the env var becomes a no-op for them.
"""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.models import UserRole


async def seed_user_roles_from_env(db: AsyncSession) -> int:
    """Insert user_roles rows for any env-listed email not yet in the table.

    Returns the number of rows inserted.
    """
    admins = {e.lower() for e in settings.admin_emails_list}
    uploaders = {e.lower() for e in settings.uploader_emails_list}
    reviewers = {e.lower() for e in settings.reviewer_emails_list}

    desired = {}  # email -> (is_admin, is_uploader, is_reviewer)
    for email in admins | uploaders | reviewers:
        is_admin = email in admins
        # Admins implicitly hold lower-privilege roles.
        is_uploader = is_admin or email in uploaders
        is_reviewer = is_admin or email in reviewers
        desired[email] = (is_admin, is_uploader, is_reviewer)

    if not desired:
        return 0

    existing_rows = await db.execute(
        select(UserRole.email).where(UserRole.email.in_(desired.keys()))
    )
    existing = {row[0].lower() for row in existing_rows.all()}

    inserted = 0
    for email, (is_admin, is_uploader, is_reviewer) in desired.items():
        if email in existing:
            continue
        db.add(UserRole(
            email=email,
            is_admin=is_admin,
            is_uploader=is_uploader,
            is_reviewer=is_reviewer,
            updated_by_email="__env_seed__",
        ))
        inserted += 1
    return inserted
