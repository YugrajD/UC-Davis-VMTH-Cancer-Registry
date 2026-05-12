"""Best-effort email notifications for role requests."""

import logging
import smtplib
import ssl
from email.mime.text import MIMEText

from app.config import settings

logger = logging.getLogger(__name__)


def send_role_request_email(
    requester_email: str,
    requested_role: str,
    admin_emails: list[str],
) -> None:
    """Send a notification email to admins about a new role request.

    If SMTP is not configured (empty host), logs a warning and returns.
    Failures are logged but never raised — email is best-effort.
    """
    if not settings.SMTP_HOST:
        logger.warning(
            "SMTP not configured — skipping role-request notification for %s requesting %s",
            requester_email,
            requested_role,
        )
        return

    if not admin_emails:
        logger.warning("No admin emails to notify for role request")
        return

    subject = f"New Role Request: {requested_role}"
    body = (
        f"{requester_email} has requested the {requested_role} role.\n\n"
        "Review it in the admin panel under User Management."
    )

    try:
        with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=30) as server:
            if settings.SMTP_USER and settings.SMTP_PASSWORD:
                ctx = ssl.create_default_context()
                server.starttls(context=ctx)
                server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)

            for admin_email in admin_emails:
                msg = MIMEText(body)
                msg["Subject"] = subject
                msg["From"] = settings.SMTP_FROM or settings.SMTP_USER or "noreply@example.com"
                msg["To"] = admin_email
                server.sendmail(msg["From"], [admin_email], msg.as_string())

        logger.info("Sent role-request notification to %d admin(s)", len(admin_emails))
    except Exception:
        logger.exception("Failed to send role-request email notification")
