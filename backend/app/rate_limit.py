"""Global rate-limiter instance shared across the application."""

import base64
import json
import logging

from fastapi import Request
from slowapi import Limiter

from app.config import settings

logger = logging.getLogger(__name__)


def get_client_ip(request: Request) -> str:
    """Return the real client IP, only trusting proxy headers from known IPs.

    When deployed behind a reverse proxy (Nginx, Cloudflare, etc.) the TCP
    peer address is the proxy's IP, not the end user's.  We read
    ``X-Forwarded-For`` only when the peer is listed in
    ``FORWARDED_ALLOW_IPS``; otherwise we use the raw TCP peer address.

    This prevents attackers from rotating ``X-Forwarded-For`` values to
    bypass rate limits when the app is exposed directly (no trusted proxy).
    """
    peer_ip = request.client.host if request.client else "127.0.0.1"

    trusted = settings.forwarded_allow_ips_set
    if trusted and peer_ip in trusted:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            # Leftmost entry is the original client.
            return forwarded.split(",")[0].strip()

    return peer_ip


def _extract_sub_from_token(token: str) -> str | None:
    """Extract the 'sub' claim from a JWT without any crypto operations.

    We only use this for rate-limit bucket keying — not for authentication.
    We parse the payload segment directly rather than calling jwt.decode()
    with verify_signature=False, which invokes the PyJWT machinery and
    could surface timing differences on malformed tokens.

    A crafted token that passes this function still gets rejected by the
    full signature verification in get_current_user() later in the chain.
    """
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        # Add padding so base64 doesn't raise on odd-length segments.
        payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)
        payload_bytes = base64.urlsafe_b64decode(payload_b64)
        payload = json.loads(payload_bytes)
        sub = payload.get("sub")
        return str(sub) if sub else None
    except Exception:
        return None


def get_rate_limit_key(request: Request) -> str:
    """Return user sub (from JWT) if authenticated, otherwise client IP.

    This lets authenticated users have their own per-user bucket,
    while anonymous users share a per-IP bucket.
    """
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        sub = _extract_sub_from_token(token)
        if sub:
            return f"user:{sub}"
    return get_client_ip(request)


limiter = Limiter(
    key_func=get_rate_limit_key,
    default_limits=[settings.RATE_LIMIT_ANONYMOUS],
)
