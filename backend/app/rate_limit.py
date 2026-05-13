"""Global rate-limiter instance shared across the application."""

import jwt
from fastapi import Request
from slowapi import Limiter

from app.config import settings


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


def get_rate_limit_key(request: Request) -> str:
    """Return user sub (from JWT) if authenticated, otherwise client IP.

    This lets authenticated users have their own per-user bucket,
    while anonymous users share a per-IP bucket.
    """
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        try:
            # Decode without verification — we only need the 'sub' claim
            # for rate-limit keying. Auth verification happens later in
            # the dependency chain.
            payload = jwt.decode(token, options={"verify_signature": False})
            sub = payload.get("sub")
            if sub:
                return f"user:{sub}"
        except Exception:
            pass
    return get_client_ip(request)


limiter = Limiter(
    key_func=get_rate_limit_key,
    default_limits=[settings.RATE_LIMIT_ANONYMOUS],
)
