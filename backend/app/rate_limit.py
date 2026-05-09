"""Global rate-limiter instance shared across the application."""

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


limiter = Limiter(key_func=get_client_ip, default_limits=["60/minute"])
