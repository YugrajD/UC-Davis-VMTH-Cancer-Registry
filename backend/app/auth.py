"""JWT verification dependencies for Amazon Cognito.

The frontend forwards Cognito ID tokens (not access tokens) as the bearer
credential — ID tokens carry the `email` claim needed for role lookup and
`aud` = the app client ID; Cognito access tokens carry neither (identity
attributes live only on the ID token, and the access token's `username`
claim is the opaque `sub`, not the email).
"""

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import jwt
from jwt import PyJWKClient
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models.models import UserRole
from app.rate_limit import get_client_ip

logger = logging.getLogger(__name__)

bearer_scheme = HTTPBearer()
bearer_scheme_optional = HTTPBearer(auto_error=False)

# Cached JWKS client (fetches keys lazily and caches them)
_jwks_client: Optional[PyJWKClient] = None


def _get_jwks_client() -> PyJWKClient:
    global _jwks_client
    if _jwks_client is None:
        url = f"{settings.COGNITO_ISSUER_URL.rstrip('/')}/{settings.COGNITO_USER_POOL_ID}/.well-known/jwks.json"
        _jwks_client = PyJWKClient(url, timeout=10)
    return _jwks_client


# --- Auth failure rate limiting (in-memory) ---
_AUTH_WINDOW = 900          # 15 minutes in seconds
_AUTH_MAX_FAILURES = 5
# Hard cap on tracked IPs to prevent memory exhaustion from distributed
# brute-force with many source addresses.  When the cap is reached the
# oldest half of entries is evicted.
_AUTH_MAX_TRACKED_IPS = 10_000
# NOTE: this dict is per-process. With multiple uvicorn workers (or replicas)
# each worker tracks failures independently, so the effective threshold is
# _AUTH_MAX_FAILURES * num_workers before a lockout. Acceptable for current
# single-worker Cloud Run deployment; replace with Redis if scaling out.
_failed_attempts: dict[str, list[float]] = defaultdict(list)


def _evict_stale_entries() -> None:
    """Remove IPs with no recent failures and enforce the size cap."""
    now = time.time()
    cutoff = now - _AUTH_WINDOW
    # Remove entries whose timestamps are all expired.
    stale_keys = [k for k, v in _failed_attempts.items() if not v or v[-1] <= cutoff]
    for k in stale_keys:
        del _failed_attempts[k]
    # If still over the cap, drop the oldest half by earliest timestamp.
    if len(_failed_attempts) > _AUTH_MAX_TRACKED_IPS:
        sorted_keys = sorted(_failed_attempts, key=lambda k: _failed_attempts[k][0])
        for k in sorted_keys[: len(sorted_keys) // 2]:
            del _failed_attempts[k]


def _check_auth_rate_limit(request: Request) -> None:
    """Raise 429 if the IP has too many recent auth failures."""
    ip = get_client_ip(request)
    now = time.time()
    cutoff = now - _AUTH_WINDOW
    # Prune old timestamps for this IP.
    _failed_attempts[ip] = [t for t in _failed_attempts[ip] if t > cutoff]
    if len(_failed_attempts[ip]) >= _AUTH_MAX_FAILURES:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many failed authentication attempts. Try again later.",
        )


def _record_auth_failure(request: Request) -> None:
    """Record a failed auth attempt for the requesting IP."""
    ip = get_client_ip(request)
    _failed_attempts[ip].append(time.time())
    # Periodic eviction — only run when the dict grows large.
    if len(_failed_attempts) > _AUTH_MAX_TRACKED_IPS:
        _evict_stale_entries()


def _verify_token(token: str) -> dict:
    """Verify a Cognito ID token via JWKS.

    Only RS256 is accepted (Cognito's signing algorithm) — reject anything
    else to prevent algorithm-confusion attacks.
    """
    try:
        header = jwt.get_unverified_header(token)
    except jwt.DecodeError as e:
        raise jwt.InvalidTokenError(f"Cannot read token header: {e}")

    alg = header.get("alg", "")
    if alg != "RS256":
        raise jwt.InvalidTokenError(f"Unsupported algorithm: {alg}")

    jwks_client = _get_jwks_client()
    signing_key = jwks_client.get_signing_key_from_jwt(token)
    payload = jwt.decode(
        token, signing_key.key, algorithms=["RS256"], audience=settings.COGNITO_CLIENT_ID,
    )

    if payload.get("token_use") != "id":
        raise jwt.InvalidTokenError("Not an ID token")

    return payload


@dataclass
class CurrentUser:
    sub: str
    email: str
    is_admin: bool
    # Scoped roles (admins implicitly hold both).
    is_uploader: bool = False
    is_reviewer: bool = False


def _resolve_roles_from_env(email: str) -> tuple[bool, bool, bool]:
    """Fallback role lookup against the env-var allow lists."""
    is_admin = email in settings.admin_emails_list
    # Admins implicitly inherit lower-privilege roles.
    is_uploader = is_admin or email in settings.uploader_emails_list
    is_reviewer = is_admin or email in settings.reviewer_emails_list
    return is_admin, is_uploader, is_reviewer


async def _resolve_roles(db: AsyncSession, email: str) -> tuple[bool, bool, bool]:
    """Return (is_admin, is_uploader, is_reviewer) for a given email.

    The user_roles table is the source of truth. If no row exists, fall
    back to the env-var allow lists (for first-boot before the seed runs
    and for emails not yet inserted).
    """
    result = await db.execute(
        select(UserRole.is_admin, UserRole.is_uploader, UserRole.is_reviewer)
        .where(func.lower(UserRole.email) == email.lower())
    )
    row = result.one_or_none()
    if row is None:
        return _resolve_roles_from_env(email)
    is_admin, is_uploader, is_reviewer = row
    # Admins implicitly inherit lower-privilege roles even if the DB row
    # forgot to set them (defensive — UI normalizes this on write too).
    if is_admin:
        is_uploader = True
        is_reviewer = True
    return is_admin, is_uploader, is_reviewer


_VERIFY_TIMEOUT = 10  # seconds — fail fast rather than hang


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> CurrentUser:
    """Decode Cognito ID token and return the current user."""
    _check_auth_rate_limit(request)
    token = credentials.credentials
    try:
        payload = await asyncio.wait_for(
            asyncio.to_thread(_verify_token, token),
            timeout=_VERIFY_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.error("JWT verification timed out after %ds (JWKS fetch may be hanging)", _VERIFY_TIMEOUT)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication timed out — please try again",
        )
    except jwt.ExpiredSignatureError:
        _record_auth_failure(request)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
        )
    except jwt.InvalidTokenError:
        _record_auth_failure(request)
        logger.warning("JWT decode failed")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
        )

    email = payload.get("email", "")
    sub = payload.get("sub", "")

    if not email or not sub:
        _record_auth_failure(request)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing required claims",
        )

    is_admin, is_uploader, is_reviewer = await _resolve_roles(db, email)

    return CurrentUser(
        sub=sub,
        email=email,
        is_admin=is_admin,
        is_uploader=is_uploader,
        is_reviewer=is_reviewer,
    )


async def get_optional_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme_optional),
    db: AsyncSession = Depends(get_db),
) -> Optional[CurrentUser]:
    """Decode JWT if provided, otherwise return None."""
    if credentials is None:
        return None
    _check_auth_rate_limit(request)
    try:
        payload = await asyncio.wait_for(
            asyncio.to_thread(_verify_token, credentials.credentials),
            timeout=_VERIFY_TIMEOUT,
        )
    except (asyncio.TimeoutError, jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        _record_auth_failure(request)
        return None

    email = payload.get("email", "")
    sub = payload.get("sub", "")
    if not email or not sub:
        return None

    is_admin, is_uploader, is_reviewer = await _resolve_roles(db, email)
    return CurrentUser(
        sub=sub,
        email=email,
        is_admin=is_admin,
        is_uploader=is_uploader,
        is_reviewer=is_reviewer,
    )


async def require_admin(
    user: CurrentUser = Depends(get_current_user),
) -> CurrentUser:
    """Require the current user to be an admin."""
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return user


async def require_reviewer(
    user: CurrentUser = Depends(get_current_user),
) -> CurrentUser:
    """Require the current user to hold the reviewer or admin role."""
    if not user.is_reviewer:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Reviewer access required",
        )
    return user
