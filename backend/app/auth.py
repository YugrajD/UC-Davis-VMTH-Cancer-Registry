"""JWT verification dependencies for Supabase Auth.

Supports both HS256 (shared secret) and ES256 (JWKS) depending on the
algorithm in the token header. Supabase newer projects use ES256.
"""

import base64
import logging
from dataclasses import dataclass
from typing import Optional

import httpx
import jwt
from jwt import PyJWKClient
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import settings

logger = logging.getLogger(__name__)

bearer_scheme = HTTPBearer()
bearer_scheme_optional = HTTPBearer(auto_error=False)

# Cached JWKS client (fetches keys lazily and caches them)
_jwks_client: Optional[PyJWKClient] = None


def _get_jwks_client() -> Optional[PyJWKClient]:
    global _jwks_client
    if _jwks_client is not None:
        return _jwks_client
    if settings.SUPABASE_URL:
        url = f"{settings.SUPABASE_URL.rstrip('/')}/auth/v1/.well-known/jwks.json"
        _jwks_client = PyJWKClient(url)
        return _jwks_client
    return None


def _decode_hs256_secret(raw: str) -> bytes:
    """Base64-decode the Supabase JWT secret, falling back to raw bytes."""
    try:
        return base64.b64decode(raw)
    except Exception:
        return raw.encode()


def _verify_token(token: str) -> dict:
    """Verify a Supabase JWT, auto-detecting HS256 vs ES256."""
    # Peek at the header to determine algorithm
    try:
        header = jwt.get_unverified_header(token)
    except jwt.DecodeError as e:
        raise jwt.InvalidTokenError(f"Cannot read token header: {e}")

    alg = header.get("alg", "")

    if alg == "HS256":
        secret = _decode_hs256_secret(settings.SUPABASE_JWT_SECRET)
        return jwt.decode(
            token, secret, algorithms=["HS256"], audience="authenticated",
        )

    # ES256 / asymmetric — use JWKS
    jwks_client = _get_jwks_client()
    if jwks_client is None:
        raise jwt.InvalidTokenError(
            f"Token uses {alg} but SUPABASE_URL is not configured for JWKS"
        )

    signing_key = jwks_client.get_signing_key_from_jwt(token)
    return jwt.decode(
        token,
        signing_key.key,
        algorithms=[alg],
        audience="authenticated",
    )


@dataclass
class CurrentUser:
    sub: str
    email: str
    is_admin: bool
    # Scoped roles (admins implicitly hold both).
    is_uploader: bool = False
    is_reviewer: bool = False


def _resolve_roles(email: str) -> tuple[bool, bool, bool]:
    """Return (is_admin, is_uploader, is_reviewer) for a given email."""
    is_admin = email in settings.admin_emails_list
    # Admins implicitly inherit lower-privilege roles.
    is_uploader = is_admin or email in settings.uploader_emails_list
    is_reviewer = is_admin or email in settings.reviewer_emails_list
    return is_admin, is_uploader, is_reviewer


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> CurrentUser:
    """Decode Supabase JWT and return the current user."""
    token = credentials.credentials
    try:
        payload = _verify_token(token)
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
        )
    except jwt.InvalidTokenError as e:
        logger.warning("JWT decode failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {e}",
        )

    email = payload.get("email", "")
    sub = payload.get("sub", "")

    if not email or not sub:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing required claims",
        )

    is_admin, is_uploader, is_reviewer = _resolve_roles(email)

    return CurrentUser(
        sub=sub,
        email=email,
        is_admin=is_admin,
        is_uploader=is_uploader,
        is_reviewer=is_reviewer,
    )


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme_optional),
) -> Optional[CurrentUser]:
    """Decode JWT if provided, otherwise return None."""
    if credentials is None:
        return None
    try:
        payload = _verify_token(credentials.credentials)
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None

    email = payload.get("email", "")
    sub = payload.get("sub", "")
    if not email or not sub:
        return None

    is_admin, is_uploader, is_reviewer = _resolve_roles(email)
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
