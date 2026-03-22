"""Auth router — returns current user info."""

from fastapi import APIRouter, Depends

from app.auth import CurrentUser, get_current_user

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


@router.get("/me")
async def get_me(user: CurrentUser = Depends(get_current_user)):
    """Return current user's email and admin status."""
    return {
        "email": user.email,
        "is_admin": user.is_admin,
    }
