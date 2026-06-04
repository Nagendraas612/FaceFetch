"""
auth.py
Google OAuth 2.0 via Authlib + Starlette sessions.
Handles login, callback, logout, and session management.
"""

import os
import logging
from typing import Optional

from authlib.integrations.starlette_client import OAuth, OAuthError
from starlette.requests import Request
from starlette.responses import RedirectResponse, JSONResponse
from fastapi import APIRouter, HTTPException
from database import get_db

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

GOOGLE_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")

if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
    raise EnvironmentError(
        "GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET must be set in .env"
    )

# ── OAuth client ──────────────────────────────────────────────────────────────
oauth = OAuth()
oauth.register(
    name="google",
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    authorize_url="https://accounts.google.com/o/oauth2/v2/auth",
    access_token_url="https://oauth2.googleapis.com/token",
    client_kwargs={
        "scope": (
            "openid email profile "
            "https://www.googleapis.com/auth/drive.readonly"
        ),
        "prompt": "select_account consent",
        "access_type": "offline",
    },
)

router = APIRouter(prefix="/auth", tags=["auth"])


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_current_user(request: Request) -> Optional[dict]:
    """Get user info from session, or None if not logged in."""
    return request.session.get("user")


def require_user(request: Request) -> dict:
    """Get user info or raise 401."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


def _build_redirect_uri(request: Request) -> str:
    """
    Build the OAuth callback URI, forcing HTTPS in production.
    Handles reverse proxies that strip the scheme.
    """
    root_url = str(request.base_url).rstrip("/")
    redirect_uri = f"{root_url}/auth/callback"

    # Force HTTPS when not running locally
    if "localhost" not in redirect_uri and "127.0.0.1" not in redirect_uri:
        redirect_uri = redirect_uri.replace("http://", "https://")

    return redirect_uri


# ── Routes ────────────────────────────────────────────────────────────────────
@router.get("/login")
async def login(request: Request):
    """Redirect the browser to Google's consent screen."""
    redirect_uri = _build_redirect_uri(request)
    logger.info("OAuth redirect URI: %s", redirect_uri)
    return await oauth.google.authorize_redirect(request, redirect_uri)


@router.get("/callback")
async def callback(request: Request):
    """Handle the OAuth redirect; store user info and the Refresh Token."""
    try:
        token = await oauth.google.authorize_access_token(request)
    except OAuthError as exc:
        logger.error("OAuth error: %s", exc)
        # Redirect to home with error instead of raw JSON
        return RedirectResponse(url="/?auth_error=oauth_failed")

    user_info = token.get("userinfo") or await oauth.google.userinfo(token=token)

    if not user_info or "sub" not in user_info:
        logger.error("OAuth callback: missing user info")
        return RedirectResponse(url="/?auth_error=no_user_info")

    user_id = user_info["sub"]

    # Store Refresh Token in MongoDB for persistent auth
    refresh_token = token.get("refresh_token")
    if refresh_token:
        try:
            db = get_db()
            await db["user_profiles"].update_one(
                {"user_id": user_id},
                {"$set": {"refresh_token": refresh_token}},
                upsert=True,
            )
            logger.info("Refresh token saved for user: %s", user_id)
        except Exception as e:
            logger.error("Failed to save refresh token: %s", e)

    # Store session info
    request.session["user"] = {
        "sub":     user_id,
        "email":   user_info.get("email", ""),
        "name":    user_info.get("name", ""),
        "picture": user_info.get("picture", ""),
    }

    request.session["drive_token"] = token.get("access_token", "")

    logger.info("User authenticated: %s", user_info.get("email", user_id))
    return RedirectResponse(url="/")


@router.get("/logout")
async def logout(request: Request):
    """Clear the session and redirect to home."""
    request.session.clear()
    return RedirectResponse(url="/")


@router.get("/me")
async def me(request: Request):
    """Return current user info or {authenticated: false}."""
    user = get_current_user(request)
    if not user:
        return JSONResponse({"authenticated": False})
    return JSONResponse({"authenticated": True, "user": user})
