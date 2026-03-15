"""Admin authentication — separate from the main API key auth.

Admin endpoints require a dedicated ``ADMIN_API_KEY`` and are gated behind
``ADMIN_UI_ENABLED``.  Authentication uses a short-lived HttpOnly cookie
(``admin_session``) scoped to ``/admin`` with ``SameSite=Strict``.

Flow:
1. ``POST /admin/api/login`` — validates ADMIN_API_KEY, sets cookie
2. All ``/admin/api/*`` routes — checked by ``require_admin`` dependency
3. ``POST /admin/api/logout`` — clears cookie
"""

import hashlib
import hmac
import logging
import os
import secrets
import time
from fastapi import HTTPException, Request, Response

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ADMIN_UI_ENABLED = os.getenv("ADMIN_UI_ENABLED", "false").lower() in ("true", "1", "yes", "on")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")
ADMIN_SESSION_TTL = int(os.getenv("ADMIN_SESSION_TTL", "3600"))  # 1 hour default

# HMAC key for signing session cookies — derived once at import time.
# Using a random key means sessions don't survive server restarts,
# which is acceptable for an admin panel.
_COOKIE_SECRET = secrets.token_bytes(32)
_COOKIE_NAME = "admin_session"

# ---------------------------------------------------------------------------
# Session token helpers
# ---------------------------------------------------------------------------


def _make_session_token(issued_at: int) -> str:
    """Create an HMAC-signed session token encoding the issue timestamp."""
    payload = f"{issued_at}".encode()
    sig = hmac.new(_COOKIE_SECRET, payload, hashlib.sha256).hexdigest()
    return f"{issued_at}.{sig}"


def _verify_session_token(token: str) -> bool:
    """Verify an admin session token's signature and TTL."""
    if not token or "." not in token:
        return False
    parts = token.split(".", 1)
    if len(parts) != 2:
        return False
    try:
        issued_at = int(parts[0])
    except ValueError:
        return False

    # Check TTL
    if time.time() - issued_at > ADMIN_SESSION_TTL:
        return False

    # Verify HMAC (timing-safe)
    expected = _make_session_token(issued_at)
    return hmac.compare_digest(token, expected)


# ---------------------------------------------------------------------------
# Login / logout helpers
# ---------------------------------------------------------------------------


def check_admin_enabled() -> None:
    """Raise 404 if admin UI is disabled (stealth — don't reveal admin exists)."""
    if not ADMIN_UI_ENABLED:
        raise HTTPException(status_code=404, detail="Not found")


def check_admin_configured() -> None:
    """Raise 503 if ADMIN_API_KEY is not configured."""
    if not ADMIN_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Admin API key not configured. Set ADMIN_API_KEY environment variable.",
        )


def login(provided_key: str, response: Response) -> dict:
    """Validate the admin key and set an HttpOnly session cookie."""
    check_admin_enabled()
    check_admin_configured()

    if not hmac.compare_digest(provided_key, ADMIN_API_KEY):
        logger.warning("Admin login failed: invalid API key")
        raise HTTPException(status_code=401, detail="Invalid admin API key")

    token = _make_session_token(int(time.time()))
    response.set_cookie(
        key=_COOKIE_NAME,
        value=token,
        httponly=True,
        samesite="strict",
        path="/admin",
        max_age=ADMIN_SESSION_TTL,
        secure=os.getenv("ADMIN_COOKIE_SECURE", "false").lower() in ("true", "1"),
    )
    logger.info("Admin login successful")
    return {"status": "ok", "ttl": ADMIN_SESSION_TTL}


def logout(response: Response) -> dict:
    """Clear the admin session cookie."""
    response.delete_cookie(key=_COOKIE_NAME, path="/admin", samesite="strict")
    return {"status": "logged_out"}


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------


def require_admin(request: Request) -> bool:
    """FastAPI dependency that enforces admin authentication.

    Checks:
    1. ``ADMIN_UI_ENABLED`` must be true
    2. ``ADMIN_API_KEY`` must be configured
    3. Valid session cookie OR valid Bearer token in Authorization header

    Bearer token fallback allows programmatic/curl access without cookies.
    """
    check_admin_enabled()
    check_admin_configured()

    # Try cookie first
    token = request.cookies.get(_COOKIE_NAME)
    if token and _verify_session_token(token):
        return True

    # Fallback: Bearer token in Authorization header
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        bearer_token = auth_header[7:]
        if hmac.compare_digest(bearer_token, ADMIN_API_KEY):
            return True

    raise HTTPException(
        status_code=401,
        detail="Admin authentication required",
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_admin_status() -> dict:
    """Return admin UI status for diagnostics (no secrets)."""
    return {
        "enabled": ADMIN_UI_ENABLED,
        "configured": bool(ADMIN_API_KEY),
        "session_ttl": ADMIN_SESSION_TTL,
    }
