"""Unit tests for admin_auth — login, cookie sessions, and require_admin."""

import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from src.admin_auth import (
    _COOKIE_NAME,
    _make_session_token,
    _verify_session_token,
    check_admin_configured,
    check_admin_enabled,
    login,
    logout,
    require_admin,
)


# ---------------------------------------------------------------------------
# Feature gate
# ---------------------------------------------------------------------------


class TestAdminEnabled:
    def test_disabled_by_default(self):
        with patch("src.admin_auth.ADMIN_UI_ENABLED", False):
            with pytest.raises(HTTPException) as exc_info:
                check_admin_enabled()
            assert exc_info.value.status_code == 404

    def test_enabled(self):
        with patch("src.admin_auth.ADMIN_UI_ENABLED", True):
            check_admin_enabled()  # should not raise


class TestAdminConfigured:
    def test_not_configured(self):
        with patch("src.admin_auth.ADMIN_API_KEY", ""):
            with pytest.raises(HTTPException) as exc_info:
                check_admin_configured()
            assert exc_info.value.status_code == 503

    def test_configured(self):
        with patch("src.admin_auth.ADMIN_API_KEY", "test-key"):
            check_admin_configured()  # should not raise


# ---------------------------------------------------------------------------
# Session tokens
# ---------------------------------------------------------------------------


class TestSessionTokens:
    def test_valid_token(self):
        token = _make_session_token(int(time.time()))
        assert _verify_session_token(token)

    def test_expired_token(self):
        old_time = int(time.time()) - 7200  # 2 hours ago
        token = _make_session_token(old_time)
        with patch("src.admin_auth.ADMIN_SESSION_TTL", 3600):
            assert not _verify_session_token(token)

    def test_tampered_token(self):
        token = _make_session_token(int(time.time()))
        tampered = token[:-1] + ("a" if token[-1] != "a" else "b")
        assert not _verify_session_token(tampered)

    def test_empty_token(self):
        assert not _verify_session_token("")
        assert not _verify_session_token("nodot")


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------


class TestLogin:
    @patch("src.admin_auth.ADMIN_UI_ENABLED", True)
    @patch("src.admin_auth.ADMIN_API_KEY", "correct-key")
    def test_successful_login(self):
        response = MagicMock()
        result = login("correct-key", response)
        assert result["status"] == "ok"
        response.set_cookie.assert_called_once()
        call_kwargs = response.set_cookie.call_args
        assert call_kwargs.kwargs.get("httponly") or call_kwargs[1].get("httponly")

    @patch("src.admin_auth.ADMIN_UI_ENABLED", True)
    @patch("src.admin_auth.ADMIN_API_KEY", "correct-key")
    def test_wrong_key(self):
        response = MagicMock()
        with pytest.raises(HTTPException) as exc_info:
            login("wrong-key", response)
        assert exc_info.value.status_code == 401

    @patch("src.admin_auth.ADMIN_UI_ENABLED", False)
    def test_login_when_disabled(self):
        response = MagicMock()
        with pytest.raises(HTTPException) as exc_info:
            login("any-key", response)
        assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# Logout
# ---------------------------------------------------------------------------


class TestLogout:
    def test_clears_cookie(self):
        response = MagicMock()
        result = logout(response)
        assert result["status"] == "logged_out"
        response.delete_cookie.assert_called_once()


# ---------------------------------------------------------------------------
# require_admin dependency
# ---------------------------------------------------------------------------


class TestRequireAdmin:
    @patch("src.admin_auth.ADMIN_UI_ENABLED", True)
    @patch("src.admin_auth.ADMIN_API_KEY", "test-key")
    def test_valid_cookie(self):
        token = _make_session_token(int(time.time()))
        request = MagicMock()
        request.cookies = {_COOKIE_NAME: token}
        request.headers = {}
        assert require_admin(request) is True

    @patch("src.admin_auth.ADMIN_UI_ENABLED", True)
    @patch("src.admin_auth.ADMIN_API_KEY", "test-key")
    def test_valid_bearer(self):
        request = MagicMock()
        request.cookies = {}
        request.headers = {"authorization": "Bearer test-key"}
        assert require_admin(request) is True

    @patch("src.admin_auth.ADMIN_UI_ENABLED", True)
    @patch("src.admin_auth.ADMIN_API_KEY", "test-key")
    def test_no_auth(self):
        request = MagicMock()
        request.cookies = {}
        request.headers = {}
        with pytest.raises(HTTPException) as exc_info:
            require_admin(request)
        assert exc_info.value.status_code == 401

    @patch("src.admin_auth.ADMIN_UI_ENABLED", True)
    @patch("src.admin_auth.ADMIN_API_KEY", "test-key")
    def test_wrong_bearer(self):
        request = MagicMock()
        request.cookies = {}
        request.headers = {"authorization": "Bearer wrong-key"}
        with pytest.raises(HTTPException) as exc_info:
            require_admin(request)
        assert exc_info.value.status_code == 401

    @patch("src.admin_auth.ADMIN_UI_ENABLED", False)
    def test_disabled(self):
        request = MagicMock()
        with pytest.raises(HTTPException) as exc_info:
            require_admin(request)
        assert exc_info.value.status_code == 404

    @patch("src.admin_auth.ADMIN_UI_ENABLED", True)
    @patch("src.admin_auth.ADMIN_API_KEY", "test-key")
    def test_expired_cookie_with_valid_bearer(self):
        old_token = _make_session_token(int(time.time()) - 99999)
        request = MagicMock()
        request.cookies = {_COOKIE_NAME: old_token}
        request.headers = {"authorization": "Bearer test-key"}
        assert require_admin(request) is True
