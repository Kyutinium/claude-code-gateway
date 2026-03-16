"""Tests for Open WebUI session mapping (CD_OPENWEBUI)."""

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

import src.main as main
from src.backend_registry import BackendRegistry
from src.constants import DEFAULT_MODEL


@contextmanager
def client_context(cd_openwebui: bool = False):
    """Create a TestClient with optional CD_OPENWEBUI enabled."""
    mock_cli = MagicMock()
    mock_cli.verify_cli = AsyncMock(return_value=True)
    mock_cli.verify = AsyncMock(return_value=True)
    from src.backends.claude.client import ClaudeCodeCLI

    mock_cli.build_options = ClaudeCodeCLI.build_options.__get__(mock_cli, type(mock_cli))

    if main.limiter and hasattr(main.limiter, "_storage"):
        main.limiter._storage.reset()

    def _mock_discover():
        from tests.conftest import register_all_descriptors

        register_all_descriptors()
        BackendRegistry.register("claude", mock_cli)

    patches = [
        patch.object(main, "discover_backends", _mock_discover),
        patch.object(main, "verify_api_key", new=AsyncMock(return_value=True)),
        patch.object(main, "validate_claude_code_auth", return_value=(True, {"method": "test"})),
        patch.object(main, "_validate_backend_auth"),
        patch.object(main.session_manager, "start_cleanup_task"),
        patch.object(main.session_manager, "async_shutdown", new=AsyncMock()),
        patch.object(main, "CD_OPENWEBUI", cd_openwebui),
    ]

    for p in patches:
        p.start()

    try:
        with TestClient(main.app) as client:
            yield client, mock_cli
    finally:
        for p in patches:
            p.stop()

    if main.limiter and hasattr(main.limiter, "_storage"):
        main.limiter._storage.reset()


def _chat_body(session_id=None):
    body = {
        "model": DEFAULT_MODEL,
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }
    if session_id is not None:
        body["session_id"] = session_id
    return body


def test_openwebui_chat_id_mapped_to_session_id_when_enabled():
    """X-OpenWebUI-Chat-Id header becomes session_id when CD_OPENWEBUI=true."""
    chat_id = "owui-chat-abc123"

    with client_context(cd_openwebui=True) as (client, mock_cli):
        # Mock run_completion to return a simple response
        async def fake_run(**kwargs):
            yield {"type": "assistant", "content": [{"type": "text", "text": "hi"}]}
            yield {
                "type": "result",
                "subtype": "success",
                "result": "hi",
                "usage": {
                    "input_tokens": 5,
                    "output_tokens": 2,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                },
            }

        mock_cli.run_completion = MagicMock(side_effect=fake_run)
        mock_cli.parse_message = MagicMock(return_value="hi")
        mock_cli.estimate_token_usage = MagicMock(
            return_value={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
        )

        resp = client.post(
            "/v1/chat/completions",
            json=_chat_body(),
            headers={"X-OpenWebUI-Chat-Id": chat_id},
        )

    assert resp.status_code == 200
    # The session should have been created with the Open WebUI chat_id
    call_kwargs = mock_cli.run_completion.call_args
    assert call_kwargs is not None
    # session_id should be the Open WebUI chat ID
    assert call_kwargs.kwargs.get("session_id") == chat_id


def test_openwebui_chat_id_ignored_when_disabled():
    """X-OpenWebUI-Chat-Id header is ignored when CD_OPENWEBUI is not set."""
    chat_id = "owui-chat-abc123"

    with client_context(cd_openwebui=False) as (client, mock_cli):
        async def fake_run(**kwargs):
            yield {"type": "assistant", "content": [{"type": "text", "text": "hi"}]}
            yield {
                "type": "result",
                "subtype": "success",
                "result": "hi",
                "usage": {
                    "input_tokens": 5,
                    "output_tokens": 2,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                },
            }

        mock_cli.run_completion = MagicMock(side_effect=fake_run)
        mock_cli.parse_message = MagicMock(return_value="hi")
        mock_cli.estimate_token_usage = MagicMock(
            return_value={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
        )

        resp = client.post(
            "/v1/chat/completions",
            json=_chat_body(),
            headers={"X-OpenWebUI-Chat-Id": chat_id},
        )

    assert resp.status_code == 200
    # Without CD_OPENWEBUI, no session_id should be set (stateless mode)
    call_kwargs = mock_cli.run_completion.call_args
    assert call_kwargs is not None
    # No session_id in stateless mode
    assert call_kwargs.kwargs.get("session_id") is None


def test_explicit_session_id_not_overridden_by_openwebui_header():
    """An explicit session_id in the body takes precedence over the header."""
    explicit_id = "my-explicit-session"
    chat_id = "owui-chat-abc123"

    with client_context(cd_openwebui=True) as (client, mock_cli):
        async def fake_run(**kwargs):
            yield {"type": "assistant", "content": [{"type": "text", "text": "hi"}]}
            yield {
                "type": "result",
                "subtype": "success",
                "result": "hi",
                "usage": {
                    "input_tokens": 5,
                    "output_tokens": 2,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                },
            }

        mock_cli.run_completion = MagicMock(side_effect=fake_run)
        mock_cli.parse_message = MagicMock(return_value="hi")
        mock_cli.estimate_token_usage = MagicMock(
            return_value={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
        )

        resp = client.post(
            "/v1/chat/completions",
            json=_chat_body(session_id=explicit_id),
            headers={"X-OpenWebUI-Chat-Id": chat_id},
        )

    assert resp.status_code == 200
    call_kwargs = mock_cli.run_completion.call_args
    assert call_kwargs is not None
    # Explicit session_id should win
    assert call_kwargs.kwargs.get("session_id") == explicit_id


def test_openwebui_no_header_stays_stateless():
    """Without X-OpenWebUI-Chat-Id header, requests stay stateless even with CD_OPENWEBUI."""
    with client_context(cd_openwebui=True) as (client, mock_cli):
        async def fake_run(**kwargs):
            yield {"type": "assistant", "content": [{"type": "text", "text": "hi"}]}
            yield {
                "type": "result",
                "subtype": "success",
                "result": "hi",
                "usage": {
                    "input_tokens": 5,
                    "output_tokens": 2,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                },
            }

        mock_cli.run_completion = MagicMock(side_effect=fake_run)
        mock_cli.parse_message = MagicMock(return_value="hi")
        mock_cli.estimate_token_usage = MagicMock(
            return_value={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
        )

        resp = client.post(
            "/v1/chat/completions",
            json=_chat_body(),
        )

    assert resp.status_code == 200
    call_kwargs = mock_cli.run_completion.call_args
    assert call_kwargs is not None
    assert call_kwargs.kwargs.get("session_id") is None
