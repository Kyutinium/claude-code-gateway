# Phase 3: `/v1/responses` Codex Support

## Summary

Make the `/v1/responses` endpoint backend-aware so it dispatches to Claude or
Codex based on the `model` field, reusing the existing `BackendRegistry` and
`resolve_model()` infrastructure from Phase 1. Also tighten `previous_response_id`
validation to latest-only semantics with `409 Conflict` for stale IDs.

## Current State

The `/v1/responses` endpoint (`src/main.py:1395-1600`) currently:
- Hardcodes `claude_cli` for both streaming and non-streaming paths
- Hardcodes `validate_claude_code_auth()` for auth
- Does NOT use `resolve_model()`, `BackendRegistry`, or `_validate_backend_auth()`
- Allows stale `previous_response_id` (any past turn, not just latest)
- Does NOT acquire `session.lock` for concurrency safety
- Does NOT set `session.backend` or check backend mismatch
- Does NOT capture `provider_session_id` from Codex chunks

## Goals

1. Backend dispatch: route `/v1/responses` by `model` via `resolve_model()` + `BackendRegistry.get()`
2. Strict `previous_response_id`: only the latest turn's response ID is valid; stale → `409`
3. Backend mismatch guard: reject requests where `model` resolves to a different backend than the session
4. Per-session locking: acquire `session.lock` around state mutation and backend execution
5. Codex session continuity: capture `provider_session_id` (thread_id) and use `resume` for follow-ups
6. Success-only turn commit: increment `turn_counter` only after successful completion
7. Tests: cover all new behaviors and update existing tests affected by refactoring

## Non-Goals

- Changing the public response shape (`ResponseObject`, `OutputItem`, etc.)
- Adding new fields to `ResponseCreateRequest`
- Modifying `codex_cli.py`
- Making substantial changes to `streaming_utils.py` (a minor addition to propagate `assistant_text` in `stream_result` was needed — see File Changes)
- Adding Codex support to `/v1/messages` (remains Claude-only)
- Fixing pre-existing Codex thinking/reasoning block or tool_use SSE emission gaps (Phase 5 backlog)

## Deliberate Gateway Deviations from OpenAI Responses API

These are intentional deviations from the OpenAI Responses API spec, driven by
backend constraints. They must be documented in `README.md` and reflected in tests.

1. **`instructions` + `previous_response_id` rejection (pre-existing)**:
   The gateway currently returns `400` when both are provided. OpenAI's API allows
   this combination (prior instructions do not carry over). This is a known
   deviation because neither Claude SDK `resume` nor Codex `thread_id` resume
   supports changing the system prompt mid-session. Relaxing this is possible in
   the future but is out of Phase 3 scope.

2. **Strict latest-only `previous_response_id` (new in Phase 3)**:
   OpenAI's API may allow branching from earlier response IDs. This gateway
   enforces latest-only because both Claude SDK `resume` and Codex `thread_id`
   only continue from the current head state. Stale IDs return `409 Conflict`
   with the current latest response ID in the error message for client recovery.

## Design

### 1. Auth: Replace hardcoded Claude auth with backend-aware auth

**Before:**
```python
auth_valid, auth_info = validate_claude_code_auth()
```

**After:**
```python
resolved, backend = _resolve_and_get_backend(body.model)
_validate_backend_auth(resolved.backend)
```

Reuses the same helpers that `/v1/chat/completions` already uses.

### 2. Strict `previous_response_id` (latest-only)

**Before:** Accepts any past turn ≤ `session.turn_counter`.

**After:** Only accepts the *exact* latest turn. Stale but structurally valid IDs return `409 Conflict`.

**IMPORTANT**: This check MUST happen inside `session.lock` to prevent TOCTOU with
concurrent `turn_counter` increments.

```python
# Inside session.lock:
if turn != session.turn_counter:
    if turn < session.turn_counter:
        raise HTTPException(
            status_code=409,
            detail=f"Stale previous_response_id: only the latest response "
                   f"(resp_{session_id}_{session.turn_counter}) can be continued",
        )
    else:
        raise HTTPException(status_code=404, detail="Future turn ...")
```

Rationale: resumable backend threads (both Claude SDK `resume` and Codex `thread_id`)
continue from the current head state. Accepting old response IDs implies fork
semantics that don't exist.

**Note**: This is an intentional breaking change. The old behavior silently accepted
stale IDs but the SDK would continue from head regardless, making the behavior
misleading. The new behavior makes the error explicit.

### 3. Backend mismatch guard

When `previous_response_id` resolves to an existing session, validate that the
requested model's backend matches the session's backend:

```python
if session.backend != resolved.backend:
    raise HTTPException(status_code=400, detail="Backend mismatch ...")
```

**Note**: Uses `400 Bad Request` (not `409`) to match the existing convention in
`/v1/chat/completions` `_streaming_session_preflight()` at `src/main.py:682`.

### 4. Per-session locking

Wrap the entire request execution (both streaming and non-streaming) under
`session.lock` to prevent concurrent mutations.

**Streaming path**: Use a preflight function (`_responses_streaming_preflight()`)
that runs BEFORE `StreamingResponse` is constructed. This is critical because
once Starlette commits the 200 status line, `HTTPException` inside the generator
cannot produce proper HTTP error responses.

The preflight function:
1. Acquires `session.lock`
2. Validates stale-ID, backend mismatch, Codex resume guard (all inside lock)
3. Computes `next_turn = session.turn_counter + 1` (inside lock)
4. Tags `session.backend` on first turn (inside lock)
5. On validation failure, releases lock and raises `HTTPException`
6. Returns lock-held state to the generator

The generator's `finally` block releases the lock.

**Non-streaming path**: `async with session.lock` wrapping the entire execution.

**`is_new_session` semantics**: In `/v1/responses`, `is_new_session` is determined
by `previous_response_id is None` (not `len(session.messages) == 0` as in
`/v1/chat/completions`). This is safe because new sessions use a fresh UUID,
preventing concurrent targeting.

### 5. Backend dispatch

Replace direct `claude_cli.run_completion()` / `claude_cli.parse_message()` calls
with `backend.run_completion()` / `backend.parse_message()`:

```python
resolved, backend = _resolve_and_get_backend(body.model)

# Compute resume_id using the same fallback as /v1/chat/completions:
resume_id = session.provider_session_id or session_id

chunk_source = backend.run_completion(
    prompt=prompt,
    model=resolved.provider_model,
    system_prompt=system_prompt if is_new_session else None,
    permission_mode=PERMISSION_MODE_BYPASS,
    mcp_servers=get_mcp_servers() if resolved.backend == "claude" else None,
    session_id=session_id if is_new_session else None,
    resume=resume_id if not is_new_session else None,
)
```

**Critical**: `resume_id` MUST use `session.provider_session_id or session_id`.
- Claude: `provider_session_id` is typically `None`, falls back to `session_id`
  (which the SDK maps via `--session-id` on first turn)
- Codex: `provider_session_id` is the `thread_id` from the previous turn

**Token usage mapping**: When using `backend.estimate_token_usage()`, note that it
returns `prompt_tokens`/`completion_tokens` keys, but `ResponseUsage` uses
`input_tokens`/`output_tokens`. The existing mapping pattern must be preserved:
```python
ResponseUsage(input_tokens=prompt_tokens, output_tokens=completion_tokens)
```

### 6. Codex session continuity

After streaming/non-streaming completion, scan chunks for `codex_session` meta-event
using the existing `_capture_provider_session_id()` helper.

**Both paths must call `_capture_provider_session_id()` ALWAYS (even on failure)**:
- **Streaming**: After chunk collection, while still holding the lock, BEFORE
  the success check — not inside `if stream_result.get("success")`
- **Non-streaming**: After chunk collection, inside `async with session.lock`,
  BEFORE the success-only commit block

The Codex `thread_id` arrives via `thread.started` early in the turn, before
the turn may fail. Capturing it on failure allows the next attempt to resume.

The `codex_session` meta-event passes through `stream_response_chunks()` into
`chunks_buffer` unfiltered (verified: it falls to content path, returns None,
and is silently appended to buffer).

For Codex resume on follow-up turns:
- `resume = session.provider_session_id` (the Codex thread_id)
- If `provider_session_id` is missing on a Codex follow-up → `409 Conflict`

**Atomicity**: `_capture_provider_session_id` and `turn_counter` increment MUST
both happen under the same lock acquisition — never between release and re-acquire.

**Failure-path `provider_session_id` capture**: The Codex `thread_id` may arrive
(via `codex_session` meta-event) before a turn later fails. In this case,
`_capture_provider_session_id()` MUST still be called even on failure, so the
`thread_id` is retained internally. The `turn_counter` and messages are NOT
committed on failure, but `provider_session_id` IS written. However, because no
`response_id` is committed on failed turns, this captured `thread_id` is
internal-only — the client has no valid `previous_response_id` to reference the
failed turn, so it cannot directly recover through the `/v1/responses` API.
The internal capture ensures server-side consistency (the backend thread exists
regardless of whether the gateway considers the turn successful).

### 7. New session backend tagging

On first turn (new session), tag `session.backend = resolved.backend` immediately
after session creation, inside the lock, BEFORE any resume logic.

## File Changes

### Modified files

| File | Change |
|------|--------|
| `src/main.py` | Refactor `create_response()` for backend dispatch, locking, strict latest-only, Codex support; add `_responses_streaming_preflight()` |
| `tests/test_main_api_unit.py` | Update existing `/v1/responses` tests for BackendRegistry dispatch and new stale-ID semantics |
| `tests/conftest.py` | Add `BackendRegistry.register("claude", mock_cli)` to `client_context()`; promote `FakeCodexBackend` to shared fixture |
| `README.md` | Document deliberate gateway deviations: `instructions` + `previous_response_id` rejection, strict latest-only semantics, stale-ID 409 recovery, backend-mismatch/Codex-resume errors |

### Minor changes

| File | Change |
|------|--------|
| `src/streaming_utils.py` | Added `stream_result["assistant_text"] = final_text` in `stream_response_chunks()` so the caller can access the final assistant text without re-extracting it from chunks |

### No changes needed

| File | Why |
|------|-----|
| `src/codex_cli.py` | Already produces normalized chunks |
| `src/backend_registry.py` | Already has all needed infrastructure |
| `src/session_manager.py` | Already has `backend` and `provider_session_id` fields |
| `src/response_models.py` | Response shape unchanged; `model: str` accepts Codex models |

## Detailed Implementation Plan

### Step 1: Update test infrastructure
- Add `BackendRegistry.register("claude", mock_cli)` to `client_context()` in `tests/conftest.py`
- Promote `FakeCodexBackend` from `tests/integration/conftest.py` to `tests/conftest.py`
- This prevents 7+ existing tests from breaking when endpoint switches to BackendRegistry

### Step 2: Refactor `create_response()` auth and model resolution
- Replace `validate_claude_code_auth()` with `_resolve_and_get_backend()` + `_validate_backend_auth()`
- Store `resolved` and `backend` for later use

### Step 3: Add per-session locking and validation (non-streaming path)
- Acquire `session.lock` immediately after session lookup
- Inside lock: validate stale-ID (`turn != session.turn_counter`), backend mismatch
- Inside lock: compute `next_turn`, tag `session.backend` on first turn
- Inside lock: validate Codex resume capability (`provider_session_id` present)
- Inside lock: compute `resume_id = session.provider_session_id or session_id`
- Release in `finally` block

### Step 4: Add `_responses_streaming_preflight()` for streaming path
- Acquires `session.lock` BEFORE `StreamingResponse` creation
- Performs all validation (stale-ID, backend mismatch, Codex resume guard)
- Computes `next_turn` and `resume_id` inside lock
- Tags `session.backend` on first turn
- On failure: releases lock, raises HTTPException (proper HTTP status)
- On success: returns lock-held state dict to generator
- Generator's `finally` releases the lock

### Step 5: Replace `claude_cli` with `backend` dispatch
- Streaming: `backend.run_completion(...)` with backend-appropriate kwargs
- Non-streaming: same
- `backend.parse_message()` instead of `claude_cli.parse_message()`
- `backend.estimate_token_usage()` instead of `MessageAdapter.estimate_tokens()`
- Preserve `prompt_tokens` → `input_tokens` key mapping for `ResponseUsage`
- Error messages: use backend-agnostic wording (not "Claude SDK error")

### Step 6: Add Codex session continuity
- Call `_capture_provider_session_id(chunks_buffer, session)` in BOTH paths,
  ALWAYS (even on failure), because the backend thread exists regardless:
  - Streaming: after chunk collection, before success-only commit, under lock
  - Non-streaming: after chunk collection, before success-only commit, under lock
- Only commit `turn_counter` and messages on success (NOT `provider_session_id`)
- Codex follow-up: use `session.provider_session_id` as `resume`
- Missing `provider_session_id` on Codex follow-up → `409`

### Step 7: MCP servers — Claude-only
- Pass `mcp_servers=get_mcp_servers()` only when `resolved.backend == "claude"`

### Step 8: Document gateway deviations in README.md
- Add section documenting the two deliberate Responses API deviations
- Document stale-ID 409 recovery path (latest response ID in error message)
- Document backend-mismatch and Codex-resume error behavior

### Step 9: Tests (new)
- See Testing Strategy section below

## Execution Order

The actual execution order inside the lock for a follow-up request:

1. Acquire `session.lock`
2. Validate `turn == session.turn_counter` (stale/future check)
3. Validate `session.backend == resolved.backend` (mismatch check)
4. Validate Codex `provider_session_id` if backend is codex
5. Compute `resume_id = session.provider_session_id or session_id`
6. Compute `next_turn = session.turn_counter + 1`
7. Execute `backend.run_completion(...)`, collect chunks
8. **Always**: `_capture_provider_session_id(chunks, session)` — even on failure,
   because the backend thread exists regardless of turn success
9. On success only: `session.turn_counter = next_turn`
10. On success only: `session.add_messages(...)` — user + assistant messages
11. Release lock (in `finally`)

**Critical invariant**: Steps 9 and 10 (turn commit and message append) are
deferred until success. This differs from `/v1/chat/completions` preflight which
appends messages before execution. The `/v1/responses` endpoint must NOT adopt
that pattern — session state must remain clean on failure.

## Error Matrix

| Scenario | HTTP Status | Detail |
|----------|-------------|--------|
| Invalid `previous_response_id` format | `404` | "is invalid" |
| Session missing/expired | `404` | "not found or expired" |
| Future turn in `previous_response_id` | `404` | "references a future turn" |
| Stale `previous_response_id` (past turn) | `409` | "Stale previous_response_id: only the latest response (resp_X_N) can be continued" |
| Backend mismatch | `400` | "Session belongs to backend X, but model resolves to Y" |
| Codex resume without `provider_session_id` | `409` | "Cannot resume Codex session: no thread_id from previous turn" |
| Backend not enabled | `400` | "Codex backend is not enabled" |
| Backend auth failed | `503` | "backend authentication failed" |

## Testing Strategy

### Existing tests requiring modification

| Test | Change needed |
|------|--------------|
| `test_create_response_rejects_invalid_or_future_previous_response_ids` | Add assertion for stale turn → 409 (currently only tests future) |
| `test_create_response_returns_503_when_auth_is_invalid` | Change mock target from `validate_claude_code_auth` to `validate_backend_auth` |
| All `client_context()`-dependent `/v1/responses` tests (7+) | `client_context()` must register mock as Claude backend in BackendRegistry |

### New unit tests (mocked backends, no real binaries)

1. **test_responses_codex_streaming**: Codex model → backend dispatch → SSE stream
2. **test_responses_codex_non_streaming**: Codex model → non-streaming → ResponseObject
3. **test_responses_stale_previous_response_id**: past turn → 409 with latest ID in message
4. **test_responses_latest_previous_response_id**: current turn → success
5. **test_responses_backend_mismatch**: Claude session + codex model → 400
6. **test_responses_codex_resume_with_thread_id**: follow-up with captured thread_id
7. **test_responses_codex_resume_no_thread_id**: follow-up without thread_id → 409
8. **test_responses_new_session_backend_tagged**: first turn sets session.backend
9. **test_responses_claude_unchanged**: existing Claude behavior regression check
10. **test_responses_codex_non_streaming_error**: Codex error chunk → backend-agnostic 502
11. **test_responses_codex_streaming_error**: Codex mid-stream failure → error SSE event
12. **test_responses_mcp_servers_not_passed_to_codex**: verify `mcp_servers=None` for Codex
13. **test_responses_codex_token_estimation_fallback**: verify `backend.estimate_token_usage()` when SDK usage unavailable
14. **test_responses_codex_session_meta_event_captured**: verify `codex_session` flows through `stream_response_chunks` into `chunks_buffer`
15. **test_responses_concurrent_stale_id_race**: two requests with same latest `previous_response_id` → one succeeds, other gets 409 (proves lock serialization)
16. **test_responses_non_streaming_failure_no_commit**: non-streaming failure → `session.messages` and `turn_counter` unchanged
17. **test_responses_codex_failure_path_captures_thread_id**: Codex turn fails but `provider_session_id` is still captured for next attempt
18. **test_responses_streaming_success_commits_with_streamed_text**: streaming success path commits assistant text and increments turn_counter
19. **test_responses_codex_exception_after_session_event_captures_thread_id**: non-streaming Codex failure after `codex_session` meta-event still captures thread_id
20. **test_responses_streaming_exception_after_session_event_captures_thread_id**: streaming Codex failure after `codex_session` meta-event still captures thread_id
21. **test_responses_truly_concurrent_lock_serialization**: two truly concurrent follow-up requests prove per-session lock serialization (one succeeds, one gets 409)
22. **test_responses_streaming_immediate_exception_before_any_yield**: backend raises in `run_completion()` before yielding any chunks — SSE stream returns `response.failed`, turn_counter unchanged
23. **test_responses_streaming_sync_raise_before_async_iteration**: `run_completion()` raises synchronously (not async generator) before any iteration — outer exception handler emits `response.failed`, turn_counter unchanged

## Known Pre-existing Issues (Not Phase 3 Scope)

- Codex thinking/reasoning blocks rendered as `<think>...</think>` text in SSE (Phase 5)
- Codex tool_use blocks silently filtered by `_filter_tool_blocks`, not emitted as structured SSE events (Phase 5)
- ~~Outer streaming exception handler emits `"error"` SSE event type instead of `response.failed`~~ — fixed in Phase 3 Round 6: outer handler now emits `response.failed` with a proper `ResponseObject`
