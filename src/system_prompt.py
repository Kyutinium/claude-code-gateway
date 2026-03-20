"""Custom system prompt management.

Provides a thread-safe store for the global base system prompt.
Completely separate from ``RuntimeConfig`` to avoid logging prompt
content and to support large text values cleanly.

Priority: runtime override > file default > None (preset mode).
"""

import logging
from pathlib import Path
from threading import Lock
from typing import Optional

logger = logging.getLogger(__name__)

_lock = Lock()
_default_prompt: Optional[str] = None  # loaded from file at startup
_runtime_prompt: Optional[str] = None  # admin override (in-memory)


def load_default_prompt(file_path: str = "") -> None:
    """Load the default system prompt from *file_path*.

    * If *file_path* is empty/blank, preset mode is used (no custom prompt).
    * If the file does not exist, ``FileNotFoundError`` is raised (fail-fast).
    """
    global _default_prompt

    if not file_path or not file_path.strip():
        _default_prompt = None
        logger.info("System prompt: using claude_code preset (no file configured)")
        return

    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"SYSTEM_PROMPT_FILE not found: {file_path}")

    content = path.read_text(encoding="utf-8").strip()
    if not content:
        _default_prompt = None
        logger.warning("System prompt file is empty, falling back to preset mode")
        return

    _default_prompt = content
    logger.info("System prompt: loaded from file (%d chars)", len(content))


def get_system_prompt() -> Optional[str]:
    """Return the active base system prompt.

    Returns ``None`` when in preset mode (no custom prompt active).
    """
    with _lock:
        if _runtime_prompt is not None:
            return _runtime_prompt
    return _default_prompt


def get_default_prompt() -> Optional[str]:
    """Return the file-loaded default prompt (ignoring runtime override)."""
    return _default_prompt


def set_system_prompt(text: str) -> None:
    """Set a runtime override for the system prompt.

    Raises ``ValueError`` if *text* is empty or whitespace-only.
    """
    global _runtime_prompt
    stripped = text.strip()
    if not stripped:
        raise ValueError("System prompt cannot be empty. Use reset to revert to default.")
    with _lock:
        _runtime_prompt = stripped
    logger.info("System prompt: runtime override set (%d chars)", len(stripped))


def reset_system_prompt() -> None:
    """Clear the runtime override, reverting to file default or preset."""
    global _runtime_prompt
    with _lock:
        _runtime_prompt = None
    logger.info("System prompt: runtime override cleared")


def is_using_preset() -> bool:
    """Return ``True`` when no custom prompt is active (preset mode)."""
    return get_system_prompt() is None


def get_prompt_mode() -> str:
    """Return the current prompt mode as a string label."""
    with _lock:
        if _runtime_prompt is not None:
            return "custom"
    if _default_prompt is not None:
        return "file"
    return "preset"
