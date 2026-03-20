"""Tests for src/system_prompt module."""

import pytest

from src import system_prompt as sp


@pytest.fixture(autouse=True)
def _reset_module():
    """Reset module-level state before each test."""
    sp._default_prompt = None
    sp._runtime_prompt = None
    yield
    sp._default_prompt = None
    sp._runtime_prompt = None


class TestLoadDefaultPrompt:
    def test_empty_path_uses_preset(self):
        sp.load_default_prompt("")
        assert sp._default_prompt is None
        assert sp.is_using_preset()

    def test_blank_path_uses_preset(self):
        sp.load_default_prompt("   ")
        assert sp._default_prompt is None

    def test_valid_file_loads_content(self, tmp_path):
        f = tmp_path / "prompt.txt"
        f.write_text("You are a helpful assistant.", encoding="utf-8")
        sp.load_default_prompt(str(f))
        assert sp._default_prompt == "You are a helpful assistant."
        assert not sp.is_using_preset()

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            sp.load_default_prompt("/nonexistent/path.txt")

    def test_empty_file_falls_back_to_preset(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        sp.load_default_prompt(str(f))
        assert sp._default_prompt is None
        assert sp.is_using_preset()

    def test_whitespace_only_file_falls_back(self, tmp_path):
        f = tmp_path / "blank.txt"
        f.write_text("   \n  \n  ", encoding="utf-8")
        sp.load_default_prompt(str(f))
        assert sp._default_prompt is None


class TestGetSetReset:
    def test_preset_mode_returns_none(self):
        assert sp.get_system_prompt() is None
        assert sp.is_using_preset()
        assert sp.get_prompt_mode() == "preset"

    def test_file_default(self):
        sp._default_prompt = "from file"
        assert sp.get_system_prompt() == "from file"
        assert sp.get_prompt_mode() == "file"
        assert not sp.is_using_preset()

    def test_runtime_override_takes_priority(self):
        sp._default_prompt = "from file"
        sp.set_system_prompt("runtime override")
        assert sp.get_system_prompt() == "runtime override"
        assert sp.get_prompt_mode() == "custom"

    def test_reset_reverts_to_file_default(self):
        sp._default_prompt = "from file"
        sp.set_system_prompt("override")
        sp.reset_system_prompt()
        assert sp.get_system_prompt() == "from file"
        assert sp.get_prompt_mode() == "file"

    def test_reset_reverts_to_preset(self):
        sp.set_system_prompt("override")
        sp.reset_system_prompt()
        assert sp.get_system_prompt() is None
        assert sp.get_prompt_mode() == "preset"

    def test_set_empty_raises_value_error(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            sp.set_system_prompt("")

    def test_set_whitespace_raises_value_error(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            sp.set_system_prompt("   \n  ")

    def test_set_strips_whitespace(self):
        sp.set_system_prompt("  hello world  ")
        assert sp.get_system_prompt() == "hello world"

    def test_get_default_prompt_ignores_runtime(self):
        sp._default_prompt = "from file"
        sp.set_system_prompt("runtime")
        assert sp.get_default_prompt() == "from file"


class TestGetPromptMode:
    def test_preset(self):
        assert sp.get_prompt_mode() == "preset"

    def test_file(self):
        sp._default_prompt = "file content"
        assert sp.get_prompt_mode() == "file"

    def test_custom(self):
        sp.set_system_prompt("custom content")
        assert sp.get_prompt_mode() == "custom"

    def test_custom_over_file(self):
        sp._default_prompt = "file"
        sp.set_system_prompt("custom")
        assert sp.get_prompt_mode() == "custom"
