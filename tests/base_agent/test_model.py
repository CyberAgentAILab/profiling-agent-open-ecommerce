"""Unit tests for BaseAgent helper methods (no model loading involved)."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from typing import Any


def _load_module(rel_path: str, name: str) -> ModuleType:
    path = Path(__file__).parent.parent.parent / "src" / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


model = _load_module("base_agent/model.py", "base_agent_model")


class _FakeTokenizer:
    """Stub that mimics apply_chat_template(tokenize=False) returning a string."""

    def apply_chat_template(
        self,
        chat: list[dict[str, str]],
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str:
        return f"<chat thinking={enable_thinking}>{chat[0]['content']}"


def _bare_agent() -> Any:
    """BaseAgent instance without running __init__ (which loads models)."""
    agent = model.BaseAgent.__new__(model.BaseAgent)
    agent.tokenizer = _FakeTokenizer()
    agent.prompt_template = "Q: {query}"
    agent.prompt_template_extra = "EXTRA: {query}"
    agent.enable_thinking = None
    return agent


class TestFindTokenPosition(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = _bare_agent()

    def test_returns_position_after_last_occurrence(self) -> None:
        self.assertEqual(self.agent._find_token_position([1, 9, 2, 9, 3], token_id=9), 4)

    def test_missing_token_returns_zero(self) -> None:
        self.assertEqual(self.agent._find_token_position([1, 2, 3], token_id=9), 0)

    def test_empty_sequence_returns_zero(self) -> None:
        self.assertEqual(self.agent._find_token_position([], token_id=9), 0)


class TestFindConfigDir(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = _bare_agent()

    def test_finds_nested_config_json(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            nested = Path(root) / "snapshots" / "abc"
            nested.mkdir(parents=True)
            (nested / "config.json").write_text("{}", encoding="utf-8")
            self.assertEqual(self.agent.find_config_dir(root), nested)

    def test_returns_none_when_absent(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            self.assertIsNone(self.agent.find_config_dir(root))


class TestBuildMessage(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = _bare_agent()

    def _build(self, prompt: str | list[str], mode: str = "thinking", use_extra_prompt: bool = False) -> list[str]:
        return self.agent._BaseAgent__build_message(prompt=prompt, mode=mode, use_extra_prompt=use_extra_prompt)

    def test_single_prompt_applies_template(self) -> None:
        messages = self._build("hello")
        self.assertEqual(messages, ["<chat thinking=True>Q: hello"])

    def test_batch_prompt_applies_template_per_item(self) -> None:
        messages = self._build(["one", "two"])
        self.assertEqual(
            messages,
            ["<chat thinking=True>Q: one", "<chat thinking=True>Q: two"],
        )

    def test_extra_prompt_template(self) -> None:
        messages = self._build("hello", use_extra_prompt=True)
        self.assertEqual(messages, ["<chat thinking=True>EXTRA: hello"])

    def test_non_thinking_mode(self) -> None:
        messages = self._build("hello", mode="plain")
        self.assertEqual(messages, ["<chat thinking=False>Q: hello"])

    def test_enable_thinking_overrides_mode(self) -> None:
        self.agent.enable_thinking = False
        messages = self._build("hello", mode="thinking")
        self.assertEqual(messages, ["<chat thinking=False>Q: hello"])


if __name__ == "__main__":
    unittest.main()
