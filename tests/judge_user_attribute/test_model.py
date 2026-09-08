"""Unit tests for judge_user_attribute parsing and context-building helpers."""

import importlib.util
import json
import unittest
from pathlib import Path
from types import ModuleType


def _load_module(rel_path: str, name: str) -> ModuleType:
    path = Path(__file__).parent.parent.parent / "src" / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


model = _load_module("judge_user_attribute/model.py", "judge_user_attribute_model")


class TestExtractJsonObject(unittest.TestCase):
    def setUp(self) -> None:
        self.extract = model.JudgeUserAttributeAgent._extract_json_object

    def test_raw_json(self) -> None:
        self.assertEqual(self.extract('{"a": 1}'), {"a": 1})

    def test_fenced_json(self) -> None:
        self.assertEqual(self.extract('```json\n{"a": 1}\n```'), {"a": 1})

    def test_embedded_json(self) -> None:
        self.assertEqual(self.extract('The answer is {"a": 1} as requested.'), {"a": 1})

    def test_no_json_returns_none(self) -> None:
        self.assertIsNone(self.extract("no json here"))


class TestCoerceAttributeEntry(unittest.TestCase):
    def setUp(self) -> None:
        self.coerce = model.JudgeUserAttributeAgent._coerce_attribute_entry

    def test_high_confidence_is_positive(self) -> None:
        self.assertEqual(self.coerce(0.9), (True, 0.9))

    def test_low_confidence_is_negative(self) -> None:
        self.assertEqual(self.coerce(0.1), (False, 0.1))

    def test_out_of_range_clamped(self) -> None:
        self.assertEqual(self.coerce(1.7), (True, 1.0))
        self.assertEqual(self.coerce(-0.3), (False, 0.0))

    def test_non_numeric_defaults_to_neutral(self) -> None:
        self.assertEqual(self.coerce("yes"), (True, 0.5))
        self.assertEqual(self.coerce(None), (True, 0.5))


class TestBuildBaselineUserContext(unittest.TestCase):
    def test_renders_ranked_titles(self) -> None:
        context = model.build_baseline_user_context({"cable": 2, "beans": 1})
        self.assertIn("- cable (x2)", context)
        self.assertIn("- beans (x1)", context)
        self.assertIn("2 unique titles, 3 total purchases", context)

    def test_truncation_note_when_capped(self) -> None:
        context = model.build_baseline_user_context({"a": 3, "b": 2, "c": 1}, max_titles=2)
        self.assertIn("showing top 2 unique titles", context)
        self.assertNotIn("- c (x1)", context)

    def test_empty_history(self) -> None:
        context = model.build_baseline_user_context({})
        self.assertIn("(no purchases recorded for this user)", context)


class TestParseResponse(unittest.TestCase):
    def _parse(self, raw_text: str) -> tuple:
        agent = model.JudgeUserAttributeAgent.__new__(model.JudgeUserAttributeAgent)
        return agent._parse_response(raw_text)

    def test_parse_miss_collapses_to_neutral(self) -> None:
        predicted, proba, reasoning, raw = self._parse("not a json")
        self.assertEqual(reasoning, "")
        self.assertEqual(raw, {})
        self.assertTrue(all(p == 0.5 for p in proba.values()))
        self.assertTrue(all(v is False for v in predicted.values()))

    def test_valid_response_sets_attributes(self) -> None:
        key = model.ATTRIBUTE_KEYS[0]
        payload = {"reasoning": "because", "attributes": {key: 0.9}}
        predicted, proba, reasoning, _ = self._parse(json.dumps(payload))
        self.assertEqual(reasoning, "because")
        self.assertTrue(predicted[key])
        self.assertEqual(proba[key], 0.9)


if __name__ == "__main__":
    unittest.main()
