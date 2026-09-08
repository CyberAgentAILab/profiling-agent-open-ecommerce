"""Unit tests for UserSubAgent combined-attribute parsing and batch logic."""

import importlib.util
import json
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


user_model = _load_module("predict_user/model.py", "predict_user_model")

ATTRIBUTE_KEYS = user_model.ECOMMERCE_ATTRIBUTE_KEYS

VALID_GROUP = {
    "consumer_attributes": [{"category": "c", "reason": "r", "attribute": "a"}],
}


def _stub_agent(out_dir: str, content: str) -> Any:
    """UserSubAgent without __init__ (no model loading); generate() is stubbed."""
    agent = user_model.UserSubAgent.__new__(user_model.UserSubAgent)
    agent.json_schema_parent = "consumer_attributes"
    agent.json_schema_child = ["category", "reason", "attribute"]
    agent.out_dir = out_dir
    agent.timestamp = "ts"
    agent.generate = lambda prompt, use_extra_prompt=False: [
        {"content": content, "input_tokens": 1, "output_tokens": 2} for _ in prompt
    ]
    return agent


def _combined_content() -> str:
    return json.dumps(dict.fromkeys(ATTRIBUTE_KEYS, VALID_GROUP))


class TestParseCombinedAttributes(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = _stub_agent(".", _combined_content())

    def test_valid_combined_output(self) -> None:
        result = self.agent._parse_combined_attributes(_combined_content())
        for key in ATTRIBUTE_KEYS:
            self.assertTrue(result[key]["is_json"], key)

    def test_parse_failure_marks_all_groups_invalid(self) -> None:
        result = self.agent._parse_combined_attributes("not a json")
        for key in ATTRIBUTE_KEYS:
            self.assertFalse(result[key]["is_json"], key)

    def test_non_dict_output_marks_all_groups_invalid(self) -> None:
        result = self.agent._parse_combined_attributes(json.dumps([1, 2]))
        for key in ATTRIBUTE_KEYS:
            self.assertFalse(result[key]["is_json"], key)

    def test_failure_groups_are_independent_instances(self) -> None:
        result = self.agent._parse_combined_attributes("not a json")
        result[ATTRIBUTE_KEYS[0]]["mutated"] = True
        self.assertNotIn("mutated", result[ATTRIBUTE_KEYS[1]])
        self.assertNotIn("mutated", result[ATTRIBUTE_KEYS[2]])

    def test_missing_group_marked_invalid_others_kept(self) -> None:
        partial = dict.fromkeys(ATTRIBUTE_KEYS[:2], VALID_GROUP)
        result = self.agent._parse_combined_attributes(json.dumps(partial))
        self.assertTrue(result[ATTRIBUTE_KEYS[0]]["is_json"])
        self.assertTrue(result[ATTRIBUTE_KEYS[1]]["is_json"])
        self.assertFalse(result[ATTRIBUTE_KEYS[2]]["is_json"])


class TestPredictUserAttributes(unittest.TestCase):
    def _read_results(self, save_path: str) -> list[dict]:
        with open(save_path, encoding="utf-8") as f:
            return json.load(f)

    def _item(self, query: str) -> dict:
        return {
            "resolved_query": query,
            "items": [{"product_name": query, "canonical_name": query}],
        }

    def test_empty_input_still_writes_valid_output(self) -> None:
        with tempfile.TemporaryDirectory() as out_dir:
            agent = _stub_agent(out_dir, _combined_content())
            save_path = agent.predict_user_attributes(transactions=[], batch_size=2, debug_num_sample=None)
            self.assertEqual(self._read_results(save_path), [])

    def test_attaches_user_dict_with_success_flag(self) -> None:
        with tempfile.TemporaryDirectory() as out_dir:
            agent = _stub_agent(out_dir, _combined_content())
            save_path = agent.predict_user_attributes(
                transactions=[self._item("q1"), self._item("q2")],
                batch_size=2,
                debug_num_sample=None,
            )
            results = self._read_results(save_path)
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertTrue(r["user"]["success_predict"])
            for key in ATTRIBUTE_KEYS:
                self.assertTrue(r["user"][key]["is_json"])

    def test_unparseable_output_marks_predict_failure(self) -> None:
        with tempfile.TemporaryDirectory() as out_dir:
            agent = _stub_agent(out_dir, "not a json")
            save_path = agent.predict_user_attributes(
                transactions=[self._item("q1")], batch_size=2, debug_num_sample=None
            )
            results = self._read_results(save_path)
        self.assertFalse(results[0]["user"]["success_predict"])

    def test_debug_num_sample_caps_processed_samples(self) -> None:
        with tempfile.TemporaryDirectory() as out_dir:
            agent = _stub_agent(out_dir, _combined_content())
            save_path = agent.predict_user_attributes(
                transactions=[self._item(f"q{i}") for i in range(10)],
                batch_size=2,
                debug_num_sample=3,
            )
            results = self._read_results(save_path)
        self.assertEqual(len(results), 3)


if __name__ == "__main__":
    unittest.main()
