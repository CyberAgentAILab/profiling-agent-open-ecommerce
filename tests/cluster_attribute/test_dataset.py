"""Unit tests for the cluster_attribute profile loader and DataFrame builder."""

import importlib.util
import json
import tempfile
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


dataset = _load_module("cluster_attribute/dataset.py", "cluster_attribute_dataset")


def _profile(query: str, pattern: str, pseudo_confidence: dict[str, float]) -> dict:
    return {
        "resolved_query": query,
        "user": {
            pattern: {
                "consumer_attributes": [{"pseudo_confidence": pseudo_confidence}],
            }
        },
    }


class TestExtractAttributes(unittest.TestCase):
    def test_extracts_pseudo_confidence_of_last_entry(self) -> None:
        nested = {
            "p": {
                "consumer_attributes": [
                    {"pseudo_confidence": {"old": 0.1}},
                    {"pseudo_confidence": {"new": 0.9}},
                ]
            }
        }
        self.assertEqual(dataset.extract_attributes(nested, "p"), {"new": 0.9})

    def test_confidence_key_of_slim_prompt_is_accepted(self) -> None:
        entries = [{"category": "signal:x", "attribute": "unknown"}, {"confidence": {"a": 0.7}}]
        nested = {"p": {"consumer_attributes": entries}}
        self.assertEqual(dataset.extract_attributes(nested, "p"), {"a": 0.7})

    def test_pseudo_confidence_takes_precedence_over_confidence(self) -> None:
        nested = {"p": {"consumer_attributes": [{"pseudo_confidence": {"a": 0.9}, "confidence": {"b": 0.1}}]}}
        self.assertEqual(dataset.extract_attributes(nested, "p"), {"a": 0.9})

    def test_missing_pattern_returns_none(self) -> None:
        self.assertIsNone(dataset.extract_attributes({}, "p"))

    def test_malformed_nesting_returns_none(self) -> None:
        self.assertIsNone(dataset.extract_attributes({"p": {"consumer_attributes": "not-a-list"}}, "p"))
        self.assertIsNone(dataset.extract_attributes({"p": {"consumer_attributes": ["not-a-dict"]}}, "p"))
        self.assertIsNone(dataset.extract_attributes({"p": {"consumer_attributes": [{}]}}, "p"))

    def test_empty_consumer_attributes_returns_none(self) -> None:
        self.assertIsNone(dataset.extract_attributes({"p": {"consumer_attributes": []}}, "p"))


class TestPatternLabel(unittest.TestCase):
    def test_mapped_pattern(self) -> None:
        self.assertEqual(dataset._pattern_label("user_attribute_ecommerce"), "purchase")

    def test_prefix_stripped(self) -> None:
        self.assertEqual(dataset._pattern_label("user_attribute_demographic"), "demographic")


class TestLoadProfiles(unittest.TestCase):
    def test_directory_input_merges_files(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            for i in range(2):
                with open(Path(d) / f"part-{i}.json", "w", encoding="utf-8") as f:
                    json.dump([{"resolved_query": f"q{i}"}], f)
            profiles = dataset.load_profiles(d)
        self.assertEqual({p["resolved_query"] for p in profiles}, {"q0", "q1"})

    def test_invalid_path_raises(self) -> None:
        with self.assertRaises(ValueError):
            dataset.load_profiles("/nonexistent/missing.json")


class TestBuildAttributeDf(unittest.TestCase):
    def test_builds_long_form_rows(self) -> None:
        profiles = [
            _profile("q1", "user_attribute_demographic", {"attr-a": 0.8, "attr-b": 0.5}),
        ]
        df = dataset.build_attribute_df(profiles, ["user_attribute_demographic"])
        self.assertEqual(len(df), 2)
        self.assertEqual(set(df["attribute"]), {"attr-a", "attr-b"})
        self.assertEqual(set(df["pattern"]), {"demographic"})

    def test_skips_profiles_without_target_pattern(self) -> None:
        profiles = [
            _profile("q1", "user_attribute_demographic", {"attr-a": 0.8}),
            _profile("q2", "user_attribute_life_event", {"attr-b": 0.5}),
            {"resolved_query": "q3"},
        ]
        df = dataset.build_attribute_df(profiles, ["user_attribute_demographic"])
        self.assertEqual(df["resolved_query"].tolist(), ["q1"])

    def test_empty_profiles_return_empty_df(self) -> None:
        df = dataset.build_attribute_df([], ["user_attribute_demographic"])
        self.assertEqual(len(df), 0)


if __name__ == "__main__":
    unittest.main()
