"""Unit tests for the tag_cluster cluster/confidence mapping builders."""

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


dataset = _load_module("tag_cluster/dataset.py", "tag_cluster_dataset")

PATTERN = "user_attribute_demographic"


def _profile(mapping: dict) -> dict:
    return {"attribute2cluster_id": {PATTERN: mapping}}


class TestBuildClusterId2Attributes(unittest.TestCase):
    def test_new_format_groups_attributes_by_cluster(self) -> None:
        profiles = [
            _profile(
                {
                    "attr-a": {"cluster_id": 0, "pseudo_confidence": 0.8},
                    "attr-b": {"cluster_id": 0, "pseudo_confidence": 0.5},
                    "attr-c": {"cluster_id": 1, "pseudo_confidence": 0.9},
                }
            )
        ]
        result = dataset.build_cluster_id2attributes(profiles)
        self.assertEqual(result[PATTERN][0], ["attr-a", "attr-b"])
        self.assertEqual(result[PATTERN][1], ["attr-c"])

    def test_old_format_plain_cluster_id(self) -> None:
        profiles = [_profile({"attr-a": 3})]
        result = dataset.build_cluster_id2attributes(profiles)
        self.assertEqual(result[PATTERN][3], ["attr-a"])

    def test_duplicate_attributes_not_repeated(self) -> None:
        mapping = {"attr-a": {"cluster_id": 0}}
        result = dataset.build_cluster_id2attributes([_profile(mapping), _profile(mapping)])
        self.assertEqual(result[PATTERN][0], ["attr-a"])

    def test_none_cluster_id_skipped(self) -> None:
        result = dataset.build_cluster_id2attributes([_profile({"attr-a": {"cluster_id": None}})])
        self.assertEqual(result.get(PATTERN, {}), {})


class TestBuildAttribute2PseudoConfidence(unittest.TestCase):
    def test_keeps_maximum_across_profiles(self) -> None:
        profiles = [
            _profile({"attr-a": {"cluster_id": 0, "pseudo_confidence": 0.3}}),
            _profile({"attr-a": {"cluster_id": 0, "pseudo_confidence": 0.7}}),
        ]
        result = dataset.build_attribute2pseudo_confidence(profiles)
        self.assertEqual(result["attr-a"], 0.7)

    def test_string_confidence_converted(self) -> None:
        result = dataset.build_attribute2pseudo_confidence(
            [_profile({"attr-a": {"cluster_id": 0, "pseudo_confidence": "0.4"}})]
        )
        self.assertEqual(result["attr-a"], 0.4)

    def test_invalid_entries_skipped(self) -> None:
        result = dataset.build_attribute2pseudo_confidence(
            [
                _profile(
                    {
                        "old-format": 1,
                        "no-confidence": {"cluster_id": 0},
                        "bad-string": {"cluster_id": 0, "pseudo_confidence": "n/a"},
                    }
                )
            ]
        )
        self.assertEqual(result, {})


class TestLoadAttributeClusters(unittest.TestCase):
    def test_loads_profiles_and_builds_mappings(self) -> None:
        profiles = [_profile({"attr-a": {"cluster_id": 0, "pseudo_confidence": 0.8}})]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(profiles, f)
        loaded, cluster_map, confidence_map = dataset.load_attribute_clusters(f.name)
        self.assertEqual(loaded, profiles)
        self.assertEqual(cluster_map[PATTERN][0], ["attr-a"])
        self.assertEqual(confidence_map["attr-a"], 0.8)


if __name__ == "__main__":
    unittest.main()
