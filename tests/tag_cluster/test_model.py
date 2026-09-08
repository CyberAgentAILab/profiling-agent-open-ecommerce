"""Unit tests for TaggerSubAgent tagging logic (LLM generation stubbed out)."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np


def _load_module(rel_path: str, name: str) -> ModuleType:
    path = Path(__file__).parent.parent.parent / "src" / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tag_model = _load_module("tag_cluster/model.py", "tag_cluster_model")

PATTERN = "user_attribute_demographic"


def _stub_agent(out_dir: str, tag: str = "T") -> Any:
    """TaggerSubAgent without __init__ (no model loading); generate() is stubbed."""
    agent = tag_model.TaggerSubAgent.__new__(tag_model.TaggerSubAgent)
    agent.json_schema_tag = ["reasoning", "tag"]
    agent.sampling_method = "head"
    agent.embeddings = None
    agent.attribute_to_index = None
    agent.seed = 0
    agent.out_dir = out_dir
    agent.timestamp = "ts"
    agent.generate = lambda prompt: [
        {"content": json.dumps({"reasoning": "r", "tag": tag}), "input_tokens": 1, "output_tokens": 2} for _ in prompt
    ]
    return agent


class TestSelectRepresentativeIndices(unittest.TestCase):
    def test_first_method(self) -> None:
        indices = tag_model.select_representative_indices(5, 3, method="first")
        self.assertEqual(indices.tolist(), [0, 1, 2])

    def test_n_select_capped_at_n_samples(self) -> None:
        indices = tag_model.select_representative_indices(2, 10, method="first")
        self.assertEqual(indices.tolist(), [0, 1])

    def test_kmedoids_picks_one_per_group(self) -> None:
        x = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0]])
        indices = tag_model.select_representative_indices(4, 2, method="kmedoids", seed=0, x=x)
        self.assertEqual(len(indices), 2)
        groups = {0 if x[i][0] < 5 else 1 for i in indices}
        self.assertEqual(groups, {0, 1})

    def test_kmedoids_without_embeddings_raises(self) -> None:
        with self.assertRaises(ValueError):
            tag_model.select_representative_indices(4, 2, method="kmedoids")

    def test_unknown_method_raises(self) -> None:
        with self.assertRaises(ValueError):
            tag_model.select_representative_indices(4, 2, method="magic")


class TestBuildContext(unittest.TestCase):
    def test_head_sampling_joins_first_100(self) -> None:
        agent = _stub_agent(".")
        attrs = [f"attr-{i}" for i in range(150)]
        context = agent._build_context({"cluster_id": 0, "cluster_attributes": attrs})
        self.assertEqual(context.split("\n"), attrs[:100])


class TestAddTag(unittest.TestCase):
    def _clusters(self) -> list[dict]:
        return [{"cluster_id": 0, "cluster_attributes": ["attr-a"]}]

    def test_valid_output_marks_success(self) -> None:
        agent = _stub_agent(".")
        results = agent.add_tag(batch_cluster=self._clusters())
        self.assertTrue(results[0]["success_tagger"])

    def test_invalid_json_marks_failure(self) -> None:
        agent = _stub_agent(".")
        agent.generate = lambda prompt: [{"content": "not a json", "input_tokens": 1, "output_tokens": 2}]
        results = agent.add_tag(batch_cluster=self._clusters())
        self.assertFalse(results[0]["success_tagger"])

    def test_schema_mismatch_marks_failure(self) -> None:
        agent = _stub_agent(".")
        agent.generate = lambda prompt: [
            {"content": json.dumps({"reasoning": "r"}), "input_tokens": 1, "output_tokens": 2}
        ]
        results = agent.add_tag(batch_cluster=self._clusters())
        self.assertFalse(results[0]["success_tagger"])


class TestRunTagging(unittest.TestCase):
    def _profiles(self) -> list[dict]:
        return [
            {
                "resolved_query": "q1",
                "attribute2cluster_id": {
                    PATTERN: {
                        "attr-a": {"cluster_id": 0, "pseudo_confidence": 0.8},
                        "attr-b": {"cluster_id": -1, "pseudo_confidence": 0.4},
                    }
                },
            }
        ]

    def test_per_pattern_tagging_merges_into_profiles(self) -> None:
        profiles = self._profiles()
        cluster_map = {PATTERN: {0: ["attr-a"], -1: ["attr-b"]}}
        with tempfile.TemporaryDirectory() as out_dir:
            agent = _stub_agent(out_dir, tag="DemographicTag")
            save_path = agent.run_tagging(
                profiles=profiles,
                cluster_id2attributes=cluster_map,
                attribute2pseudo_confidence={"attr-a": 0.8, "attr-b": 0.4},
                batch_size=10,
                debug_num_sample=None,
                cluster_id_namespace="per_pattern",
            )
            with open(save_path, encoding="utf-8") as f:
                saved = json.load(f)

        profile = saved[0]
        self.assertEqual(profile["cluster_id2tag_demographic"]["0"], "DemographicTag")
        self.assertEqual(profile["cluster_id2tag_demographic"]["-1"], "outlier")
        self.assertEqual(profile["tag2pseudo_confidence_demographic"]["DemographicTag"], 0.8)
        self.assertIn("duration_tag", profile)

    def test_missing_per_profile_confidence_falls_back_to_global_map(self) -> None:
        profiles: list[dict] = [
            {
                "resolved_query": "q1",
                "attribute2cluster_id": {PATTERN: {"attr-a": {"cluster_id": 0}}},
            }
        ]
        with tempfile.TemporaryDirectory() as out_dir:
            agent = _stub_agent(out_dir, tag="T")
            save_path = agent.run_tagging(
                profiles=profiles,
                cluster_id2attributes={PATTERN: {0: ["attr-a"]}},
                attribute2pseudo_confidence={"attr-a": 0.6},
                batch_size=10,
                debug_num_sample=None,
                cluster_id_namespace="per_pattern",
            )
            with open(save_path, encoding="utf-8") as f:
                saved = json.load(f)
        self.assertEqual(saved[0]["tag2pseudo_confidence_demographic"]["T"], 0.6)

    def test_debug_num_sample_caps_tagged_clusters(self) -> None:
        cluster_map = {PATTERN: {i: [f"attr-{i}"] for i in range(5)}}
        with tempfile.TemporaryDirectory() as out_dir:
            agent = _stub_agent(out_dir)
            tags = agent._process_single_pattern(
                cluster_id2attributes=cluster_map[PATTERN],
                batch_size=2,
                debug_num_sample=3,
                pattern_name=PATTERN,
            )
        self.assertEqual(len(tags), 3)


if __name__ == "__main__":
    unittest.main()
