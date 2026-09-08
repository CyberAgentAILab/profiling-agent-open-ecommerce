"""Unit tests for ClusterSubAgent clustering logic (embedding model stubbed out)."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd


def _load_module(rel_path: str, name: str) -> ModuleType:
    path = Path(__file__).parent.parent.parent / "src" / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cluster_model = _load_module("cluster_attribute/model.py", "cluster_attribute_model")


def _stub_agent(cache_dir: str) -> Any:
    """ClusterSubAgent without __init__ (no embedding model loading)."""
    agent = cluster_model.ClusterSubAgent.__new__(cluster_model.ClusterSubAgent)
    agent.model_name = "stub-model"
    agent.model_slug = "stub-model"
    agent.model_cache_dir = str(Path(cache_dir) / "stub-model")
    agent.embedding_backend = "sentence_transformer"
    agent.batch_size = 16
    agent.timestamp = "ts"
    agent.algorithm = "kmeans"
    agent.n_clusters = 2
    agent.seed = 0
    agent.dimension_reduction = "none"
    return agent


class TestModelSlug(unittest.TestCase):
    def test_uses_leaf_name(self) -> None:
        self.assertEqual(
            cluster_model.ClusterSubAgent._model_slug("./models/Qwen3-Embedding-0.6B"),
            "Qwen3-Embedding-0.6B",
        )

    def test_sanitizes_unsafe_characters(self) -> None:
        self.assertEqual(cluster_model.ClusterSubAgent._model_slug("org/model name"), "model_name")


class TestToNumpy(unittest.TestCase):
    def test_ndarray_passthrough_and_list_conversion(self) -> None:
        arr = np.array([1.0, 2.0])
        self.assertIs(cluster_model.ClusterSubAgent._to_numpy(arr), arr)
        self.assertTrue((cluster_model.ClusterSubAgent._to_numpy([1.0, 2.0]) == arr).all())


class TestFitClusters(unittest.TestCase):
    def test_kmeans_separates_obvious_groups(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            agent = _stub_agent(d)
            embeddings = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0]])
            labels = agent._fit_clusters(embeddings)
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[2], labels[3])
        self.assertNotEqual(labels[0], labels[2])

    def test_unknown_algorithm_raises(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            agent = _stub_agent(d)
            agent.algorithm = "dbscan"
            with self.assertRaises(ValueError):
                agent._fit_clusters(np.zeros((4, 2)))


class TestRunClustering(unittest.TestCase):
    def test_merges_cluster_ids_into_profiles(self) -> None:
        pattern = "user_attribute_demographic"
        profiles: list[dict] = [
            {
                "resolved_query": "q1",
                "user": {
                    pattern: {
                        "consumer_attributes": [{"pseudo_confidence": {"attr-a": 0.8, "attr-b": 0.3}}],
                    }
                },
            }
        ]
        attribute_df = pd.DataFrame(
            [
                {"resolved_query": "q1", "pattern": "demographic", "attribute": "attr-a", "pseudo_confidence": 0.8},
                {"resolved_query": "q1", "pattern": "demographic", "attribute": "attr-b", "pseudo_confidence": 0.3},
            ]
        )
        with tempfile.TemporaryDirectory() as d:
            agent = _stub_agent(d)
            agent._compute_embeddings = lambda texts: np.array([[0.0, 0.0], [10.0, 10.0]])
            agent.run_clustering(profiles=profiles, attribute_df=attribute_df, target_patterns=[pattern])

        merged = profiles[0]["attribute2cluster_id"][pattern]
        self.assertEqual(set(merged.keys()), {"attr-a", "attr-b"})
        self.assertNotEqual(merged["attr-a"]["cluster_id"], merged["attr-b"]["cluster_id"])
        self.assertEqual(merged["attr-a"]["pseudo_confidence"], 0.8)

    def test_mismatched_cache_triggers_recompute(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            agent = _stub_agent(d)
            cache_dir = Path(agent.model_cache_dir)
            cache_dir.mkdir(parents=True)
            pd.DataFrame(
                {
                    "resolved_query": ["q1"],
                    "pattern": ["demographic"],
                    "attribute_text": ["stale-attribute"],
                    "embedding": [[9.0, 9.0]],
                }
            ).to_csv(cache_dir / "attribute_embeddings_old.csv", index=False)

            agent.dimension_reduction = "none"
            agent._compute_embeddings = lambda texts: np.array([[1.0, 2.0]])
            df = pd.DataFrame(
                [{"resolved_query": "q1", "pattern": "demographic", "attribute": "attr-a", "pseudo_confidence": 0.8}]
            )
            result = agent.vectorize_attribute(df_user_attributes=df)
        self.assertEqual(result["embedding"].tolist(), [[1.0, 2.0]])

    def test_embeddings_loaded_from_cache_skip_compute(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            agent = _stub_agent(d)
            cache_dir = Path(agent.model_cache_dir)
            cache_dir.mkdir(parents=True)
            pd.DataFrame(
                {
                    "resolved_query": ["q1"],
                    "pattern": ["demographic"],
                    "attribute_text": ["attr-a"],
                    "embedding": [[1.0, 2.0]],
                }
            ).to_csv(cache_dir / "attribute_embeddings_old.csv", index=False)

            def _fail(texts: list[str]) -> np.ndarray:
                raise AssertionError("compute should not be called when cache exists")

            agent._compute_embeddings = _fail
            df = pd.DataFrame(
                [{"resolved_query": "q1", "pattern": "demographic", "attribute": "attr-a", "pseudo_confidence": 0.8}]
            )
            result = agent.vectorize_attribute(df_user_attributes=df)
        self.assertEqual(result["embedding"].tolist(), [[1.0, 2.0]])


class TestFitClustersTinyInputs(unittest.TestCase):
    def test_single_attribute_gets_one_cluster_instead_of_failing(self) -> None:
        for algorithm in ("hdbscan", "kmeans", "minibatch_kmeans"):
            agent = _stub_agent(tempfile.mkdtemp())
            agent.algorithm = algorithm
            labels = agent._fit_clusters(np.array([[1.0, 0.0]]))
            self.assertEqual(labels.tolist(), [0], algorithm)

    def test_kmeans_never_requests_more_clusters_than_samples(self) -> None:
        agent = _stub_agent(tempfile.mkdtemp())
        agent.algorithm = "kmeans"
        agent.n_clusters = 10
        labels = agent._fit_clusters(np.array([[0.0, 0.0], [10.0, 10.0]]))
        self.assertEqual(len(labels), 2)
        self.assertEqual(len(set(labels.tolist())), 2)


if __name__ == "__main__":
    unittest.main()
