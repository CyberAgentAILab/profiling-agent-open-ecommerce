"""Unit tests for the judge_user_attribute loaders and per-user aggregation."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

import pandas as pd


def _load_module(rel_path: str, name: str) -> ModuleType:
    path = Path(__file__).parent.parent.parent / "src" / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dataset = _load_module("judge_user_attribute/dataset.py", "judge_user_attribute_dataset")


class TestLoadTagDb(unittest.TestCase):
    def _write_db(self, records: list[dict]) -> str:
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
        json.dump(records, f)
        f.close()
        return f.name

    def test_tags_sorted_by_confidence_and_outlier_dropped(self) -> None:
        path = self._write_db(
            [
                {
                    "resolved_query": "USB Cable",
                    "tag2pseudo_confidence_demographic": {"low": 0.2, "high": 0.9, "outlier": 1.0},
                }
            ]
        )
        title_to_tags, all_tags = dataset.load_tag_db(path, tag_suffix="demographic")
        self.assertEqual(title_to_tags["USB Cable"], ["high", "low"])
        self.assertEqual(all_tags, ["high", "low"])

    def test_records_without_tags_skipped(self) -> None:
        path = self._write_db(
            [
                {"resolved_query": "no-map"},
                {"resolved_query": "only-outlier", "tag2pseudo_confidence_demographic": {"outlier": 0.5}},
            ]
        )
        title_to_tags, all_tags = dataset.load_tag_db(path, tag_suffix="demographic")
        self.assertEqual(title_to_tags, {})
        self.assertEqual(all_tags, [])


class TestFilterPurchasesByMinBuyers(unittest.TestCase):
    def test_drops_titles_below_threshold(self) -> None:
        df = pd.DataFrame(
            {
                "Title": ["popular", "popular", "niche"],
                "Survey ResponseID": ["u1", "u2", "u1"],
            }
        )
        filtered = dataset.filter_purchases_by_min_buyers(df, min_unique_buyers=2)
        self.assertEqual(set(filtered["Title"]), {"popular"})


class TestAggregateTitlesPerUser(unittest.TestCase):
    def test_counts_ordered_desc(self) -> None:
        df = pd.DataFrame(
            {
                "Title": ["a", "b", "b", "c"],
                "Survey ResponseID": ["u1"] * 4,
            }
        )
        result = dataset.aggregate_titles_per_user(df)
        self.assertEqual(list(result["u1"].items()), [("b", 2), ("a", 1), ("c", 1)])


class TestAggregateTagsPerUser(unittest.TestCase):
    def test_counts_weighted_and_zero_filled(self) -> None:
        df = pd.DataFrame(
            {
                "Title": ["cable", "cable", "beans"],
                "Survey ResponseID": ["u1"] * 3,
            }
        )
        title_to_tags = {"cable": ["tech"], "beans": ["food"]}
        all_tags = ["food", "tech", "unused"]
        user_tags, user_title_tags = dataset.aggregate_tags_per_user(
            purchases=df, title_to_tags=title_to_tags, all_tags=all_tags
        )
        self.assertEqual(user_tags["u1"], {"tech": 2, "food": 1, "unused": 0})
        self.assertEqual(user_title_tags["u1"], {"cable": ["tech"], "beans": ["food"]})

    def test_max_products_caps_title_selection(self) -> None:
        df = pd.DataFrame(
            {
                "Title": ["cable", "cable", "beans"],
                "Survey ResponseID": ["u1"] * 3,
            }
        )
        title_to_tags = {"cable": ["tech"], "beans": ["food"]}
        user_tags, _ = dataset.aggregate_tags_per_user(
            purchases=df, title_to_tags=title_to_tags, all_tags=["food", "tech"], max_products=1
        )
        self.assertEqual(user_tags["u1"], {"tech": 2, "food": 0})


if __name__ == "__main__":
    unittest.main()
