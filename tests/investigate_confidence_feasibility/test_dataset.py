"""Unit tests for the feasibility-investigation data loading."""

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


dataset = _load_module("investigate_confidence_feasibility/dataset.py", "icf_dataset")


class TestIterProductObjects(unittest.TestCase):
    def _write_db(self, records: list[dict]) -> str:
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
        json.dump(records, f, indent=2)
        f.close()
        return f.name

    def test_streams_each_object(self) -> None:
        records = [{"resolved_query": "a"}, {"resolved_query": "b"}]
        path = self._write_db(records)
        self.assertEqual(list(dataset.iter_product_objects(path)), records)

    def test_braces_and_quotes_inside_strings_do_not_break_boundaries(self) -> None:
        records = [{"snippet": 'tricky "quoted" {brace} \\ text', "nested": {"k": "}{"}}]
        path = self._write_db(records)
        self.assertEqual(list(dataset.iter_product_objects(path)), records)


class TestLoadTagConfidences(unittest.TestCase):
    def test_max_confidence_per_title(self) -> None:
        records = [
            {
                "raw_query_mapping": ["title-1", "title-2"],
                "tag2pseudo_confidence_life_event": {"moving": 0.4},
            },
            {
                "raw_query_mapping": ["title-1"],
                "tag2pseudo_confidence_life_event": {"moving": 0.9},
            },
            {"raw_query_mapping": [], "tag2pseudo_confidence_life_event": {"moving": 1.0}},
        ]
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
        json.dump(records, f)
        f.close()
        result = dataset.load_tag_confidences(f.name, {"moving": "life_event"})
        self.assertEqual(result["moving"], {"title-1": 0.9, "title-2": 0.4})


class TestBuildUserTagConfidence(unittest.TestCase):
    def _write_purchases(self, rows: list[tuple[str, str]]) -> str:
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8")
        f.write("Title,Survey ResponseID\n")
        for title, user in rows:
            f.write(f"{title},{user}\n")
        f.close()
        return f.name

    def test_max_aggregation(self) -> None:
        path = self._write_purchases([("t-low", "u1"), ("t-high", "u1"), ("t-low", "u2")])
        tag_title_conf = {"moving": {"t-low": 0.3, "t-high": 0.8}}
        result = dataset.build_user_tag_confidence(path, tag_title_conf, agg="max")
        self.assertEqual(result["moving"], {"u1": 0.8, "u2": 0.3})

    def test_mean_aggregation(self) -> None:
        path = self._write_purchases([("t-low", "u1"), ("t-high", "u1")])
        tag_title_conf = {"moving": {"t-low": 0.2, "t-high": 0.8}}
        result = dataset.build_user_tag_confidence(path, tag_title_conf, agg="mean")
        self.assertAlmostEqual(result["moving"]["u1"], 0.5)

    def test_numeric_user_ids_keyed_as_strings(self) -> None:
        # pandas infers numeric-looking IDs as int; keys must still be str so
        # they match the survey's string index downstream.
        path = self._write_purchases([("t", "12345")])
        result = dataset.build_user_tag_confidence(path, {"moving": {"t": 0.7}})
        self.assertEqual(result["moving"], {"12345": 0.7})

    def test_users_without_tagged_titles_absent(self) -> None:
        path = self._write_purchases([("untagged", "u1")])
        result = dataset.build_user_tag_confidence(path, {"moving": {"t": 0.5}})
        self.assertEqual(result["moving"], {})


class TestLabelFor(unittest.TestCase):
    def test_yes_no(self) -> None:
        self.assertEqual(dataset.label_for("Yes", "yes_no", "Yes"), 1)
        self.assertEqual(dataset.label_for("no", "yes_no", "Yes"), 0)
        self.assertIsNone(dataset.label_for("Prefer not to say", "yes_no", "Yes"))
        self.assertIsNone(dataset.label_for(float("nan"), "yes_no", "Yes"))

    def test_substring(self) -> None:
        self.assertEqual(dataset.label_for("Divorce,Moved", "substring", "Divorce"), 1)
        self.assertEqual(dataset.label_for("Moved", "substring", "Divorce"), 0)
        # absence (even NaN) is a definite 0 for multi-select fields
        self.assertEqual(dataset.label_for(float("nan"), "substring", "Divorce"), 0)


if __name__ == "__main__":
    unittest.main()
