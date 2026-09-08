"""Unit tests for the search dataset loaders."""

import csv
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


search_dataset = _load_module("search/dataset.py", "search_dataset")


class TestLoadEcommerceSearchList(unittest.TestCase):
    def _write_scan_json(self, records: list[dict]) -> str:
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
        json.dump(records, f)
        f.close()
        return f.name

    def test_need_search_filter(self) -> None:
        path = self._write_scan_json(
            [
                {"resolved_query": "keep", "scan": {"need_search": True, "is_diagnostic": False}},
                {"resolved_query": "drop", "scan": {"need_search": False, "is_diagnostic": False}},
            ]
        )
        result = search_dataset.load_ecommerce_search_list(path)
        self.assertEqual([r["resolved_query"] for r in result], ["keep"])

    def test_diagnostic_gate_bypassed_by_need_search(self) -> None:
        path = self._write_scan_json(
            [
                {"resolved_query": "diag", "scan": {"need_search": False, "is_diagnostic": True}},
                {"resolved_query": "needed", "scan": {"need_search": True, "is_diagnostic": False}},
            ]
        )
        result = search_dataset.load_ecommerce_search_list(path, filter_need_search=False)
        self.assertEqual({r["resolved_query"] for r in result}, {"diag", "needed"})

    def test_both_filters_keep_need_search_only(self) -> None:
        # filter_need_search=True takes precedence: the diagnostic gate adds
        # nothing in this mode (documented behavior, matches the paper runs).
        path = self._write_scan_json(
            [
                {"resolved_query": "needed", "scan": {"need_search": True, "is_diagnostic": False}},
                {"resolved_query": "diag-only", "scan": {"need_search": False, "is_diagnostic": True}},
                {"resolved_query": "neither", "scan": {"need_search": False, "is_diagnostic": False}},
            ]
        )
        result = search_dataset.load_ecommerce_search_list(path)
        self.assertEqual([r["resolved_query"] for r in result], ["needed"])

    def test_no_filters_keep_everything(self) -> None:
        path = self._write_scan_json(
            [
                {"resolved_query": "a", "scan": {"need_search": False, "is_diagnostic": False}},
                {"resolved_query": "b", "scan": {"need_search": True, "is_diagnostic": True}},
            ]
        )
        result = search_dataset.load_ecommerce_search_list(path, filter_need_search=False, filter_is_diagnostic=False)
        self.assertEqual(len(result), 2)

    def test_missing_scan_dict_treated_as_no_search(self) -> None:
        path = self._write_scan_json([{"resolved_query": "no-scan"}])
        result = search_dataset.load_ecommerce_search_list(path)
        self.assertEqual(result, [])

    def test_directory_input_merges_all_json_files(self) -> None:
        with tempfile.TemporaryDirectory() as scan_dir:
            for i in range(2):
                with open(Path(scan_dir) / f"part-{i}.json", "w", encoding="utf-8") as f:
                    json.dump(
                        [{"resolved_query": f"q{i}", "scan": {"need_search": True, "is_diagnostic": False}}],
                        f,
                    )
            result = search_dataset.load_ecommerce_search_list(scan_dir)
        self.assertEqual({r["resolved_query"] for r in result}, {"q0", "q1"})

    def test_directory_without_json_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as empty_dir:
            self.assertEqual(search_dataset.load_ecommerce_search_list(empty_dir), [])

    def test_invalid_path_raises(self) -> None:
        with self.assertRaises(ValueError):
            search_dataset.load_ecommerce_search_list("/nonexistent/not-a-file-or-dir")

    def test_missing_json_file_raises(self) -> None:
        with self.assertRaises(ValueError):
            search_dataset.load_ecommerce_search_list("/nonexistent/missing.json")


class TestLoadOpenEcommerceDataset(unittest.TestCase):
    def test_builds_search_items_with_dummy_scan(self) -> None:
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8")
        writer = csv.DictWriter(f, fieldnames=["Title", "freq"])
        writer.writeheader()
        writer.writerow({"Title": "USB Cable", "freq": "3"})
        f.close()

        result = search_dataset.load_open_ecommerce_dataset(f.name)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["resolved_query"], "USB Cable")
        self.assertTrue(result[0]["scan"]["need_search"])


if __name__ == "__main__":
    unittest.main()
