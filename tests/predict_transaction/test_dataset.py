"""Unit tests for the predict_transaction search-result loader."""

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


dataset = _load_module("predict_transaction/dataset.py", "predict_transaction_dataset")


def _record(query: str, success_search: bool | None) -> dict:
    """success_search=None builds a record without the search dict."""
    record: dict = {"resolved_query": query}
    if success_search is not None:
        record["search"] = {"success_search": success_search, "search_result": []}
    return record


class TestLoadSearchResult(unittest.TestCase):
    def _write_json(self, records: list[dict]) -> str:
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
        json.dump(records, f)
        f.close()
        return f.name

    def test_keeps_only_successful_searches(self) -> None:
        path = self._write_json(
            [
                _record("hit", True),
                _record("miss", False),
                _record("no-search-dict", None),
            ]
        )
        results = dataset.load_search_result(path, is_reverse=False)
        self.assertEqual([r["resolved_query"] for r in results], ["hit"])

    def test_reverse_order(self) -> None:
        path = self._write_json([_record("a", True), _record("b", True)])
        results = dataset.load_search_result(path, is_reverse=True)
        self.assertEqual([r["resolved_query"] for r in results], ["b", "a"])

    def test_directory_input_merges_files(self) -> None:
        with tempfile.TemporaryDirectory() as in_dir:
            for i in range(2):
                with open(Path(in_dir) / f"part-{i}.json", "w", encoding="utf-8") as f:
                    json.dump([_record(f"q{i}", True)], f)
            results = dataset.load_search_result(in_dir, is_reverse=False)
        self.assertEqual({r["resolved_query"] for r in results}, {"q0", "q1"})

    def test_directory_without_json_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as empty_dir:
            self.assertEqual(dataset.load_search_result(empty_dir, is_reverse=False), [])

    def test_invalid_path_raises(self) -> None:
        with self.assertRaises(ValueError):
            dataset.load_search_result("/nonexistent/not-a-dir", is_reverse=False)

    def test_missing_json_file_raises(self) -> None:
        with self.assertRaises(ValueError):
            dataset.load_search_result("/nonexistent/missing.json", is_reverse=False)


if __name__ == "__main__":
    unittest.main()
