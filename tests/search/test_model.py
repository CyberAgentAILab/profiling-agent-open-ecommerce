"""Unit tests for WebRetriever (search engines and network stubbed out)."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest import mock


def _load_module(rel_path: str, name: str) -> ModuleType:
    path = Path(__file__).parent.parent.parent / "src" / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


search_model = _load_module("search/model.py", "search_model")


def _stub_retriever(results: list[dict[str, str]]) -> Any:
    """WebRetriever without __init__ (no cache / network); searcher is stubbed."""
    retriever = search_model.WebRetriever.__new__(search_model.WebRetriever)
    retriever.use_cache = False
    retriever.search_cache = None
    retriever.query_suffix = ""
    retriever.searcher = lambda query: results
    return retriever


def _search_item(query: str, need_search: bool) -> dict:
    return {"resolved_query": query, "scan": {"need_search": need_search}}


class TestRunSearch(unittest.TestCase):
    def _read_results(self, save_path: str) -> list[dict]:
        with open(save_path, encoding="utf-8") as f:
            return json.load(f)

    def test_empty_search_list_still_writes_valid_output(self) -> None:
        with tempfile.TemporaryDirectory() as out_dir:
            retriever = _stub_retriever([])
            save_path = retriever.run_search(search_list=[], out_dir=out_dir, timestamp="ts")
            self.assertEqual(self._read_results(save_path), [])

    def test_searches_only_when_need_search(self) -> None:
        hit = [{"retrieval_context": "ctx", "title": "t", "body": "b", "url": "u"}]
        with tempfile.TemporaryDirectory() as out_dir:
            retriever = _stub_retriever(hit)
            save_path = retriever.run_search(
                search_list=[_search_item("searched", True), _search_item("skipped", False)],
                out_dir=out_dir,
                timestamp="ts",
            )
            results = self._read_results(save_path)
        self.assertTrue(results[0]["search"]["searched"])
        self.assertTrue(results[0]["search"]["success_search"])
        self.assertEqual(results[0]["search"]["search_result"], hit)
        self.assertFalse(results[1]["search"]["searched"])
        self.assertEqual(results[1]["search"]["search_result"], [])

    def test_empty_search_result_marks_failure(self) -> None:
        with tempfile.TemporaryDirectory() as out_dir:
            retriever = _stub_retriever([])
            save_path = retriever.run_search(
                search_list=[_search_item("no-hits", True)],
                out_dir=out_dir,
                timestamp="ts",
            )
            results = self._read_results(save_path)
        self.assertTrue(results[0]["search"]["searched"])
        self.assertFalse(results[0]["search"]["success_search"])

    def test_num_debug_caps_processed_items(self) -> None:
        with tempfile.TemporaryDirectory() as out_dir:
            retriever = _stub_retriever([])
            save_path = retriever.run_search(
                search_list=[_search_item(f"q{i}", False) for i in range(5)],
                out_dir=out_dir,
                timestamp="ts",
                num_debug=2,
            )
            results = self._read_results(save_path)
        self.assertEqual(len(results), 2)

    def test_query_suffix_appended_to_search_query(self) -> None:
        seen_queries: list[str] = []

        def recording_searcher(query: str) -> list[dict[str, str]]:
            seen_queries.append(query)
            return []

        with tempfile.TemporaryDirectory() as out_dir:
            retriever = _stub_retriever([])
            retriever.query_suffix = " Amazon product"
            retriever.searcher = recording_searcher
            retriever.run_search(
                search_list=[_search_item("USB Cable", True)],
                out_dir=out_dir,
                timestamp="ts",
            )
        self.assertEqual(seen_queries, ["USB Cable Amazon product"])


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        return


class TestApiSearchResponseHandling(unittest.TestCase):
    def _api_retriever(self) -> Any:
        retriever = search_model.WebRetriever.__new__(search_model.WebRetriever)
        retriever.serper_url = "https://example.invalid/search"
        retriever.serp_url = "https://example.invalid/search"
        retriever.serper_api_token = "dummy"  # noqa: S105
        retriever.serp_api_token = "dummy"  # noqa: S105
        retriever.timeout = 1
        return retriever

    def test_serper_non_dict_response_returns_empty(self) -> None:
        retriever = self._api_retriever()
        with mock.patch.object(search_model.requests, "request", return_value=_FakeResponse("[1, 2]")):
            self.assertEqual(retriever.serper_search(query="q"), [])

    def test_serp_non_dict_response_returns_empty(self) -> None:
        retriever = self._api_retriever()
        with mock.patch.object(search_model.requests, "get", return_value=_FakeResponse("[1, 2]")):
            self.assertEqual(retriever.serp_search(query="q"), [])

    def test_serper_dict_response_builds_contexts(self) -> None:
        retriever = self._api_retriever()
        body = json.dumps({"organic": [{"title": "T", "snippet": "S", "link": "https://example.com"}]})
        with mock.patch.object(search_model.requests, "request", return_value=_FakeResponse(body)):
            results = retriever.serper_search(query="q")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "T")
        self.assertEqual(results[0]["url"], "https://example.com")


if __name__ == "__main__":
    unittest.main()
