"""Unit tests for TransactionSubAgent batch logic (LLM generation stubbed out)."""

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


transaction_model = _load_module("predict_transaction/model.py", "predict_transaction_model")

SCHEMA = ["reasoning", "canonical_name", "category_inferred", "use_case", "household_signals"]


def _stub_agent(out_dir: str, canonical_name: str) -> Any:
    """TransactionSubAgent without __init__ (no model loading); generate() is stubbed."""
    agent = transaction_model.TransactionSubAgent.__new__(transaction_model.TransactionSubAgent)
    agent.json_schema_transaction = SCHEMA
    agent.out_dir = out_dir
    agent.timestamp = "ts"
    prediction = {
        "reasoning": "r",
        "canonical_name": canonical_name,
        "category_inferred": "c",
        "use_case": "u",
        "household_signals": "h",
    }
    agent.generate = lambda prompt: [
        {"content": json.dumps(prediction), "input_tokens": 1, "output_tokens": 2} for _ in prompt
    ]
    return agent


def _search_item(query: str, contexts: list[str] | None = None) -> dict:
    return {
        "resolved_query": query,
        "search": {
            "success_search": True,
            "search_result": [{"retrieval_context": c} for c in (contexts or [])],
        },
    }


class TestBuildContext(unittest.TestCase):
    def test_joins_query_and_retrieval_contexts(self) -> None:
        with tempfile.TemporaryDirectory() as out_dir:
            agent = _stub_agent(out_dir, "USB Cable")
            context = agent._build_context(_search_item("usb cable 2m", ["ctx1", "ctx2"]))
        self.assertIn("usb cable 2m", context)
        self.assertIn("ctx1\nctx2", context)

    def test_missing_search_dict_yields_empty_context(self) -> None:
        with tempfile.TemporaryDirectory() as out_dir:
            agent = _stub_agent(out_dir, "USB Cable")
            context = agent._build_context({"resolved_query": "q"})
        self.assertIn("Search results", context)


class TestPredictTransactionAttribute(unittest.TestCase):
    def _read_results(self, save_path: str) -> list[dict]:
        with open(save_path, encoding="utf-8") as f:
            return json.load(f)

    def test_empty_input_still_writes_valid_output(self) -> None:
        with tempfile.TemporaryDirectory() as out_dir:
            agent = _stub_agent(out_dir, "USB Cable")
            save_path = agent.predict_transaction_attribute(search_results=[], batch_size=2, debug_num_sample=None)
            self.assertEqual(self._read_results(save_path), [])

    def test_attaches_prediction_and_success_flag(self) -> None:
        with tempfile.TemporaryDirectory() as out_dir:
            agent = _stub_agent(out_dir, "USB Cable")
            save_path = agent.predict_transaction_attribute(
                search_results=[_search_item("q1"), _search_item("q2")],
                batch_size=2,
                debug_num_sample=None,
            )
            results = self._read_results(save_path)
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertEqual(r["transaction"]["prediction"]["canonical_name"], "USB Cable")
            self.assertTrue(r["transaction"]["prediction"]["is_json"])
            self.assertTrue(r["transaction"]["success_recovery"])

    def test_blank_canonical_name_marks_recovery_failure(self) -> None:
        with tempfile.TemporaryDirectory() as out_dir:
            agent = _stub_agent(out_dir, "   ")
            save_path = agent.predict_transaction_attribute(
                search_results=[_search_item("q1")], batch_size=2, debug_num_sample=None
            )
            results = self._read_results(save_path)
        self.assertFalse(results[0]["transaction"]["success_recovery"])

    def test_debug_num_sample_caps_processed_samples(self) -> None:
        with tempfile.TemporaryDirectory() as out_dir:
            agent = _stub_agent(out_dir, "USB Cable")
            save_path = agent.predict_transaction_attribute(
                search_results=[_search_item(f"q{i}") for i in range(10)],
                batch_size=2,
                debug_num_sample=3,
            )
            results = self._read_results(save_path)
        self.assertEqual(len(results), 3)


if __name__ == "__main__":
    unittest.main()
