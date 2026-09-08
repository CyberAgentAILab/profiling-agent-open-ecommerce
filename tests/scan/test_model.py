"""Unit tests for ScanSubAgent batch logic (LLM generation stubbed out)."""

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


scan_model = _load_module("scan/model.py", "scan_model")


def _stub_agent(out_dir: str) -> Any:
    """ScanSubAgent without __init__ (no model loading); generate() is stubbed."""
    agent = scan_model.ScanSubAgent.__new__(scan_model.ScanSubAgent)
    agent.json_schema_scan = ["need_search", "is_diagnostic", "diagnostic_for", "reason"]
    agent.yes_tokens = ["yes"]
    agent.out_dir = out_dir
    agent.timestamp = "ts"
    agent.generate = lambda prompt: [
        {"content": json.dumps({"need_search": "yes", "is_diagnostic": None, "diagnostic_for": None, "reason": "r"})}
        for _ in prompt
    ]
    return agent


def _make_inputs(titles: list[str]) -> dict[str, Any]:
    return {
        "noisy_queries": titles,
        "freqs": [1] * len(titles),
        "queries": titles,
        "resolved_queries": [
            {
                "resolved_query": t,
                "noisy_queries": [t],
                "freqs": [1],
                "queries": [t],
                "directions": ["purchase"],
            }
            for t in titles
        ],
    }


class TestScanNeedSearch(unittest.TestCase):
    def _read_results(self, save_path: str) -> list[dict]:
        with open(save_path, encoding="utf-8") as f:
            return json.load(f)

    def test_empty_queries_still_writes_valid_output(self) -> None:
        with tempfile.TemporaryDirectory() as out_dir:
            agent = _stub_agent(out_dir)
            save_path = agent.scan_need_search(batch_size=5, **_make_inputs([]))
            self.assertEqual(self._read_results(save_path), [])

    def test_processes_all_samples_across_batches(self) -> None:
        titles = [f"title-{i}" for i in range(5)]
        with tempfile.TemporaryDirectory() as out_dir:
            agent = _stub_agent(out_dir)
            save_path = agent.scan_need_search(batch_size=2, **_make_inputs(titles))
            results = self._read_results(save_path)
        self.assertEqual([r["resolved_query"] for r in results], titles)
        self.assertTrue(all(r["scan"]["is_json"] for r in results))

    def test_debug_num_sample_caps_processed_samples(self) -> None:
        titles = [f"title-{i}" for i in range(10)]
        with tempfile.TemporaryDirectory() as out_dir:
            agent = _stub_agent(out_dir)
            save_path = agent.scan_need_search(batch_size=2, debug_num_sample=3, **_make_inputs(titles))
            results = self._read_results(save_path)
        self.assertEqual(len(results), 3)

    def test_job_id_appears_in_output_filename(self) -> None:
        with tempfile.TemporaryDirectory() as out_dir:
            agent = _stub_agent(out_dir)
            save_path = agent.scan_need_search(batch_size=2, job_id="0-3", **_make_inputs(["a"]))
        self.assertIn("0-3", Path(save_path).name)


if __name__ == "__main__":
    unittest.main()
