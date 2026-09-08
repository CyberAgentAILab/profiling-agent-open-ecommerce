"""Unit tests for the predict_user transaction loader."""

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


dataset = _load_module("predict_user/dataset.py", "predict_user_dataset")


def _transaction(query: str, canonical_name: str | None) -> dict:
    """canonical_name=None builds a record with an empty prediction."""
    prediction: dict = {}
    if canonical_name is not None:
        prediction = {"canonical_name": canonical_name}
    return {"resolved_query": query, "transaction": {"prediction": prediction}}


def _write_json(records: list[dict], directory: str, name: str = "out.json") -> str:
    path = str(Path(directory) / name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f)
    return path


class TestLoadRecoveredTransaction(unittest.TestCase):
    def test_keeps_recovered_and_attaches_items(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = _write_json(
                [
                    _transaction("good", "Canonical Good"),
                    _transaction("blank-name", "   "),
                    _transaction("empty-prediction", None),
                ],
                d,
            )
            results = dataset.load_recovered_transaction(path)
        self.assertEqual([t["resolved_query"] for t in results], ["good"])
        self.assertEqual(
            results[0]["items"],
            [{"product_name": "good", "canonical_name": "Canonical Good"}],
        )

    def test_reverse_order(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = _write_json([_transaction("a", "A"), _transaction("b", "B")], d)
            results = dataset.load_recovered_transaction(path, is_reverse=True)
        self.assertEqual([t["resolved_query"] for t in results], ["b", "a"])

    def test_invalid_path_raises(self) -> None:
        with self.assertRaises(ValueError):
            dataset.load_recovered_transaction("/nonexistent/missing.json")

    def test_allowed_titles_filter(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = _write_json([_transaction("kept", "K"), _transaction("dropped", "D")], d)
            csv_path = str(Path(d) / "allowed.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["Title"])
                writer.writeheader()
                writer.writerow({"Title": "kept"})
            results = dataset.load_recovered_transaction(path, allowed_titles_csv=csv_path)
        self.assertEqual([t["resolved_query"] for t in results], ["kept"])

    def test_scan_only_records_appended_without_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = _write_json([_transaction("already-recovered", "C")], d, "transactions.json")
            scan_dir = Path(d) / "scan"
            scan_dir.mkdir()
            _write_json(
                [
                    {"resolved_query": "already-recovered", "scan": {"need_search": True}},
                    {"resolved_query": "scan-only", "scan": {"need_search": False}},
                ],
                str(scan_dir),
                "scan.json",
            )
            results = dataset.load_recovered_transaction(path, scan_output_path=str(scan_dir))
        queries = [t["resolved_query"] for t in results]
        self.assertEqual(queries, ["already-recovered", "scan-only"])
        scan_only = results[1]
        self.assertTrue(scan_only["transaction"]["from_scan_only"])
        self.assertEqual(scan_only["transaction"]["prediction"]["canonical_name"], "scan-only")

    def test_frequent_patterns_resolved_and_unresolvable_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = _write_json([_transaction("apple", "Apple"), _transaction("pen", "Pen")], d)
            patterns_path = str(Path(d) / "patterns.json")
            with open(patterns_path, "w", encoding="utf-8") as f:
                json.dump(
                    [
                        [{"product_name": "apple"}, {"product_name": "pen"}],
                        [{"product_name": "apple"}, {"product_name": "unknown"}],
                    ],
                    f,
                )
            results = dataset.load_recovered_transaction(path, frequent_patterns_path=patterns_path)
        pattern_records = [t for t in results if t["transaction"].get("from_pattern")]
        self.assertEqual(len(pattern_records), 1)
        self.assertEqual(pattern_records[0]["resolved_query"], "apple + pen")
        self.assertEqual(
            pattern_records[0]["items"],
            [
                {"product_name": "apple", "canonical_name": "Apple"},
                {"product_name": "pen", "canonical_name": "Pen"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
