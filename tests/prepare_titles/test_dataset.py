"""Unit tests for the prepare_titles dataset utilities."""

import csv
import importlib.util
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


dataset = _load_module("prepare_titles/dataset.py", "prepare_titles_dataset")


def _write_purchases_csv(rows: list[dict[str, str]]) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=["Title", "Survey ResponseID"])
    writer.writeheader()
    writer.writerows(rows)
    f.close()
    return f.name


class TestFilterTitlesByBuyers(unittest.TestCase):
    def test_aggregates_freq_and_unique_buyers(self) -> None:
        path = _write_purchases_csv(
            [
                {"Title": "USB Cable", "Survey ResponseID": "u1"},
                {"Title": "USB Cable", "Survey ResponseID": "u2"},
                {"Title": "USB Cable", "Survey ResponseID": "u1"},
            ]
        )
        records = dataset.filter_titles_by_buyers(path, min_unique_buyers=1)
        self.assertEqual(records, [{"Title": "USB Cable", "freq": 3, "n_unique_buyers": 2}])

    def test_threshold_filters_titles(self) -> None:
        path = _write_purchases_csv(
            [
                {"Title": "Popular", "Survey ResponseID": "u1"},
                {"Title": "Popular", "Survey ResponseID": "u2"},
                {"Title": "Niche", "Survey ResponseID": "u1"},
            ]
        )
        records = dataset.filter_titles_by_buyers(path, min_unique_buyers=2)
        self.assertEqual([r["Title"] for r in records], ["Popular"])

    def test_sorted_by_unique_buyers_desc(self) -> None:
        path = _write_purchases_csv(
            [
                {"Title": "A", "Survey ResponseID": "u1"},
                {"Title": "B", "Survey ResponseID": "u1"},
                {"Title": "B", "Survey ResponseID": "u2"},
                {"Title": "C", "Survey ResponseID": "u1"},
                {"Title": "C", "Survey ResponseID": "u2"},
                {"Title": "C", "Survey ResponseID": "u3"},
            ]
        )
        records = dataset.filter_titles_by_buyers(path, min_unique_buyers=1)
        self.assertEqual([r["Title"] for r in records], ["C", "B", "A"])

    def test_skips_rows_with_empty_title_or_user(self) -> None:
        path = _write_purchases_csv(
            [
                {"Title": "", "Survey ResponseID": "u1"},
                {"Title": "Orphan", "Survey ResponseID": ""},
                {"Title": "Kept", "Survey ResponseID": "u1"},
            ]
        )
        records = dataset.filter_titles_by_buyers(path, min_unique_buyers=1)
        self.assertEqual([r["Title"] for r in records], ["Kept"])

    def test_empty_input_returns_empty_list(self) -> None:
        path = _write_purchases_csv([])
        self.assertEqual(dataset.filter_titles_by_buyers(path, min_unique_buyers=1), [])


class TestSaveTitlesCsv(unittest.TestCase):
    def test_writes_header_and_rows(self) -> None:
        records = [{"Title": "USB Cable", "freq": 3, "n_unique_buyers": 2}]
        with tempfile.TemporaryDirectory() as out_dir:
            saved = dataset.save_titles_csv(records, str(Path(out_dir) / "titles.csv"))
            with open(saved, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows, [{"Title": "USB Cable", "freq": "3", "n_unique_buyers": "2"}])

    def test_returns_absolute_path_and_creates_parent_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as out_dir:
            out_path = str(Path(out_dir) / "nested" / "dir" / "titles.csv")
            saved = dataset.save_titles_csv([], out_path)
            self.assertTrue(Path(saved).is_absolute())
            self.assertTrue(Path(saved).exists())

    def test_non_ascii_title_roundtrip(self) -> None:
        records = [{"Title": "Café au Lait", "freq": 1, "n_unique_buyers": 1}]
        with tempfile.TemporaryDirectory() as out_dir:
            saved = dataset.save_titles_csv(records, str(Path(out_dir) / "titles.csv"))
            with open(saved, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["Title"], "Café au Lait")


if __name__ == "__main__":
    unittest.main()
