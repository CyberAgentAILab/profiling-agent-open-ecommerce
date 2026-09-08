"""Unit tests for the scan dataset loader."""

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


scan_dataset = _load_module("scan/dataset.py", "scan_dataset")


class TestLoadTitlesForEcommerce(unittest.TestCase):
    def _write_csv(self, rows: list[str]) -> str:
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8")
        f.write("\n".join(rows))
        f.close()
        return f.name

    def test_basic_load(self) -> None:
        path = self._write_csv(
            [
                "Title,freq,n_unique_buyers",
                "USB Cable,10,8",
                "Coffee Beans,3,3",
            ]
        )
        noisy, freqs, queries, resolved = scan_dataset.load_titles_for_ecommerce(path)
        self.assertEqual(noisy, ["USB Cable", "Coffee Beans"])
        self.assertEqual(freqs, [10, 3])
        self.assertEqual(queries, noisy)
        self.assertEqual(resolved[0]["resolved_query"], "USB Cable")
        self.assertEqual(resolved[0]["directions"], ["purchase"])

    def test_missing_title_column_raises(self) -> None:
        path = self._write_csv(["name,freq", "x,1"])
        with self.assertRaises(ValueError):
            scan_dataset.load_titles_for_ecommerce(path)

    def test_missing_freq_column_defaults_to_one(self) -> None:
        path = self._write_csv(["Title", "USB Cable"])
        _, freqs, _, resolved = scan_dataset.load_titles_for_ecommerce(path)
        self.assertEqual(freqs, [1])
        self.assertEqual(resolved[0]["freqs"], [1])

    def test_reverse_order(self) -> None:
        path = self._write_csv(
            [
                "Title,freq,n_unique_buyers",
                "A,1,1",
                "B,2,2",
            ]
        )
        noisy, freqs, _, _ = scan_dataset.load_titles_for_ecommerce(path, is_reverse=True)
        self.assertEqual(noisy, ["B", "A"])
        self.assertEqual(freqs, [2, 1])


if __name__ == "__main__":
    unittest.main()
