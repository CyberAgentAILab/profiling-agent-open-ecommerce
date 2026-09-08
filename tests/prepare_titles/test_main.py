"""Unit tests for prepare_titles.__main__ helpers."""

import unittest

from prepare_titles.__main__ import resolve_out_csv


class TestResolveOutCsv(unittest.TestCase):
    def test_placeholder_follows_threshold(self) -> None:
        self.assertEqual(
            resolve_out_csv("./data/public/open-ecommerce/titles_ge{min_unique_buyers}_buyers.csv", 3),
            "./data/public/open-ecommerce/titles_ge3_buyers.csv",
        )

    def test_plain_path_is_unchanged(self) -> None:
        self.assertEqual(resolve_out_csv("./data/titles.csv", 3), "./data/titles.csv")


if __name__ == "__main__":
    unittest.main()
