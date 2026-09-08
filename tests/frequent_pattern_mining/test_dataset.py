"""Unit tests for the frequent_pattern_mining data preparation."""

import unittest

import pandas as pd

from frequent_pattern_mining.dataset import build_baskets, encode_baskets, filter_titles_by_min_users


def _purchases() -> pd.DataFrame:
    rows = [
        ("u1", "2024-01-01", "Bread"),
        ("u1", "2024-01-01", "Butter"),
        ("u1", "2024-01-01", "Butter"),  # duplicate inside a basket
        ("u1", "2024-01-02", "Bread"),  # size-1 basket
        ("u2", "2024-01-01", "Bread"),
        ("u2", "2024-01-01", "Butter"),
        ("u3", "2024-01-05", "Bread"),
        ("u3", "2024-01-05", "Rare"),  # bought by a single user
        ("u3", "2024-01-06", None),  # empty title
    ]
    return pd.DataFrame(rows, columns=["Survey ResponseID", "Order Date", "Title"])


class TestFilterTitlesByMinUsers(unittest.TestCase):
    def test_drops_titles_below_threshold(self) -> None:
        df, stats = filter_titles_by_min_users(_purchases(), "Title", "Survey ResponseID", min_users_per_title=2)
        self.assertEqual(set(df["Title"]), {"Bread", "Butter"})
        self.assertEqual(stats["unique_titles_before"], 3)
        self.assertEqual(stats["unique_titles_after"], 2)
        self.assertEqual(stats["rows_after"], 7)


class TestBuildBaskets(unittest.TestCase):
    def test_groups_by_user_and_date_and_dedups(self) -> None:
        baskets, stats = build_baskets(_purchases(), "Title", ["Survey ResponseID", "Order Date"], min_basket_size=2)
        self.assertEqual(baskets, [["Bread", "Butter"], ["Bread", "Butter"], ["Bread", "Rare"]])
        self.assertEqual(stats["baskets_all_sizes"], 4)  # the size-1 basket is counted but dropped
        self.assertEqual(stats["basket_size_distribution"], {"1": 1, "2": 3})
        self.assertEqual(stats["baskets_used"], 3)
        self.assertEqual(stats["rows_after_cleaning"], 8)

    def test_raises_when_nothing_to_mine(self) -> None:
        with self.assertRaises(ValueError):
            build_baskets(_purchases(), "Title", ["Survey ResponseID", "Order Date"], min_basket_size=5)


class TestEncodeBaskets(unittest.TestCase):
    def test_one_hot_shape_and_density(self) -> None:
        df_encoded, stats = encode_baskets([["A", "B"], ["B", "C"]])
        self.assertEqual(stats["shape"], [2, 3])
        self.assertEqual(stats["nonzero"], 4)
        self.assertEqual(list(df_encoded.columns), ["A", "B", "C"])
        self.assertTrue(bool(df_encoded.loc[0, "A"]))
        self.assertFalse(bool(df_encoded.loc[0, "C"]))


if __name__ == "__main__":
    unittest.main()
