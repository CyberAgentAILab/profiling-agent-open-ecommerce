"""Unit tests for the FP-Growth wrappers, the frequent_patterns.json export and the reproduction check."""

import json
import tempfile
import unittest
from pathlib import Path

from frequent_pattern_mining.dataset import encode_baskets
from frequent_pattern_mining.model import (
    check_reproduction,
    export_frequent_patterns,
    run_association_rules,
    run_fpgrowth,
)

BASKETS = [["Bread", "Butter"], ["Bread", "Butter"], ["Bread", "Butter", "Jam"], ["Bread", "Jam"], ["Milk", "Jam"]]


class TestRunFpgrowth(unittest.TestCase):
    def setUp(self) -> None:
        df_encoded, _ = encode_baskets(BASKETS)
        self.frequent_itemsets, self.stats = run_fpgrowth(
            df_encoded, n_baskets=len(BASKETS), min_support=0.4, max_len=3, top_n=5
        )

    def test_counts_and_length_column(self) -> None:
        # support >= 0.4 (2 of 5 baskets): Bread, Butter, Jam, {Bread,Butter}, {Bread,Jam}
        self.assertEqual(self.stats["frequent_itemsets"], 5)
        self.assertEqual(self.stats["itemset_length_distribution"], {"1": 3, "2": 2})
        self.assertEqual(self.stats["patterns_size_ge2"], 2)
        self.assertEqual(self.stats["min_support_count"], 2)
        self.assertIn("length", self.frequent_itemsets.columns)

    def test_export_format_and_reproduction_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "frequent_patterns.json"
            payload = export_frequent_patterns(self.frequent_itemsets, out, min_pattern_length=2)
            self.assertEqual(
                sorted(tuple(d["product_name"] for d in p) for p in payload),
                [("Bread", "Butter"), ("Bread", "Jam")],
            )
            with open(out, encoding="utf-8") as f:
                self.assertEqual(json.load(f), payload)
            # the export is the only file written
            self.assertEqual([p.name for p in Path(tmp).iterdir()], ["frequent_patterns.json"])

            ok = check_reproduction(payload, out)
            self.assertTrue(ok["set_equal"])
            self.assertTrue(ok["order_identical"])

            reference = Path(tmp) / "reference.json"
            reference.write_text(json.dumps([[{"product_name": "Bread"}, {"product_name": "Butter"}]]))
            ng = check_reproduction(payload, reference)
            self.assertFalse(ng["set_equal"])
            self.assertEqual(ng["only_in_reproduced"], [["Bread", "Jam"]])

    def test_rules_are_filtered(self) -> None:
        rules, stats = run_association_rules(
            self.frequent_itemsets,
            n_baskets=len(BASKETS),
            metric="lift",
            min_threshold=1.0,
            min_confidence=0.1,
            min_lift=1.0,
            top_n=5,
        )
        self.assertIsNotNone(rules)
        self.assertGreater(stats["rules_generated"], 0)
        self.assertEqual(stats["rules_after_filter"], len(rules))

    def test_rules_skipped_without_pairs(self) -> None:
        singles = self.frequent_itemsets[self.frequent_itemsets["length"] == 1]
        rules, stats = run_association_rules(singles, len(BASKETS), "lift", 1.0, 0.1, 1.0, 5)
        self.assertIsNone(rules)
        self.assertEqual(stats["rules_after_filter"], 0)


if __name__ == "__main__":
    unittest.main()
