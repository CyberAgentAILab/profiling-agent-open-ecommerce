#!/usr/bin/env python3
"""Aggregate the predict_user JSONs of a run (default: the recorded sample run,
results/open_ecommerce/sample/outputs/user/v4; pass another <run_dir>/outputs/user/v4
as the first argument) and count the free-form attributes
(free_description_attributes) generated for each of the demographic /
psycho_behavioral / life_event categories.
"""
import glob
import json
import os
import sys
from collections import defaultdict

V4_DIR = sys.argv[1] if len(sys.argv) > 1 else "results/open_ecommerce/sample/outputs/user/v4"
CATEGORIES = {
    "demographic": "user_attribute_demographic",
    "psycho_behavioral": "user_attribute_psycho_behavioral",
    "life_event": "user_attribute_life_event",
}


def main():
    files = sorted(glob.glob(os.path.join(V4_DIR, "*.json")))

    total_records = 0          # total number of records
    predicted_records = 0      # records where prediction ran (have a free_desc entry)
    # Per category: total free-form attributes and non-empty record counts
    free_count = defaultdict(int)
    records_with_free = defaultdict(int)
    unique_free = defaultdict(set)

    for path in files:
        with open(path) as fp:
            data = json.load(fp)
        for rec in data:
            total_records += 1
            user = rec.get("user", {}) or {}
            saw_free_entry = False
            for label, key in CATEGORIES.items():
                cat = user.get(key, {}) or {}
                attrs = cat.get("consumer_attributes", []) or []
                rec_has = False
                for item in attrs:
                    if item.get("category") != "free_description_attributes":
                        continue
                    saw_free_entry = True
                    # `attribute` may be a str, a list, or None (schema allows null).
                    vals = item.get("attribute") or []
                    if isinstance(vals, str):
                        vals = [vals]
                    vals = [v for v in vals if v and v != "unknown"]
                    for v in vals:
                        free_count[label] += 1
                        unique_free[label].add(v)
                        rec_has = True
                if rec_has:
                    records_with_free[label] += 1
            if saw_free_entry:
                predicted_records += 1

    print(f"files scanned: {len(files)}")
    print(f"total records: {total_records}")
    print(f"predicted records (with free_desc entry): {predicted_records}")
    print()
    print(f"{'category':<20}{'total':>12}{'unique':>12}{'records w/ any':>16}{'avg/record':>14}")
    print("-" * 74)
    grand = 0
    for label in CATEGORIES:
        c = free_count[label]
        grand += c
        avg = c / predicted_records if predicted_records else 0
        print(f"{label:<20}{c:>12,}{len(unique_free[label]):>12,}"
              f"{records_with_free[label]:>16,}{avg:>14.3f}")
    print("-" * 74)
    print(f"{'total':<20}{grand:>12,}")


if __name__ == "__main__":
    main()
