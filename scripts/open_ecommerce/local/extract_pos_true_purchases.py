"""Extract raw purchase histories for users whose ground_truth is positive
for each of the 10 attributes evaluated in the judge_user_attribute baseline.

Inputs
------
- results/open_ecommerce/judge_user_attribute/v1/output-*-baseline.json
    Latest baseline run is picked automatically. Each record carries a
    `ground_truth` block (bool | None per attribute).
- data/public/open-ecommerce/amazon-purchases.csv
    Raw transaction log.

Output
------
data/public/open-ecommerce/work/judge_user_attribute_pos_true/<attribute>.csv
with columns [Order Date, Title, Category, Survey ResponseID], plus a
_summary.csv aggregating user/row counts per attribute.
"""

import json
import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
EVAL_DIR = REPO_ROOT / "results/open_ecommerce/judge_user_attribute/v1"
EVAL_GLOB = "output-*-baseline.json"
PURCHASES_CSV = REPO_ROOT / "data/public/open-ecommerce/amazon-purchases.csv"
OUT_DIR = REPO_ROOT / "data/public/open-ecommerce/work/judge_user_attribute_pos_true"


def latest_eval_json(eval_dir: Path, pattern: str) -> Path:
    # Filenames embed a sortable `YYYY-MM-DD_HH-MM-SS` timestamp, so lexicographic max == newest run.
    candidates = sorted(eval_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No eval JSON matching {pattern} under {eval_dir}")
    return candidates[-1]


ATTRIBUTE_KEYS = [
    "cigarettes",
    "marijuana",
    "alcohol",
    "diabetes",
    "wheelchair",
    "had_child",
    "became_pregnant",
    "moved",
    "lost_job",
    "divorce",
]
KEEP_COLS = ["Order Date", "Title", "Category", "Survey ResponseID"]


def collect_pos_true_users(eval_json: Path) -> dict[str, set[str]]:
    with eval_json.open() as f:
        payload = json.load(f)
    pos: dict[str, set[str]] = {k: set() for k in ATTRIBUTE_KEYS}
    for rec in payload["predictions"]:
        uid = rec["user_id"]
        gt = rec.get("ground_truth", {})
        for key in ATTRIBUTE_KEYS:
            # judge_user_attribute writes ground_truth as bool | None (True/False/None).
            # Accept the canonical True, plus legacy "yes" strings for forward compat.
            val = gt.get(key)
            if val is True or str(val).strip().lower() == "yes":
                pos[key].add(uid)
    return pos


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    eval_json = latest_eval_json(EVAL_DIR, EVAL_GLOB)
    print(f"Using eval JSON: {eval_json.name}")
    pos_users = collect_pos_true_users(eval_json)
    for k in ATTRIBUTE_KEYS:
        print(f"  n_pos_true[{k:>16}] = {len(pos_users[k])}")

    print(f"\nLoading purchases: {PURCHASES_CSV}")
    purchases = pd.read_csv(
        PURCHASES_CSV,
        usecols=KEEP_COLS,
        dtype={"Title": "string", "Category": "string", "Survey ResponseID": "string"},
        parse_dates=["Order Date"],
    )
    print(f"  rows={len(purchases):,} users={purchases['Survey ResponseID'].nunique():,}")

    purchases = purchases.dropna(subset=["Order Date", "Survey ResponseID", "Title"])
    purchases = purchases[purchases["Title"].str.strip() != ""]
    print(f"  after Title filter: rows={len(purchases):,} users={purchases['Survey ResponseID'].nunique():,}")

    summary_rows = []
    for attribute in ATTRIBUTE_KEYS:
        uids = pos_users[attribute]
        if not uids:
            print(f"\n[{attribute}] no positive-ground-truth users, skipping")
            continue

        sub = purchases[purchases["Survey ResponseID"].isin(uids)]
        n_rows_raw = len(sub)
        print(
            f"\n[{attribute}] users={len(uids):,} "
            f"purchase_rows={n_rows_raw:,} "
            f"covered_users={sub['Survey ResponseID'].nunique():,}"
        )

        out_path = OUT_DIR / f"{attribute}.csv"

        sub_sorted = sub.sort_values(["Survey ResponseID", "Order Date"])[KEEP_COLS]
        sub_sorted.to_csv(out_path, index=False)

        summary_rows.append(
            {
                "attribute": attribute,
                "n_users": int(sub["Survey ResponseID"].nunique()),
                "n_rows": int(len(sub)),
                "file": out_path.name,
            }
        )
        print(f"    rows={n_rows_raw:,} users={sub['Survey ResponseID'].nunique():,} -> {out_path.name}")

    summary = pd.DataFrame(summary_rows).sort_values(["attribute"])
    summary_path = OUT_DIR / "_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSummary written to {summary_path} ({len(summary)} rows)")


if __name__ == "__main__":
    # silence the pandas re-import warning when run twice in the same shell
    _ = re
    main()
