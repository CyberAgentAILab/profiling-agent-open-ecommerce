"""Frequent pattern mining: mine co-purchased product combinations with FP-Growth.

Runs outside the main pipeline. Reads the raw Open E-Commerce purchases, keeps titles
bought by >= ``min_users_per_title`` distinct users, groups purchases into (user, order
date) baskets and runs FP-Growth. The only file written is ``frequent_patterns.json``
(under ``out_dir``), which ``predict_user`` reads through ``frequent_patterns_path`` and
profiles each combination as one multi-item purchase. Every statistic (basket sizes,
itemset length distribution, association rules, wall time, peak RSS) goes to the log only.
CPU only; the full dataset takes well under a minute.
"""

import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any

import mlxtend
import numpy as np
import pandas as pd
import scipy
from loguru import logger

from base_agent.config import load_config

from .dataset import build_baskets, encode_baskets, filter_titles_by_min_users, load_purchases
from .model import check_reproduction, export_frequent_patterns, run_association_rules, run_fpgrowth

HYPERPARAMETER_KEYS = (
    "purchases_csv",
    "item_column",
    "user_column",
    "min_users_per_title",
    "basket_keys",
    "min_basket_size",
    "min_support",
    "max_len",
    "skip_rules",
    "rule_metric",
    "rule_min_threshold",
    "rule_min_confidence",
    "rule_min_lift",
    "top_n",
    "export_min_pattern_length",
)

# Key numbers of the full-data run behind the paper (default config, recorded 2026-05-28).
PAPER_RECORDED_RESULTS: dict[str, int] = {
    "baskets_used": 53_890,
    "frequent_itemsets": 4_798,
    "patterns_size_ge2": 541,
    "rules_generated": 1_298,
    "rules_after_filter": 930,
}


def environment_info() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "mlxtend": mlxtend.__version__,
    }


def peak_rss_mb() -> float:
    """Peak resident set size of this process in MB (ru_maxrss is KB on Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def log_comparison_with_paper(observed: dict[str, int]) -> None:
    logger.info("📋 comparison with the paper's full-data run:")
    for key, paper in PAPER_RECORDED_RESULTS.items():
        mark = "✔️" if observed.get(key) == paper else "≠"
        logger.info(f"  {mark} {key}: observed {observed.get(key, 0):,} / paper {paper:,}")


def main() -> None:
    config = load_config()
    out_dir = Path(config["out_dir"])
    hp = {k: config[k] for k in HYPERPARAMETER_KEYS}
    logger.info(f"🧪 environment: {environment_info()}")
    logger.info(f"⚙️ hyperparameters: {hp}")
    t_start = time.perf_counter()

    df, _ = load_purchases(hp["purchases_csv"], hp["item_column"], hp["user_column"])
    df_filtered, _ = filter_titles_by_min_users(df, hp["item_column"], hp["user_column"], hp["min_users_per_title"])
    baskets, basket_stats = build_baskets(
        df_filtered, hp["item_column"], list(hp["basket_keys"]), hp["min_basket_size"]
    )
    del df, df_filtered

    df_encoded, _ = encode_baskets(baskets)
    frequent_itemsets, fpgrowth_stats = run_fpgrowth(
        df_encoded, len(baskets), hp["min_support"], hp["max_len"], hp["top_n"]
    )
    output_path = out_dir / "frequent_patterns.json"
    payload = export_frequent_patterns(frequent_itemsets, output_path, hp["export_min_pattern_length"])

    rule_stats: dict[str, Any] = {}
    if not hp["skip_rules"]:
        _, rule_stats = run_association_rules(
            frequent_itemsets,
            len(baskets),
            hp["rule_metric"],
            hp["rule_min_threshold"],
            hp["rule_min_confidence"],
            hp["rule_min_lift"],
            hp["top_n"],
        )

    log_comparison_with_paper(
        {
            "baskets_used": basket_stats["baskets_used"],
            "frequent_itemsets": fpgrowth_stats["frequent_itemsets"],
            "patterns_size_ge2": fpgrowth_stats["patterns_size_ge2"],
            "rules_generated": rule_stats.get("rules_generated", 0),
            "rules_after_filter": rule_stats.get("rules_after_filter", 0),
        }
    )

    check = None
    reference_json = config.get("reference_json") or ""
    if reference_json:
        check = check_reproduction(payload, reference_json)

    logger.success(
        f"🏁 frequent_pattern_mining done: {len(payload):,} patterns in {time.perf_counter() - t_start:.1f}s, "
        f"peak RSS {peak_rss_mb():.0f} MB -> {output_path}"
    )
    if check is not None and not check["set_equal"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
