"""FP-Growth, association rules (log only), the frequent_patterns.json export and the reproduction check."""

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger
from mlxtend.frequent_patterns import association_rules, fpgrowth


def dump_json(obj: object, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _items(itemset: frozenset[str]) -> list[str]:
    return sorted(itemset)


def run_fpgrowth(
    df_encoded: pd.DataFrame,
    n_baskets: int,
    min_support: float,
    max_len: int,
    top_n: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run FP-Growth and return the frequent itemsets (with a ``length`` column) plus statistics."""
    logger.info(
        f"⛏️ FP-Growth: min_support={min_support} ({min_support * 100:.4f}% of {n_baskets:,} baskets "
        f"= {int(n_baskets * min_support)} baskets), max_len={max_len}"
    )
    begin = time.perf_counter()
    frequent_itemsets = fpgrowth(df_encoded, min_support=min_support, use_colnames=True, max_len=max_len)
    elapsed = time.perf_counter() - begin
    frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(len)
    length_dist = frequent_itemsets["length"].value_counts().sort_index()
    logger.info(f"⏱️ FP-Growth took {elapsed:.1f}s, frequent itemsets: {len(frequent_itemsets):,}")
    for length, count in length_dist.items():
        logger.info(f"  length {length}: {count:,}")

    patterns = frequent_itemsets[frequent_itemsets["length"] >= 2].sort_values("support", ascending=False)
    logger.info(f"🧩 patterns of size >= 2: {len(patterns):,}; top {top_n} by support:")
    for i, (_, row) in enumerate(patterns.head(top_n).iterrows(), 1):
        logger.info(f"  {i:>2}. support={row['support']:.4f}: {_items(row['itemsets'])}")

    stats = {
        "min_support": min_support,
        "max_len": max_len,
        "min_support_count": int(n_baskets * min_support),
        "wall_time_sec": elapsed,
        "frequent_itemsets": int(len(frequent_itemsets)),
        "itemset_length_distribution": {str(int(k)): int(v) for k, v in length_dist.items()},
        "patterns_size_ge2": int(len(patterns)),
    }
    return frequent_itemsets, stats


def export_frequent_patterns(
    frequent_itemsets: pd.DataFrame,
    output_path: str | Path,
    min_pattern_length: int,
) -> list[list[dict[str, str]]]:
    """Write patterns of size >= ``min_pattern_length`` in the format predict_user consumes.

    ``[[{"product_name": "A"}, {"product_name": "B"}], ...]`` — one inner list per pattern,
    items sorted, patterns in FP-Growth output order.
    """
    target = frequent_itemsets[frequent_itemsets["length"] >= min_pattern_length]
    payload = [[{"product_name": name} for name in _items(itemset)] for itemset in target["itemsets"]]
    dump_json(payload, output_path)
    logger.info(f"💾 frequent patterns (size >= {min_pattern_length}): {len(payload):,} -> {output_path}")
    return payload


def run_association_rules(
    frequent_itemsets: pd.DataFrame,
    n_baskets: int,
    metric: str,
    min_threshold: float,
    min_confidence: float,
    min_lift: float,
    top_n: int,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    """Derive association rules from the frequent itemsets and filter by confidence / lift.

    The rules are only summarised in the log; nothing downstream consumes them.
    """
    if int((frequent_itemsets["length"] >= 2).sum()) == 0:
        logger.warning("no itemset of size >= 2; skipping association rules (lower min_support to get some)")
        return None, {"rules_generated": 0, "rules_after_filter": 0}

    rules = association_rules(
        frequent_itemsets.drop(columns=["length"]),
        metric=metric,
        min_threshold=min_threshold,
        num_itemsets=n_baskets,
    )
    rules_filtered = rules[(rules["confidence"] >= min_confidence) & (rules["lift"] >= min_lift)].copy()
    rules_filtered = rules_filtered.sort_values("lift", ascending=False)
    logger.info(
        f"📐 rules: {len(rules):,} generated, {len(rules_filtered):,} after "
        f"confidence >= {min_confidence} and lift >= {min_lift}; top {top_n} by lift:"
    )
    for i, (_, rule) in enumerate(rules_filtered.head(top_n).iterrows(), 1):
        logger.info(
            f"  {i:>2}. {_items(rule['antecedents'])} => {_items(rule['consequents'])} "
            f"(support={rule['support']:.4f}, conf={rule['confidence']:.3f}, lift={rule['lift']:.2f})"
        )
    stats = {
        "metric": metric,
        "min_threshold": min_threshold,
        "num_itemsets": n_baskets,
        "rules_generated": int(len(rules)),
        "filter": {"min_confidence": min_confidence, "min_lift": min_lift},
        "rules_after_filter": int(len(rules_filtered)),
    }
    return rules_filtered, stats


def _pattern_set(patterns: list[list[dict[str, str]]]) -> set[tuple[str, ...]]:
    return {tuple(sorted(d["product_name"] for d in p)) for p in patterns}


def check_reproduction(payload: list[list[dict[str, str]]], reference_json: str | Path) -> dict[str, Any]:
    """Compare the exported patterns with a previously produced frequent_patterns.json (result is logged)."""
    with open(reference_json, encoding="utf-8") as f:
        reference = json.load(f)
    got, ref = _pattern_set(payload), _pattern_set(reference)
    result: dict[str, Any] = {
        "reference_json": str(reference_json),
        "n_reference": len(reference),
        "n_reproduced": len(payload),
        "n_common": len(got & ref),
        "only_in_reference": [list(p) for p in sorted(ref - got)],
        "only_in_reproduced": [list(p) for p in sorted(got - ref)],
        "set_equal": got == ref,
        "order_identical": payload == reference,
    }
    if result["set_equal"]:
        logger.success(
            f"✅ reproduction check: pattern set matches the reference "
            f"({len(ref)} patterns, order identical={result['order_identical']})"
        )
    else:
        logger.error(
            f"❌ reproduction check: common {result['n_common']} / reference {len(ref)} / reproduced {len(got)} "
            f"(only in reference {len(result['only_in_reference'])}, "
            f"only in reproduced {len(result['only_in_reproduced'])})"
        )
        for p in result["only_in_reference"][:20]:
            logger.error(f"  only in reference: {p}")
        for p in result["only_in_reproduced"][:20]:
            logger.error(f"  only in reproduced: {p}")
    return result
