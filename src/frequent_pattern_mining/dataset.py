"""Data preparation for frequent pattern mining over the Open E-Commerce purchases.

Purchases -> title filter (min distinct buyers) -> (user, order date) baskets -> one-hot
sparse matrix for FP-Growth.
"""

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from mlxtend.preprocessing import TransactionEncoder


def load_purchases(csv_path: str, item_column: str, user_column: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the raw purchases CSV and return it with basic statistics."""
    logger.info(f"📚 loading purchases: {csv_path}")
    df = pd.read_csv(csv_path)
    stats: dict[str, Any] = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "unique_users": int(df[user_column].nunique()),
        "unique_titles": int(df[item_column].nunique()),
    }
    if "Category" in df.columns:
        stats["unique_categories"] = int(df["Category"].nunique())
        stats["category_missing_rate"] = float(df["Category"].isna().mean())
    logger.info(f"📊 rows={stats['rows']:,} users={stats['unique_users']:,} titles={stats['unique_titles']:,}")
    return df, stats


def filter_titles_by_min_users(
    df: pd.DataFrame,
    item_column: str,
    user_column: str,
    min_users_per_title: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Keep only titles bought by at least ``min_users_per_title`` distinct users."""
    users_per_title = df.groupby(item_column)[user_column].nunique()
    valid_titles = users_per_title[users_per_title >= min_users_per_title].index
    df_filtered = df[df[item_column].isin(valid_titles)].copy()

    n_before = int(df[item_column].nunique())
    n_after = int(len(valid_titles))
    reduction = (1 - n_after / n_before) * 100 if n_before else 0.0
    logger.info(
        f"🎯 title filter (>= {min_users_per_title} buyers): titles {n_before:,} → {n_after:,} "
        f"({reduction:.1f}% removed), rows {len(df):,} → {len(df_filtered):,}"
    )
    stats = {
        "min_users_per_title": min_users_per_title,
        "unique_titles_before": n_before,
        "unique_titles_after": n_after,
        "rows_before": int(len(df)),
        "rows_after": int(len(df_filtered)),
    }
    return df_filtered, stats


def build_baskets(
    df: pd.DataFrame,
    item_column: str,
    basket_keys: list[str],
    min_basket_size: int,
) -> tuple[list[list[str]], dict[str, Any]]:
    """Group purchases into baskets keyed by ``basket_keys`` and keep baskets of size >= ``min_basket_size``.

    Titles inside a basket are de-duplicated and sorted so the encoding is deterministic.
    """
    df_clean = df.dropna(subset=[item_column])
    df_clean = df_clean[df_clean[item_column].astype(str).str.strip() != ""]
    logger.info(f"🧹 rows after dropping empty titles: {len(df_clean):,}")

    grouped = df_clean.groupby(basket_keys)[item_column].apply(lambda titles: sorted(set(titles)))
    all_baskets: list[list[str]] = grouped.tolist()
    size_counts = Counter(len(b) for b in all_baskets)
    logger.info(f"🧺 baskets (all sizes): {len(all_baskets):,}")
    for size in sorted(size_counts)[:10]:
        share = size_counts[size] / len(all_baskets) * 100 if all_baskets else 0.0
        logger.info(f"  size {size}: {size_counts[size]:>7,} ({share:5.2f}%)")

    baskets = [b for b in all_baskets if len(b) >= min_basket_size]
    if not baskets:
        raise ValueError(f"no basket of size >= {min_basket_size}; nothing to mine")
    sizes = [len(b) for b in baskets]
    logger.info(f"🧺 baskets used (size >= {min_basket_size}): {len(baskets):,}")
    logger.info(f"  mean size {np.mean(sizes):.2f}, max size {max(sizes)}")
    stats = {
        "rows_after_cleaning": int(len(df_clean)),
        "unique_titles_after_cleaning": int(df_clean[item_column].nunique()),
        "baskets_all_sizes": int(len(all_baskets)),
        "basket_size_distribution": {str(k): int(v) for k, v in sorted(size_counts.items())},
        "min_basket_size": min_basket_size,
        "baskets_used": int(len(baskets)),
        "mean_basket_size": float(np.mean(sizes)),
        "max_basket_size": int(max(sizes)),
    }
    return baskets, stats


def encode_baskets(baskets: list[list[str]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """One-hot encode the baskets into a sparse DataFrame (rows = baskets, columns = titles)."""
    encoder = TransactionEncoder()
    matrix = encoder.fit(baskets).transform(baskets, sparse=True)
    df_encoded = pd.DataFrame.sparse.from_spmatrix(matrix, columns=encoder.columns_)
    density = float(df_encoded.sparse.density)
    nonzero = int(round(density * df_encoded.size))
    logger.info(f"🔢 encoded shape={df_encoded.shape} density={density:.5f} nonzero={nonzero:,}")
    stats = {"shape": [int(s) for s in df_encoded.shape], "density": density, "nonzero": nonzero}
    return df_encoded, stats
