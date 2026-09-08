from typing import Any

import pandas as pd
from loguru import logger


def load_titles_for_ecommerce(
    path_query: str,
    is_reverse: bool = False,
) -> tuple[list[str], list[int], list[str], list[dict[str, Any]]]:
    """Load Amazon product titles for the open-ecommerce pipeline.

    Expects CSV with columns: Title, freq, n_unique_buyers.
    Returns a 4-tuple (noisy_queries, freqs, queries, resolved_queries) so the
    downstream ScanSubAgent code path is shared.
    """
    df = pd.read_csv(path_query)
    if "Title" not in df.columns:
        raise ValueError(f"Expected 'Title' column in {path_query}, got {list(df.columns)}")

    titles = df["Title"].astype(str).tolist()
    freqs = df["freq"].astype(int).tolist() if "freq" in df.columns else [1] * len(titles)

    if is_reverse:
        titles = titles[::-1]
        freqs = freqs[::-1]

    resolved_queries: list[dict[str, Any]] = []
    for title, freq in zip(titles, freqs, strict=True):
        resolved_queries.append(
            {
                "resolved_query": title,
                "noisy_queries": [title],
                "freqs": [freq],
                "queries": [title],
                "directions": ["purchase"],
            }
        )

    logger.info(f"🛒 ecommerce: loaded {len(titles)} titles from {path_query}")
    return titles, freqs, titles, resolved_queries
