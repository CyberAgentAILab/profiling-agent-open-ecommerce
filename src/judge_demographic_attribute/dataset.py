import json
from collections import Counter
from pathlib import Path

import pandas as pd
from loguru import logger

# The four demographic signal categories evaluated by this module. Each record
# in the demographic DB carries one entry per category under
# user.user_attribute_demographic.consumer_attributes.
DEMOGRAPHIC_CATEGORIES = (
    "signal:age-bin",
    "signal:gender",
    "signal:income-bin",
    "signal:education",
)

# Sentinel value the predictor emits when a product carries no signal for a
# category. Such entries are dropped so they never pollute the per-user vote.
_UNKNOWN = "unknown"


def _normalize_attribute(value: object) -> list[str]:
    """Normalize a consumer_attribute value into a flat list of token strings.

    age-bin / income-bin are stored as a list of bins (e.g. ['25-34', '35-44']),
    while gender / education are single tokens (e.g. 'female', 'bachelor'). Both
    shapes are flattened to a list[str] so the per-user aggregation is uniform.
    The 'unknown' sentinel is dropped here.
    """
    if isinstance(value, str):
        return [] if value == _UNKNOWN else [value]
    if isinstance(value, list):
        return [v for v in value if isinstance(v, str) and v and v != _UNKNOWN]
    return []


def load_demographic_db(db_path: str) -> dict[str, dict[str, list[str]]]:
    """Load the tag/demographic DB and build {Title -> {category -> [values]}}.

    The DB is the per-Title output where each record has `resolved_query` (the
    Amazon Title) and a nested `user.user_attribute_demographic.consumer_attributes`
    list. Only the four DEMOGRAPHIC_CATEGORIES are retained, and 'unknown' values
    are dropped (see `_normalize_attribute`). Titles left with no usable signal in
    any category are omitted.
    """
    db_path_abs = Path(db_path).resolve()
    with open(db_path_abs) as f:
        records = json.load(f)

    title_to_demo: dict[str, dict[str, list[str]]] = {}
    for record in records:
        title = record.get("resolved_query")
        if not isinstance(title, str) or not title:
            continue
        user = record.get("user") or {}
        demo = user.get("user_attribute_demographic") or {}
        consumer_attributes = demo.get("consumer_attributes")
        if not isinstance(consumer_attributes, list):
            continue

        per_category: dict[str, list[str]] = {}
        for item in consumer_attributes:
            if not isinstance(item, dict):
                continue
            category = item.get("category")
            if category not in DEMOGRAPHIC_CATEGORIES:
                continue
            values = _normalize_attribute(item.get("attribute"))
            if values:
                per_category[category] = values

        if per_category:
            title_to_demo[title] = per_category

    n_titles = len(title_to_demo)
    coverage = {cat: sum(1 for d in title_to_demo.values() if cat in d) for cat in DEMOGRAPHIC_CATEGORIES}
    logger.info(f"📚 loaded demographic DB: {len(records)} records, {n_titles} titles with ≥1 demographic signal")
    logger.info(f"    per-category title coverage: {coverage}")
    return title_to_demo


def aggregate_demographics_per_user(
    purchases: pd.DataFrame,
    title_to_demo: dict[str, dict[str, list[str]]],
    log_first_n: int = 3,
) -> dict[str, dict[str, Counter]]:
    """For each user (Survey ResponseID), tally demographic signal values over
    every purchased product that carries a signal.

    Each purchase row contributes one vote, so a product bought N times is
    weighted N× (the same purchase-count weighting judge_user_attribute uses).
    Returns {user_id -> {category -> Counter(value -> count)}}; categories with
    no observed (non-unknown) value for a user are simply absent.

    The first `log_first_n` users are logged for auditability.
    """
    user_demo: dict[str, dict[str, Counter]] = {}
    logged = 0

    for user_id, group in purchases.groupby("Survey ResponseID"):
        per_category: dict[str, Counter] = {cat: Counter() for cat in DEMOGRAPHIC_CATEGORIES}
        n_titles_with_signal = 0
        for title in group["Title"].tolist():
            demo = title_to_demo.get(title)
            if not demo:
                continue
            n_titles_with_signal += 1
            for category, values in demo.items():
                for value in values:
                    per_category[category][value] += 1

        # Drop empty categories so downstream code can test presence directly.
        per_category = {cat: counter for cat, counter in per_category.items() if counter}
        user_demo[str(user_id)] = per_category

        if logged < log_first_n:
            summary = {cat: counter.most_common(3) for cat, counter in per_category.items()}
            # Survey ResponseIDs are intentionally not logged (personal identifiers).
            logger.info(
                f"🧬 user[{logged + 1}/{log_first_n}] "
                f"purchases={len(group)} titles_with_signal={n_titles_with_signal}"
            )
            logger.info(f"    top values per category: {summary}")
            logged += 1

    n_with_any = sum(1 for d in user_demo.values() if d)
    logger.info(f"🧮 aggregated demographics for {len(user_demo)} users ({n_with_any} with ≥1 non-unknown signal)")
    return user_demo


# ── relocation (life-event) ───────────────────────────────────────────────────

_RELOCATION_CATEGORY = "signal:recent-relocation"


def load_relocation_db(db_path: str) -> dict[str, bool]:
    """Load the tag DB and build {Title -> is_positive_relocation}.

    Each title is mapped to True if its `signal:recent-relocation` attribute is
    'positive', False if it is 'unknown'. Titles with no relocation entry are
    absent from the returned dict and treated as no signal at lookup time.
    """
    db_path_abs = Path(db_path).resolve()
    with open(db_path_abs) as f:
        records = json.load(f)

    title_to_reloc: dict[str, bool] = {}
    n_positive = 0
    for record in records:
        title = record.get("resolved_query")
        if not isinstance(title, str) or not title:
            continue
        user = record.get("user") or {}
        le = user.get("user_attribute_life_event") or {}
        consumer_attributes = le.get("consumer_attributes")
        if not isinstance(consumer_attributes, list):
            continue
        for item in consumer_attributes:
            if not isinstance(item, dict):
                continue
            if item.get("category") != _RELOCATION_CATEGORY:
                continue
            is_positive = item.get("attribute") == "positive"
            title_to_reloc[title] = is_positive
            if is_positive:
                n_positive += 1
            break

    n_titles = len(title_to_reloc)
    logger.info(
        f"🏠 loaded relocation DB: {len(records)} records, "
        f"{n_titles} titles with relocation signal "
        f"({n_positive} positive, {n_titles - n_positive} unknown)"
    )
    return title_to_reloc


def aggregate_relocation_per_user(
    purchases: pd.DataFrame,
    title_to_reloc: dict[str, bool],
) -> dict[str, bool]:
    """For each user, True if ANY purchased product has a positive relocation signal.

    Products absent from title_to_reloc are treated as no signal (same as 'unknown').
    OR aggregation: one positive product is sufficient to flag the user as moved.
    """
    user_reloc: dict[str, bool] = {}
    n_positive = 0
    for user_id, group in purchases.groupby("Survey ResponseID"):
        has_positive = any(title_to_reloc.get(title) is True for title in group["Title"].tolist())
        user_reloc[str(user_id)] = has_positive
        if has_positive:
            n_positive += 1

    logger.info(
        f"🧮 aggregated relocation for {len(user_reloc)} users "
        f"({n_positive} predicted moved, {len(user_reloc) - n_positive} not moved)"
    )
    return user_reloc
