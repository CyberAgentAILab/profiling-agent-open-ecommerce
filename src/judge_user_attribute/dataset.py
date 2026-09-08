import json
from collections import Counter
from pathlib import Path

import pandas as pd
from loguru import logger


def load_tag_db(db_path: str, tag_suffix: str = "ecommerce") -> tuple[dict[str, list[str]], list[str]]:
    """Load the tag DB and produce:
      - title_to_tags: {Title -> ordered list of tags assigned to that title}
      - all_tags: sorted list of every tag observed in the DB

    The DB is the per-Title output produced by tag_cluster, where each record
    has `resolved_query` (the Amazon Title) and
    `tag2pseudo_confidence_{tag_suffix}` (a dict of cluster tag -> confidence).
    tag_suffix is specified per branch in task_config.yaml
    (e.g. "demographic", "psycho_behavioral", "life_event", or "ecommerce").
    """
    db_path_abs = Path(db_path).resolve()
    with open(db_path_abs) as f:
        records = json.load(f)

    tag_db_key = f"tag2pseudo_confidence_{tag_suffix}"

    title_to_tags: dict[str, list[str]] = {}
    all_tags: set[str] = set()

    for record in records:
        title = record.get("resolved_query")
        if not isinstance(title, str) or not title:
            continue
        tag_map = record.get(tag_db_key) or {}
        if not isinstance(tag_map, dict):
            continue
        # Sort tags by descending confidence so the highest-evidence tag is first.
        # Drop the catch-all "outlier" tag (HDBSCAN noise cluster, cluster_id=-1):
        # it carries no semantic signal and only adds noise to the prompt.
        sorted_tags = sorted(tag_map.items(), key=lambda kv: -float(kv[1] or 0.0))
        tags = [t for t, _ in sorted_tags if isinstance(t, str) and t != "outlier"]
        if not tags:
            continue
        title_to_tags[title] = tags
        all_tags.update(tags)

    logger.info(
        f"📚 loaded tag DB: {len(records)} records, {len(title_to_tags)} titles with tags, {len(all_tags)} unique tags"
    )
    return title_to_tags, sorted(all_tags)


# Fixed product-level attribute taxonomy. Every "signal:"-prefixed category is
# predicted per product (resolved_query) in the tag DB, independent of the tag
# clusters, under user.user_attribute_{domain}.consumer_attributes[]. We consume
# only these categories and aggregate them per user as an additional soft prior.
SIGNAL_PREFIX = "signal:"

_SIGNAL_BRANCH_KEYS: tuple[str, ...] = (
    "user_attribute_demographic",
    "user_attribute_psycho_behavioral",
    "user_attribute_life_event",
)


def load_signal_attributes(db_path: str) -> dict[str, dict[str, list[str]]]:
    """Load per-product fixed-attribute predictions from the tag DB.

    Each record (one per resolved_query / Amazon Title) carries product-level
    predictions under user.user_attribute_{demographic,psycho_behavioral,
    life_event}.consumer_attributes[], where each entry is
    {category, reason, attribute}. We keep only categories whose name starts
    with "signal:" and whose predicted `attribute` is a concrete value (dropping
    "unknown"/null/empty).

    Keyed by `resolved_query` exactly like load_tag_db, so the signal block
    covers the same titles as the tag block.

    Returns title_to_signals: {Title -> {signal-category -> [predicted values]}}.
    Single-string predictions become a one-element list; multi-valued
    predictions are preserved as-is.
    """
    db_path_abs = Path(db_path).resolve()
    with open(db_path_abs) as f:
        records = json.load(f)

    title_to_signals: dict[str, dict[str, list[str]]] = {}
    n_titles_with_signal = 0
    for record in records:
        title = record.get("resolved_query")
        if not isinstance(title, str) or not title:
            continue
        user = record.get("user") or {}
        signals: dict[str, list[str]] = {}
        for branch_key in _SIGNAL_BRANCH_KEYS:
            consumer_attrs = (user.get(branch_key) or {}).get("consumer_attributes") or []
            if not isinstance(consumer_attrs, list):
                continue
            for entry in consumer_attrs:
                if not isinstance(entry, dict):
                    continue
                category = entry.get("category")
                if not (isinstance(category, str) and category.startswith(SIGNAL_PREFIX)):
                    continue
                value = entry.get("attribute")
                values = value if isinstance(value, list) else [value]
                clean = [v for v in values if isinstance(v, str) and v and v != "unknown"]
                if clean:
                    # one prediction per category per product; first concrete value wins
                    signals.setdefault(category, clean)
        if signals:
            title_to_signals[title] = signals
            n_titles_with_signal += 1

    logger.info(
        f"🔮 loaded signal attributes: {len(records)} records, {n_titles_with_signal} titles with ≥1 signal prediction"
    )
    return title_to_signals


def aggregate_signals_per_user(
    purchases: pd.DataFrame,
    title_to_signals: dict[str, dict[str, list[str]]],
    log_first_n: int = 3,
    max_products: int | None = None,
) -> dict[str, dict[str, Counter]]:
    """Aggregate per-product signal predictions per user.

    Uses the SAME top-`max_products` title selection (count desc, ties by title
    asc) and the SAME purchase-count weighting as aggregate_tags_per_user, so the
    signal block stays consistent with the tag block.

    Returns user_signals: {user_id -> {signal-category -> Counter(value -> count)}}.
    Only categories the user actually matched (≥1 non-unknown prediction) appear.
    """
    user_signals: dict[str, dict[str, Counter]] = {}
    logged = 0

    for user_id, group in purchases.groupby("Survey ResponseID"):
        title_counts: Counter = Counter(t for t in group["Title"].tolist() if isinstance(t, str) and t)
        ordered_titles = [t for t, _ in sorted(title_counts.items(), key=lambda kv: (-kv[1], kv[0]))]
        if max_products is not None and max_products > 0:
            ordered_titles = ordered_titles[:max_products]

        cat_to_counter: dict[str, Counter] = {}
        for title in ordered_titles:
            signals = title_to_signals.get(title)
            if not signals:
                continue
            count = title_counts[title]  # weight by purchase count (matches tag aggregation)
            for category, values in signals.items():
                counter = cat_to_counter.setdefault(category, Counter())
                for value in values:
                    counter[value] += count

        user_signals[str(user_id)] = cat_to_counter

        if logged < log_first_n:
            preview = {c: dict(cnt.most_common(5)) for c, cnt in list(cat_to_counter.items())[:6]}
            # Survey ResponseIDs are intentionally not logged (personal identifiers).
            logger.info(
                f"🔮 signal-user[{logged + 1}/{log_first_n}] matched_categories={len(cat_to_counter)}"
            )
            logger.info(f"    aggregated signals (preview): {preview}")
            logged += 1

    logger.info(f"🧮 aggregated signal attributes for {len(user_signals)} users")
    return user_signals


def load_purchases(purchases_csv: str) -> pd.DataFrame:
    df = pd.read_csv(purchases_csv, usecols=["Title", "Survey ResponseID"])
    df = df.dropna(subset=["Survey ResponseID", "Title"])
    logger.info(f"🛒 loaded purchases: {len(df)} rows, {df['Survey ResponseID'].nunique()} users")
    return df


def filter_purchases_by_min_buyers(df: pd.DataFrame, min_unique_buyers: int) -> pd.DataFrame:
    """Drop rows whose Title was purchased by fewer than min_unique_buyers unique users."""
    unique_buyers = df.groupby("Title")["Survey ResponseID"].nunique()
    qualifying_titles = unique_buyers[unique_buyers >= min_unique_buyers].index
    filtered = df[df["Title"].isin(qualifying_titles)]
    logger.info(
        f"🔍 filter min_unique_buyers={min_unique_buyers}: "
        f"{df['Title'].nunique():,} titles → {len(qualifying_titles):,} titles kept, "
        f"{len(df):,} rows → {len(filtered):,} rows"
    )
    return filtered


def aggregate_tags_per_user(
    purchases: pd.DataFrame,
    title_to_tags: dict[str, list[str]],
    all_tags: list[str],
    log_first_n: int = 3,
    max_products: int | None = None,
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, list[str]]]]:
    """For each user (Survey ResponseID), build:
      - user_tags: a frequency dict over EVERY tag in the DB universe (tags the
        user never received are recorded with count 0). Ordered by count desc,
        then tag name asc, with all `all_tags` keys present.
      - user_title_tags: {title -> tags} for the user's selected titles that
        carry at least one tag in this branch — used to render the product→tag
        mapping in the prompt.

    To stay comparable with the baseline (which shows the top `max_titles`
    titles by purchase count), the same selection is applied here: titles are
    ranked by purchase count descending (ties by title asc) and only the top
    `max_products` UNIQUE titles feed BOTH the tag-frequency counts and the
    product→tag map. With max_products == baseline_max_titles_per_user, both
    modes consume the exact same purchase set. Tag counts are weighted by each
    title's purchase count (matching the baseline's "(xN)" display).

    The first `log_first_n` users are logged for auditability.

    Returns (user_tags, user_title_tags).
    """
    user_tags: dict[str, dict[str, int]] = {}
    user_title_tags: dict[str, dict[str, list[str]]] = {}
    logged = 0

    for user_id, group in purchases.groupby("Survey ResponseID"):
        # Per-user unique titles with purchase counts (same basis as the baseline).
        title_counts: Counter = Counter(t for t in group["Title"].tolist() if isinstance(t, str) and t)
        # Select the top `max_products` unique titles by count desc (ties: title asc),
        # identical to build_baseline_user_context's top-`max_titles` selection.
        ordered_titles = [t for t, _ in sorted(title_counts.items(), key=lambda kv: (-kv[1], kv[0]))]
        n_titles = len(ordered_titles)
        if max_products is not None and max_products > 0:
            ordered_titles = ordered_titles[:max_products]

        counter: Counter = Counter()
        title_tags: dict[str, list[str]] = {}
        n_titles_with_tags = 0
        per_user_tag_lists: list[list[str]] = []

        for title in ordered_titles:
            tags = title_to_tags.get(title)
            if not tags:
                continue
            n_titles_with_tags += 1
            count = title_counts[title]
            for tag in tags:
                counter[tag] += count  # weight by purchase count (matches baseline "(xN)")
            title_tags[title] = tags
            if logged < log_first_n:
                per_user_tag_lists.append(tags)

        full_freq = {tag: int(counter.get(tag, 0)) for tag in all_tags}
        full_freq = dict(sorted(full_freq.items(), key=lambda kv: (-kv[1], kv[0])))
        user_tags[str(user_id)] = full_freq
        user_title_tags[str(user_id)] = title_tags

        if logged < log_first_n:
            n_nonzero = sum(1 for v in full_freq.values() if v > 0)
            # Survey ResponseIDs are intentionally not logged: run records are
            # committed, and personal identifiers must stay out of them.
            logger.info(
                f"🔖 user[{logged + 1}/{log_first_n}] "
                f"titles={n_titles} titles_with_tags={n_titles_with_tags} "
                f"distinct_tags_seen={n_nonzero}/{len(all_tags)}"
            )
            logger.info(f"    per-title tag lists (first 20): {per_user_tag_lists[:20]}")
            logger.info(f"    aggregated tag counts (all {len(all_tags)} DB tags, incl. zeros): {full_freq}")
            logged += 1

    logger.info(
        f"🧮 aggregated tags for {len(user_tags)} users over {len(all_tags)} DB tags (zero-count tags included)"
    )
    return user_tags, user_title_tags


def aggregate_titles_per_user(
    purchases: pd.DataFrame,
    log_first_n: int = 3,
) -> dict[str, dict[str, int]]:
    """Baseline aggregation: per user (Survey ResponseID), build a frequency
    dict of raw Amazon Titles. No tag DB is consulted — the LLM has to reason
    directly from the product names.

    Returns a mapping user_id -> {title: count} ordered by count desc, then by
    title asc. The first `log_first_n` users are logged for auditability.
    """
    user_titles: dict[str, dict[str, int]] = {}
    logged = 0

    for user_id, group in purchases.groupby("Survey ResponseID"):
        counter: Counter = Counter()
        for title in group["Title"].tolist():
            if isinstance(title, str) and title:
                counter[title] += 1

        freq = dict(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))
        user_titles[str(user_id)] = freq

        if logged < log_first_n:
            n_titles = sum(freq.values())
            n_unique = len(freq)
            preview = list(freq.items())[:10]
            # Survey ResponseIDs are intentionally not logged (personal identifiers).
            logger.info(
                f"🧾 baseline-user[{logged + 1}/{log_first_n}] "
                f"total_titles={n_titles} unique_titles={n_unique}"
            )
            logger.info(f"    top-10 titles by count: {preview}")
            logged += 1

    logger.info(f"📦 aggregated raw Titles for {len(user_titles)} users (baseline mode)")
    return user_titles


def load_survey(survey_csv: str) -> pd.DataFrame:
    df = pd.read_csv(survey_csv)
    logger.info(f"📋 loaded survey: {len(df)} respondents")
    return df
