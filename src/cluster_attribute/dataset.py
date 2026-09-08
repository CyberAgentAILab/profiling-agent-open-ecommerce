import json
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

# The free_description_attributes entry carries {attribute: confidence} under
# "pseudo_confidence" (original prompt) or "confidence" (slimmed prompt).
CONFIDENCE_KEYS = ("pseudo_confidence", "confidence")


def extract_pseudo_confidence(entry: Any) -> dict[str, float] | None:
    """Return the {attribute: confidence} dict of a consumer_attributes entry, or None."""
    if not isinstance(entry, dict):
        return None
    for key in CONFIDENCE_KEYS:
        value = entry.get(key)
        if isinstance(value, dict):
            return value
    return None


def extract_attributes(
    nested_attributes: dict[str, Any],
    pattern: str,
) -> dict[str, float] | None:
    attributes_nest = nested_attributes.get(pattern)
    if not isinstance(attributes_nest, dict):
        return None

    attributes_list = attributes_nest.get("consumer_attributes")
    if not isinstance(attributes_list, list) or not attributes_list:
        return None

    # The free-description entry (with confidences) is always the last one.
    return extract_pseudo_confidence(attributes_list[-1])


_PATTERN_LABEL_MAP = {
    "user_attribute_ecommerce": "purchase",
}


def _pattern_label(pattern: str) -> str:
    if pattern in _PATTERN_LABEL_MAP:
        return _PATTERN_LABEL_MAP[pattern]
    # Fallback: strip the user_attribute_ prefix if present
    return pattern.replace("user_attribute_", "") or pattern


def load_profiles(path: str) -> list[dict[str, Any]]:
    """Read predict_user output JSON (single file or directory) into a flat list.

    The returned profiles are independent of target_patterns, so callers can load
    them once and feed the same list to multiple clustering branches.
    """
    path_obj = Path(path)
    if path_obj.is_file():
        json_paths = [path]
    elif path_obj.is_dir():
        json_paths = sorted(str(p) for p in path_obj.glob("*.json"))
        if not json_paths:
            logger.warning(f"No .json files found in {path}")
            return []
    else:
        raise ValueError(f"Path is not a file or directory: {path}")

    all_profiles: list[dict[str, Any]] = []
    total_files = len(json_paths)

    for idx, json_path in enumerate(json_paths, 1):
        with open(str(json_path)) as f:
            profiles = json.load(f)

        all_profiles.extend(profiles)
        logger.info(f"📚 loaded {len(profiles)} profiles from {str(json_path)} ({idx}/{total_files})")

    logger.info(f"📊 Total: {len(all_profiles)} profiles from {total_files} file(s)")
    return all_profiles


def build_attribute_df(
    profiles: list[Any],
    target_patterns: list[str] | None = None,
) -> pd.DataFrame:
    """Flatten persona pseudo_confidence into a long-form DataFrame for clustering.

    Args:
        profiles: Profiles loaded via load_profiles().
        target_patterns: Pattern keys under each record's "user" dict to extract.
            Defaults to ["user_attribute_ecommerce"].
    """
    if target_patterns is None:
        target_patterns = ["user_attribute_ecommerce"]

    logger.info(f"🎯 target_patterns: {target_patterns}")

    attributes = []
    for profile in profiles:
        if not isinstance(profile, dict):
            continue

        attributes_per_transaction = profile.get("user")
        if not isinstance(attributes_per_transaction, dict):
            continue

        for pattern in target_patterns:
            attribute_list = extract_attributes(
                nested_attributes=attributes_per_transaction,
                pattern=pattern,
            )
            if attribute_list is None:
                continue

            label = _pattern_label(pattern)
            for pseudo_key, pseudo_value in attribute_list.items():
                attributes.append(
                    {
                        "resolved_query": profile.get("resolved_query"),
                        "pattern": label,
                        "attribute": pseudo_key,
                        "pseudo_confidence": pseudo_value,
                    }
                )

    df_user_attributes = pd.DataFrame(attributes)
    if len(df_user_attributes) == 0:
        logger.warning("No attributes were extracted — check target_patterns and input format")
        return df_user_attributes

    n_accounts = df_user_attributes["resolved_query"].nunique()
    n_attributes = df_user_attributes["attribute"].nunique()
    logger.info(f"🗃️  {len(df_user_attributes)} attributes were extracted")
    logger.info(f"🗃️  {n_accounts} unique accounts were extracted")
    logger.info(f"🗃️  {n_attributes} unique attributes were extracted")
    logger.info(f"🗃️  Head\n\n{df_user_attributes.head()}")
    return df_user_attributes
