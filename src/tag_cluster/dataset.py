import json
from collections import defaultdict
from typing import Any


def build_cluster_id2attributes(profiles: list[dict[str, Any]]) -> dict[str, dict[int, list[str]]]:
    """Build cluster_id2attributes from profiles.

    Args:
        profiles: List of profile records

    Returns:
        cluster_id2attributes: Dict keyed by pattern name with {cluster_id: [attributes]} values
    """
    # Build cluster_id2attributes per pattern
    pattern_cluster_id2attributes: dict[str, dict[int, list[str]]] = {}

    for record in profiles:
        attribute2cluster_id_all = record.get("attribute2cluster_id", {})

        # Process each pattern (user_attribute_demographic, etc.)
        for pattern_name, attribute2cluster_id in attribute2cluster_id_all.items():
            if pattern_name not in pattern_cluster_id2attributes:
                pattern_cluster_id2attributes[pattern_name] = defaultdict(list)

            # Process each attribute -> cluster_id mapping
            for attribute, cluster_info in attribute2cluster_id.items():
                # Handle both dict cluster_info (new format) and plain values (old format)
                if isinstance(cluster_info, dict):
                    cluster_id = cluster_info.get("cluster_id")
                else:
                    cluster_id = cluster_info

                if cluster_id is not None and attribute not in pattern_cluster_id2attributes[pattern_name][cluster_id]:
                    pattern_cluster_id2attributes[pattern_name][cluster_id].append(attribute)

    # Convert the defaultdicts to plain dicts
    result = {
        pattern_name: dict(cluster_id2attrs) for pattern_name, cluster_id2attrs in pattern_cluster_id2attributes.items()
    }

    return result


def build_attribute2pseudo_confidence(profiles: list[dict[str, Any]]) -> dict[str, float]:
    """Build attribute2pseudo_confidence from profiles.

    Args:
        profiles: List of profile records

    Returns:
        attribute2pseudo_confidence: Dict keyed by attribute name with pseudo_confidence values
    """
    attribute2pseudo_confidence: dict[str, float] = {}

    for record in profiles:
        attribute2cluster_id_all = record.get("attribute2cluster_id", {})

        # Process each pattern (user_attribute_demographic, etc.)
        for _pattern_name, attribute2cluster_id in attribute2cluster_id_all.items():
            # Process each attribute -> pseudo_confidence mapping
            for attribute, cluster_info in attribute2cluster_id.items():
                # Skip non-dict cluster_info (old format)
                if not isinstance(cluster_info, dict):
                    continue

                pseudo_confidence = cluster_info.get("pseudo_confidence")
                if pseudo_confidence is None:
                    continue

                # Convert to float when given as a string
                if isinstance(pseudo_confidence, str):
                    try:
                        pseudo_confidence = float(pseudo_confidence)
                    except ValueError:
                        continue

                # Insert when absent, otherwise keep the larger value
                if attribute not in attribute2pseudo_confidence:
                    attribute2pseudo_confidence[attribute] = pseudo_confidence
                else:
                    # Keep the maximum (the same attribute can appear in multiple profiles)
                    attribute2pseudo_confidence[attribute] = max(
                        attribute2pseudo_confidence[attribute], pseudo_confidence
                    )

    return attribute2pseudo_confidence


def load_attribute_clusters(
    path: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[int, list[str]]], dict[str, float]]:
    """Load attribute-cluster data.

    Args:
        path: Path to the JSON file

    Returns:
        profiles: The loaded JSON records
        cluster_id2attributes: Dict keyed by pattern name with {cluster_id: [attributes]} values
        attribute2pseudo_confidence: Dict keyed by attribute name with pseudo_confidence values
    """
    with open(path) as f:
        profiles = json.load(f)

    cluster_id2attributes = build_cluster_id2attributes(profiles)
    attribute2pseudo_confidence = build_attribute2pseudo_confidence(profiles)

    return profiles, cluster_id2attributes, attribute2pseudo_confidence
