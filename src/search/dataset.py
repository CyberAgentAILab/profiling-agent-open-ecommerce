import json
from pathlib import Path

import pandas as pd
from loguru import logger


def load_ecommerce_search_list(
    path: str,
    filter_need_search: bool = True,
    filter_is_diagnostic: bool = True,
) -> list[dict]:
    """Load scan output (JSON file or directory of JSONs) for the ecommerce branch
    and filter records by need_search / is_diagnostic flags.

    Filter precedence (matches the pipeline runs behind the paper):
    - filter_need_search=True: keep only records where scan.need_search is True
      (filter_is_diagnostic has no additional effect in this mode).
    - filter_need_search=False, filter_is_diagnostic=True: keep records where
      scan.is_diagnostic is True OR scan.need_search is True (need_search
      bypasses the diagnostic gate).
    - both False: keep everything.

    Args:
        path: Path to a scan output JSON file OR a directory containing them.
        filter_need_search: Keep only records where scan.need_search is True.
        filter_is_diagnostic: Keep records where scan.is_diagnostic is True.

    Returns:
        List of records ready for the search step. Records keep their original
        scan dict so downstream stages can read it.
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

    all_records: list[dict] = []
    for jp in json_paths:
        with open(jp) as f:
            records = json.load(f)
        all_records.extend(records)
        logger.info(f"📚 ecommerce: loaded {len(records)} scan records from {jp}")

    n_total = len(all_records)
    filtered = []
    for r in all_records:
        scan = r.get("scan") or {}
        need_search = bool(scan.get("need_search", False))
        is_diagnostic = bool(scan.get("is_diagnostic", False))
        if filter_need_search:
            keep = need_search
        elif filter_is_diagnostic:
            # need_search=True bypasses the is_diagnostic gate.
            keep = need_search or is_diagnostic
        else:
            keep = True
        if keep:
            filtered.append(r)

    logger.info(
        f"🛒 ecommerce search input: {n_total} → {len(filtered)} "
        f"(filter_need_search={filter_need_search}, filter_is_diagnostic={filter_is_diagnostic})"
    )
    return filtered


def load_open_ecommerce_dataset(csv_path: str) -> list[dict[str, str | bool | int]]:
    """Load open-ecommerce dataset from CSV and extract Title column with dummy input.

    Args:
        csv_path: Path to the CSV file

    Returns:
        List of dictionaries with search item structure including dummy scan data
    """
    df = pd.read_csv(csv_path)
    titles = df["Title"].tolist()

    search_list: list[dict[str, str | bool | int]] = []
    for title in titles:
        search_item = {
            "raw_query": title,
            "query": title,
            "resolved_query": title,
            "scan": {"need_search": True, "reason": "dummy", "is_json": True},
        }
        search_list.append(search_item)
    return search_list
