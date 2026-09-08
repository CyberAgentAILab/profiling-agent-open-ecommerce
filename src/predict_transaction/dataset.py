import json
from pathlib import Path
from typing import Any

from loguru import logger


def load_search_result(path: str, is_reverse: bool) -> list[dict[str, Any]]:
    path_obj = Path(path)
    if path_obj.is_file():
        logger.info("detect a single json file")
        json_paths = [path]
    elif path_obj.is_dir():
        logger.info("detect a directory that contains json files")
        json_paths = sorted(str(p) for p in path_obj.glob("*.json"))
        if not json_paths:
            logger.warning(f"No .json files found in {path}")
            return []
    else:
        raise ValueError(f"Path is not a file or directory: {path}")

    all_search_results = []
    total_samples = 0
    total_failures = 0

    for json_path in json_paths:
        with open(str(json_path)) as f:
            search_results = json.load(f)

        filtered_results = []
        num_failures = 0

        for search_result in search_results:
            search_result_content = search_result.get("search", False)
            if search_result_content:
                success_search = search_result_content.get("success_search", False)
            else:
                num_failures += 1
                continue

            if success_search:
                filtered_results.append(search_result)
            else:
                num_failures += 1

        all_search_results.extend(filtered_results)
        total_samples += len(search_results)
        total_failures += num_failures

        logger.info(f"📚 loaded {len(search_results)} samples from {str(json_path)}")
        logger.info(f"Omitted {num_failures} samples due to search failure")

    if is_reverse:
        all_search_results.reverse()

    logger.info(f"📊 Total: {total_samples} samples, {total_failures} failures")
    return all_search_results
