"""I/O utilities for saving data."""

import json
import os
from pathlib import Path

from loguru import logger


def save_as_json(
    responses: list[dict] | dict,
    out_dir: str,
    timestamp: str,
    job_id: str | None = None,
) -> str:
    """Save responses as JSON file.

    Args:
        responses: Single dict or list of dicts to save
        out_dir: Directory to save the file
        timestamp: Timestamp string to include in filename
        job_id: Optional job identifier to prevent file conflicts in parallel execution

    Returns:
        Absolute path to the saved JSON file
    """
    responses = [responses] if isinstance(responses, dict) else responses

    os.makedirs(out_dir, exist_ok=True)
    if job_id:
        save_path = f"{out_dir}/output-{timestamp}-{job_id}.json"
    else:
        save_path = f"{out_dir}/output-{timestamp}.json"
    save_path_abs = Path(save_path).resolve()
    with open(str(save_path_abs), "w") as f:
        json.dump(responses, f, ensure_ascii=False, indent=2)
    # DEBUG because checkpointing stages call this once per item/batch;
    # each stage logs the final path at INFO when its run completes.
    logger.debug(f"save_path: {str(save_path_abs)}")
    return str(save_path_abs)
