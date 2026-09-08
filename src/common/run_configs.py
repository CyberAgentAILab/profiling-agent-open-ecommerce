"""Per-run copies of the task configs.

A pipeline run keeps everything in one timestamped run directory
(``<run_dir>/configs``, ``<run_dir>/logs``, ``<run_dir>/outputs``). The committed
``task_config.yaml`` files act as templates: every path that starts with one of the
``prefix_map`` keys (e.g. ``./results/open_ecommerce/`` or ``./logs/open_ecommerce/``)
is rewritten under ``<run_dir>/<sub dir>/``, which also re-chains the stages because a
stage reads the previous stage's ``out_dir``. Paths starting with a ``keep_prefixes``
entry (pre-computed inputs such as ``frequent_patterns.json``) and every other value
(prompts, datasets, caches, numbers) are left untouched.
"""

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml


def rewrite_value(value: Any, run_dir: str, prefix_map: Mapping[str, str], keep_prefixes: Iterable[str] = ()) -> Any:
    if isinstance(value, str):
        if any(value.startswith(keep) for keep in keep_prefixes):
            return value
        for prefix, sub_dir in prefix_map.items():
            if value.startswith(prefix):
                return f"{run_dir.rstrip('/')}/{sub_dir}/{value[len(prefix) :]}"
        return value
    if isinstance(value, dict):
        return {k: rewrite_value(v, run_dir, prefix_map, keep_prefixes) for k, v in value.items()}
    if isinstance(value, list):
        return [rewrite_value(v, run_dir, prefix_map, keep_prefixes) for v in value]
    return value


def rewrite_paths(
    config: dict[str, Any],
    run_dir: str,
    prefix_map: Mapping[str, str],
    keep_prefixes: Iterable[str] = (),
) -> dict[str, Any]:
    """Return a copy of ``config`` whose result/log paths point into ``run_dir`` (nested values included)."""
    keep = tuple(keep_prefixes)
    return {key: rewrite_value(value, run_dir, prefix_map, keep) for key, value in config.items()}


def prepare_run_configs(
    config_dir: Path,
    stages: Iterable[str],
    run_dir: str,
    prefix_map: Mapping[str, str],
    llm_stages: Iterable[str] = (),
    model_name: str | None = None,
    keep_prefixes: Iterable[str] = (),
) -> list[Path]:
    """Write ``<run_dir>/configs/<stage>.yaml`` for every stage and return the paths.

    ``config_dir/<stage>/task_config.yaml`` is the template of each stage. When
    ``model_name`` is given, the stages listed in ``llm_stages`` are pointed at it
    (the embedding model of cluster_attribute is deliberately not an LLM stage).
    """
    config_out_dir = Path(run_dir) / "configs"
    config_out_dir.mkdir(parents=True, exist_ok=True)
    llm = set(llm_stages)
    written: list[Path] = []
    for stage in stages:
        with open(config_dir / stage / "task_config.yaml") as f:
            config = yaml.safe_load(f)
        config = rewrite_paths(config, run_dir, prefix_map, keep_prefixes)
        if model_name and stage in llm:
            config["model_name"] = model_name
        path = config_out_dir / f"{stage}.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        written.append(path)
    return written
