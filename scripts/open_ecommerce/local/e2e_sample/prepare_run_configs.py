"""Write per-run copies of the e2e_sample task configs into a run directory.

Every path under ``./results/e2e_sample/`` (stage outputs, which also chain the
stages together) is rewritten to ``<run_dir>/outputs/`` and every path under
``./logs/e2e_sample/`` to ``<run_dir>/logs/``, so one invocation of run.sh keeps
all of its configs, logs and outputs in a single timestamped folder. Other
paths (prompts, the sample CSV, the shared search cache) are left untouched.
The rewriting itself lives in ``common.run_configs`` and is shared with the
full-data runner (scripts/open_ecommerce/local/prepare_run_configs.py).

Usage:
    uv run scripts/open_ecommerce/local/e2e_sample/prepare_run_configs.py --run-dir ./results/e2e_sample/<timestamp> \
        [--model-name ./models/Qwen3.5-27B]
"""

import argparse
from pathlib import Path
from typing import Any

from common import run_configs

REPO_ROOT = Path(__file__).resolve().parents[4]
CONFIG_DIR = REPO_ROOT / "configs" / "open_ecommerce" / "e2e_sample"
STAGES = ("prepare_titles", "scan", "search", "predict_transaction", "predict_user", "cluster_attribute", "tag_cluster")
# Stages whose model_name is the LLM (cluster_attribute's model_name is the embedding model).
LLM_STAGES = ("scan", "predict_transaction", "predict_user", "tag_cluster")
PREFIX_MAP = {"./results/e2e_sample/": "outputs", "./logs/e2e_sample/": "logs"}


def rewrite_paths(config: dict[str, Any], run_dir: str) -> dict[str, Any]:
    """Return a copy of config whose e2e_sample result/log paths point into run_dir."""
    return run_configs.rewrite_paths(config, run_dir, PREFIX_MAP)


def prepare_run_configs(run_dir: str, model_name: str | None = None) -> list[Path]:
    """Write <run_dir>/configs/<stage>.yaml for every e2e_sample stage and return the paths."""
    return run_configs.prepare_run_configs(CONFIG_DIR, STAGES, run_dir, PREFIX_MAP, LLM_STAGES, model_name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", required=True, help="Run directory, e.g. ./results/e2e_sample/<timestamp>")
    parser.add_argument("--model-name", default=None, help="Local LLM path overriding model_name of the LLM stages")
    args = parser.parse_args()
    for path in prepare_run_configs(args.run_dir, model_name=args.model_name):
        print(path)


if __name__ == "__main__":
    main()
