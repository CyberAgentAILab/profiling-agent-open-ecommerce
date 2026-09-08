"""Write per-run copies of the full-data task configs into a run directory.

Used by run_db.sh. The committed ``configs/open_ecommerce/<stage>/task_config.yaml``
files are templates: every path under ``./results/open_ecommerce/`` is rewritten to
``<run_dir>/outputs/`` and every path under ``./logs/open_ecommerce/`` to
``<run_dir>/logs/``, so one run keeps its configs, logs and outputs together in
``./results/open_ecommerce/<timestamp>/`` and the stages chain through that folder.
The pre-computed ``./results/open_ecommerce/frequent_pattern_mining/frequent_patterns.json``
read by predict_user is a fixed input and is left untouched, as are prompts, the
dataset, and the search / embedding caches.

Usage:
    uv run scripts/open_ecommerce/local/prepare_run_configs.py --run-dir ./results/open_ecommerce/<timestamp> \
        [--model-name ./models/Qwen3.5-27B]
"""

import argparse
from pathlib import Path
from typing import Any

from common import run_configs

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs" / "open_ecommerce"
# The 8 pipeline stages of run_db.sh ...
PIPELINE_STAGES = (
    "prepare_titles",
    "scan",
    "search",
    "predict_transaction",
    "predict_user",
    "cluster_attribute",
    "tag_cluster",
    "judge_user_attribute",
)
# ... plus the standalone evaluation modules, which read the tag_cluster output of a run.
EXTRA_STAGES = ("judge_demographic_attribute", "investigate_confidence_feasibility")
STAGES = PIPELINE_STAGES + EXTRA_STAGES
# Stages whose model_name is the LLM (cluster_attribute's model_name is the embedding model).
LLM_STAGES = (
    "scan",
    "predict_transaction",
    "predict_user",
    "tag_cluster",
    "judge_user_attribute",
    "judge_demographic_attribute",
)
PREFIX_MAP = {"./results/open_ecommerce/": "outputs", "./logs/open_ecommerce/": "logs"}
# Pre-computed inputs that live at a fixed path and must not be redirected into the run dir.
KEEP_PREFIXES = ("./results/open_ecommerce/frequent_pattern_mining/",)


def rewrite_paths(config: dict[str, Any], run_dir: str) -> dict[str, Any]:
    """Return a copy of config whose open_ecommerce result/log paths point into run_dir."""
    return run_configs.rewrite_paths(config, run_dir, PREFIX_MAP, KEEP_PREFIXES)


def prepare_run_configs(run_dir: str, model_name: str | None = None) -> list[Path]:
    """Write <run_dir>/configs/<stage>.yaml for every stage and return the paths."""
    return run_configs.prepare_run_configs(
        CONFIG_DIR, STAGES, run_dir, PREFIX_MAP, LLM_STAGES, model_name, KEEP_PREFIXES
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", required=True, help="Run directory, e.g. ./results/open_ecommerce/<timestamp>")
    parser.add_argument("--model-name", default=None, help="Local LLM path overriding model_name of the LLM stages")
    args = parser.parse_args()
    for path in prepare_run_configs(args.run_dir, model_name=args.model_name):
        print(path)


if __name__ == "__main__":
    main()
