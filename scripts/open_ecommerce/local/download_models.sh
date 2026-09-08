#!/bin/bash
# Download the model weights used by the pipeline into ./models/ with
# `hf download <model> --local-dir ./models/<name>`.
#
# The pipeline uses exactly two models:
#   - LLM (all generation stages):  Qwen3.5-4B by default (~8GB). The paper used
#     Qwen3.5-27B (~52GB); switch the entry in MODELS below and set model_name to
#     ./models/Qwen3.5-27B in the task_config.yaml of every generation stage.
#   - Embedding (cluster_attribute / tag_cluster): Qwen3-Embedding-0.6B
#
# Usage:
#   bash scripts/open_ecommerce/local/download_models.sh
#
# Files already downloaded are reused (hf download resumes/skips existing files).

set -euo pipefail

WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${WORK_DIR}"

MODELS=(
    "Qwen/Qwen3.5-4B"
    # "Qwen/Qwen3.5-27B"   # model used in the paper (~52GB); needs an 80GB GPU
    "Qwen/Qwen3-Embedding-0.6B"
)

for repo in "${MODELS[@]}"; do
    name="$(basename "${repo}")"
    echo "▶ downloading ${repo} -> ./models/${name}"
    uv run hf download "${repo}" --local-dir "./models/${name}"
done

echo "✓ all models available under ./models/"
