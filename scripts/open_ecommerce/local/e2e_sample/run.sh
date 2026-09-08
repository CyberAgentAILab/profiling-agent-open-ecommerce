#!/bin/bash
# E2E sample run: 10 committed purchase rows through user-attribute inference.
#
# Runs the first 7 pipeline stages sequentially with the small configs under
# configs/open_ecommerce/e2e_sample/, using the local LLM (Qwen3.5-4B by default)
# (GPU machine). The search stage uses the ddgs (DuckDuckGo) scraper unless
# SERPER_API_TOKEN is set in .env; cached queries in ./search_cache/e2e_sample
# are reused either way.
#
# Every run is recorded under ./results/e2e_sample/<timestamp>/
# (override the parent directory with E2E_SAMPLE_RESULTS_DIR):
#   configs/   the task_config.yaml actually passed to each stage
#   logs/      per stage (prepare_titles, scan, search, transaction, user):
#              agent-*.log (pipeline log) and console.log (stdout/stderr)
#   outputs/   per stage: output-*.json (user attributes in outputs/user/, tagged
#              attribute clusters in outputs/tag/)
#   summary.txt   per-stage status and duration
#
# Usage:
#   bash scripts/open_ecommerce/local/e2e_sample/run.sh
#   MODEL_NAME=./models/Qwen3.5-27B bash scripts/open_ecommerce/local/e2e_sample/run.sh
#
# MODEL_NAME (optional) replaces model_name of the LLM stages in the recorded
# configs; unset = the model_name written in configs/open_ecommerce/e2e_sample/
# (./models/Qwen3.5-4B).
#
# For the offline, model-free variant of the same flow, run the stubbed test:
#   uv run python -m unittest tests.e2e.test_pipeline_e2e -v

set -eo pipefail

GREEN='\033[0;32m'
RED='\033[0;31m'
GRAY='\033[0;90m'
RESET='\033[0m'

WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${WORK_DIR}"

TIMESTAMP="$(TZ=Asia/Tokyo date +%Y-%m-%d_%H-%M-%S)"
RUN_DIR="${E2E_SAMPLE_RESULTS_DIR:-./results/e2e_sample}/${TIMESTAMP}"
SUMMARY="${RUN_DIR}/summary.txt"
mkdir -p "${RUN_DIR}"

if [ -n "${MODEL_NAME:-}" ] && [ ! -f "${MODEL_NAME}/config.json" ]; then
    echo -e "${RED}✗ MODEL_NAME='${MODEL_NAME}' is not a local model directory (no config.json). Download it first, e.g.:${RESET}"
    echo -e "${GRAY}  uv run hf download Qwen/Qwen3.5-27B --local-dir ./models/Qwen3.5-27B${RESET}"
    rmdir "${RUN_DIR}"
    exit 1
fi
uv run scripts/open_ecommerce/local/e2e_sample/prepare_run_configs.py --run-dir "${RUN_DIR}" \
    ${MODEL_NAME:+--model-name "${MODEL_NAME}"} > /dev/null
LLM_MODEL="$(sed -n 's/^model_name: *//p' "${RUN_DIR}/configs/scan.yaml")"

{
    echo "$ bash scripts/open_ecommerce/local/e2e_sample/run.sh"
    echo "started_at:  $(TZ=Asia/Tokyo date +%Y-%m-%dT%H:%M:%S%z)"
    echo "run_dir:     ${RUN_DIR}"
    echo "gpu:         $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | paste -sd ',' || echo 'n/a')"
    echo "llm:         ${LLM_MODEL}"
    echo
} > "${SUMMARY}"
echo -e "${GRAY}[e2e_sample] recording this run under ${RUN_DIR}${RESET}"

# The recorded configs / logs must not carry this machine's absolute repo path.
_sanitize_run_dir() {
    grep -rlF "${WORK_DIR}" "${RUN_DIR}" 2>/dev/null | xargs -r sed -i "s|${WORK_DIR}|.|g"
}

# This machine's private IPs (RFC 1918) are masked in the console stream BEFORE it is
# written, so console.log never records them (e.g. vLLM prints
# distributed_init_method=tcp://<ip>:<port> on startup).
_PRIVATE_IP_RE='\b(10\.[0-9]{1,3}|192\.168|172\.(1[6-9]|2[0-9]|3[01]))\.[0-9]{1,3}\.[0-9]{1,3}\b'
_drop_private_ips() {
    sed -uE "s/${_PRIVATE_IP_RE}/[redacted-ip]/g"
}

STAGES=(
    prepare_titles
    scan
    search
    predict_transaction
    predict_user
    cluster_attribute
    tag_cluster
)

for stage in "${STAGES[@]}"; do
    echo -e "${GRAY}========================================${RESET}"
    echo -e "${GRAY}▶ e2e_sample stage: ${stage}${RESET}"
    echo -e "${GRAY}========================================${RESET}"
    # console.log sits next to the stage's agent-*.log (log_dir from the run config).
    log_dir="$(sed -n 's/^log_dir: *//p' "${RUN_DIR}/configs/${stage}.yaml")"
    mkdir -p "${log_dir}"
    stage_begin=$(date +%s)
    if uv run -m "${stage}" --config "${RUN_DIR}/configs/${stage}.yaml" 2>&1 | _drop_private_ips | tee "${log_dir}/console.log"; then
        echo "${stage} ... completed ($(( $(date +%s) - stage_begin ))s)" >> "${SUMMARY}"
        echo -e "${GREEN}✓ ${stage} completed${RESET}\n"
    else
        echo "${stage} ... FAILED ($(( $(date +%s) - stage_begin ))s)" >> "${SUMMARY}"
        { echo; echo "finished_at: $(TZ=Asia/Tokyo date +%Y-%m-%dT%H:%M:%S%z)"; echo; echo "FAILED"; } >> "${SUMMARY}"
        echo -e "${RED}✗ ${stage} failed — aborting (see ${log_dir}/console.log)${RESET}"
        _sanitize_run_dir
        exit 1
    fi
done

{ echo; echo "finished_at: $(TZ=Asia/Tokyo date +%Y-%m-%dT%H:%M:%S%z)"; echo; echo "OK"; } >> "${SUMMARY}"
_sanitize_run_dir
echo -e "${GREEN}✓ e2e sample pipeline finished — user attributes in ${RUN_DIR}/outputs/user/, tagged clusters in ${RUN_DIR}/outputs/tag/${RESET}"
