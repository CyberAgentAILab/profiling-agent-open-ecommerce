#!/bin/bash
# Sequential local orchestrator for the open-ecommerce pipeline.
#
# Pipeline overview:
#   [1] prepare_titles
#   [2] scan
#   [3] search
#   [4] predict_transaction
#   [5] predict_user          ← outputs 3 attribute types per transaction (and per frequent
#                               product combination read from results/open_ecommerce/
#                               frequent_pattern_mining/frequent_patterns.json, which is
#                               pre-computed outside this pipeline by
#                               scripts/open_ecommerce/local/frequent_pattern_mining/run.sh)
#   [6] cluster_attribute     ← clusters all 3 attribute types in one run
#   [7] tag_cluster           ← tags all 3 attribute types in one run
#   [8] judge_user_attribute  ← judges all 3 attribute types in one run
#   Standalone evaluation modules, run only when named in STAGES:
#       judge_demographic_attribute, investigate_confidence_feasibility
#
# Every run is recorded under ./results/open_ecommerce/<timestamp>/ (RUN_DIR):
#   configs/      the task_config.yaml actually passed to each stage (paths rewritten
#                 into this folder by scripts/open_ecommerce/local/prepare_run_configs.py)
#   logs/<stage>/ agent-*.log (pipeline log) and console.log (stdout/stderr)
#   outputs/      per stage: output-*.json (scan/, search/, transaction/, user/v4/,
#                 clustering/v4/, tag/v4/, judge_user_attribute/, ...)
#   summary.txt   per-stage status and duration
# prepare_titles writes its CSV to ./data/public/open-ecommerce/ (shared by all runs).
#
# Usage:
#   bash scripts/open_ecommerce/local/run_db.sh                  # run all stages on all data
#   STAGES="scan search" bash scripts/open_ecommerce/local/run_db.sh
#   NUM_SAMPLES=100 bash scripts/open_ecommerce/local/run_db.sh  # scan only the first 100 titles
#   NUM_USERS=10 bash scripts/open_ecommerce/local/run_db.sh     # judge only the first 10 users
#   MODEL_NAME=./models/Qwen3.5-27B bash scripts/open_ecommerce/local/run_db.sh  # use another LLM
#   RUN_DIR=./results/open_ecommerce/<timestamp> STAGES="predict_user cluster_attribute" \
#       bash scripts/open_ecommerce/local/run_db.sh              # continue an existing run
#
# RUN_DIR selects the run directory (default: a new ./results/open_ecommerce/<timestamp>).
# Point it at an existing run to re-run or continue some stages on that run's outputs.
# MODEL_NAME replaces model_name of every LLM stage (scan, predict_transaction,
# predict_user, tag_cluster, judge_user_attribute, judge_demographic_attribute) in the
# recorded configs; cluster_attribute keeps its embedding model. Unset (default) = the
# model_name written in each task_config.yaml (./models/Qwen3.5-4B).
# NUM_SAMPLES passes "--begin 0 --end N" to the scan stage, which then gates
# only the first N titles and writes output-<timestamp>-0-N.json; the later
# stages read scan's output as usual, so they shrink accordingly without any
# slicing of their own. NUM_USERS passes "--debug-num-user N" to
# judge_user_attribute so only the first N evaluable users are judged.
# Both unset (default) = process everything.

set -eo pipefail

GREEN='\033[0;32m'
RED='\033[0;31m'
GRAY='\033[0;90m'
RESET='\033[0m'

WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${WORK_DIR}"

ALL_STAGES=(
    prepare_titles
    scan
    search
    predict_transaction
    predict_user
    cluster_attribute
    tag_cluster
    judge_user_attribute
)
# Run only when explicitly listed in STAGES (they read the tag_cluster output of the run).
EXTRA_STAGES=(
    judge_demographic_attribute
    investigate_confidence_feasibility
)

# Allow callers to override the full stage list, e.g. STAGES="scan search"
if [ -n "${STAGES:-}" ]; then
    read -r -a STAGES_TO_RUN <<< "${STAGES}"
    if [ "${#STAGES_TO_RUN[@]}" -eq 0 ]; then
        echo -e "${RED}✗ STAGES is set but empty — nothing to run${RESET}"
        exit 1
    fi
    # Fail fast on typos so we never "succeed" having run zero stages.
    for s in "${STAGES_TO_RUN[@]}"; do
        _known=0
        for known in "${ALL_STAGES[@]}" "${EXTRA_STAGES[@]}"; do
            [ "${s}" = "${known}" ] && { _known=1; break; }
        done
        if [ "${_known}" -eq 0 ]; then
            echo -e "${RED}✗ unknown stage: '${s}'${RESET}"
            echo -e "${GRAY}  valid stages: ${ALL_STAGES[*]} ${EXTRA_STAGES[*]}${RESET}"
            exit 1
        fi
    done
    _run_custom=1
else
    _run_custom=0
fi

TIMESTAMP="$(TZ=Asia/Tokyo date +%Y-%m-%d_%H-%M-%S)"
RUN_DIR="${RUN_DIR:-./results/open_ecommerce/${TIMESTAMP}}"
RUN_DIR="${RUN_DIR%/}"
SUMMARY="${RUN_DIR}/summary.txt"
mkdir -p "${RUN_DIR}"
echo -e "${GRAY}[open_ecommerce/local/run_db] recording this run under ${RUN_DIR}${RESET}"
echo

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

# NUM_SAMPLES is applied here only; scan's __main__ honours --begin/--end.
SLICEABLE_STAGES=(scan)

if [ -n "${NUM_SAMPLES:-}" ]; then
    if ! [[ "${NUM_SAMPLES}" =~ ^[1-9][0-9]*$ ]]; then
        echo -e "${RED}✗ NUM_SAMPLES must be a positive integer, got '${NUM_SAMPLES}'${RESET}"
        exit 1
    fi
    echo -e "${GRAY}[open_ecommerce/local/run_db] NUM_SAMPLES=${NUM_SAMPLES}: scan runs on titles 0-${NUM_SAMPLES}${RESET}"
fi

if [ -n "${NUM_USERS:-}" ]; then
    if ! [[ "${NUM_USERS}" =~ ^[1-9][0-9]*$ ]]; then
        echo -e "${RED}✗ NUM_USERS must be a positive integer, got '${NUM_USERS}'${RESET}"
        exit 1
    fi
    echo -e "${GRAY}[open_ecommerce/local/run_db] NUM_USERS=${NUM_USERS}: judge_user_attribute runs on the first ${NUM_USERS} users${RESET}"
fi

# Stages that load the LLM named by model_name (cluster_attribute uses the embedding model instead).
LLM_STAGES=(scan predict_transaction predict_user tag_cluster judge_user_attribute judge_demographic_attribute)

if [ -n "${MODEL_NAME:-}" ]; then
    if [ ! -f "${MODEL_NAME}/config.json" ]; then
        echo -e "${RED}✗ MODEL_NAME='${MODEL_NAME}' is not a local model directory (no config.json). Download it first, e.g.:${RESET}"
        echo -e "${GRAY}  uv run hf download Qwen/Qwen3.5-27B --local-dir ./models/Qwen3.5-27B${RESET}"
        exit 1
    fi
    echo -e "${GRAY}[open_ecommerce/local/run_db] MODEL_NAME=${MODEL_NAME}: LLM stages use this model${RESET}"
fi

# Per-run configs: the committed task_config.yaml files with their results/logs paths
# redirected into RUN_DIR (re-generated on every invocation, including continuations).
uv run scripts/open_ecommerce/local/prepare_run_configs.py --run-dir "${RUN_DIR}" \
    ${MODEL_NAME:+--model-name "${MODEL_NAME}"} > /dev/null
LLM_MODEL="$(sed -n 's/^model_name: *//p' "${RUN_DIR}/configs/scan.yaml")"

{
    [ -s "${SUMMARY}" ] && echo
    echo "$ ${STAGES:+STAGES=\"${STAGES}\" }${NUM_SAMPLES:+NUM_SAMPLES=${NUM_SAMPLES} }${NUM_USERS:+NUM_USERS=${NUM_USERS} }${MODEL_NAME:+MODEL_NAME=${MODEL_NAME} }bash scripts/open_ecommerce/local/run_db.sh"
    echo "started_at:  $(TZ=Asia/Tokyo date +%Y-%m-%dT%H:%M:%S%z)"
    echo "run_dir:     ${RUN_DIR}"
    echo "gpu:         $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | paste -sd ',' || echo 'n/a')"
    echo "llm:         ${LLM_MODEL}"
    echo
} >> "${SUMMARY}"

_in_list() {
    local needle="$1"; shift
    for item in "$@"; do [ "${item}" = "${needle}" ] && return 0; done
    return 1
}

# Fills STAGE_ARGS with the extra CLI arguments for the given stage.
_stage_args() {
    local stage="$1"
    STAGE_ARGS=()
    if [ -n "${NUM_SAMPLES:-}" ] && _in_list "${stage}" "${SLICEABLE_STAGES[@]}"; then
        STAGE_ARGS+=(--begin 0 --end "${NUM_SAMPLES}")
    fi
    if [ -n "${NUM_USERS:-}" ] && [ "${stage}" = "judge_user_attribute" ]; then
        STAGE_ARGS+=(--debug-num-user "${NUM_USERS}")
    fi
}

_run_stage() {
    local stage="$1"
    local config="${RUN_DIR}/configs/${stage}.yaml"
    _stage_args "${stage}"
    # console.log sits next to the stage's agent-*.log (log_dir from the run config).
    local log_dir
    log_dir="$(sed -n 's/^log_dir: *//p' "${config}")"
    mkdir -p "${log_dir}"
    echo -e "${GRAY}========================================${RESET}"
    echo -e "${GRAY}▶ stage: ${stage}${STAGE_ARGS[*]:+ (${STAGE_ARGS[*]})}${RESET}"
    echo -e "${GRAY}========================================${RESET}"
    local stage_begin
    stage_begin=$(date +%s)
    if uv run -m "${stage}" --config "${config}" "${STAGE_ARGS[@]}" 2>&1 | _drop_private_ips | tee "${log_dir}/console.log"; then
        echo "${stage} ... completed ($(( $(date +%s) - stage_begin ))s)" >> "${SUMMARY}"
        echo -e "${GREEN}✓ ${stage} completed${RESET}\n"
    else
        echo "${stage} ... FAILED ($(( $(date +%s) - stage_begin ))s)" >> "${SUMMARY}"
        { echo; echo "finished_at: $(TZ=Asia/Tokyo date +%Y-%m-%dT%H:%M:%S%z)"; echo; echo "FAILED"; } >> "${SUMMARY}"
        echo -e "${RED}✗ ${stage} failed — aborting (see ${log_dir}/console.log)${RESET}"
        _sanitize_run_dir
        exit 1
    fi
}

_should_run() {
    local stage="$1"
    if [ "${_run_custom}" -eq 1 ]; then
        _in_list "${stage}" "${STAGES_TO_RUN[@]}"
        return
    fi
    # Default run = the 8 pipeline stages; the extra evaluation modules need to be named.
    _in_list "${stage}" "${ALL_STAGES[@]}"
}

for stage in "${ALL_STAGES[@]}" "${EXTRA_STAGES[@]}"; do
    _should_run "${stage}" && _run_stage "${stage}"
done

{ echo; echo "finished_at: $(TZ=Asia/Tokyo date +%Y-%m-%dT%H:%M:%S%z)"; echo; echo "OK"; } >> "${SUMMARY}"
_sanitize_run_dir
echo -e "${GREEN}✓ open-ecommerce pipeline finished — outputs in ${RUN_DIR}/outputs/, logs in ${RUN_DIR}/logs/, summary in ${SUMMARY}${RESET}"
