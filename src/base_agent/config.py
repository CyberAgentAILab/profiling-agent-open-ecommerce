import argparse
import datetime
import json
import os
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import ddgs
import torch
import transformers
import yaml
from dotenv import find_dotenv, load_dotenv
from loguru import logger

# Re-export: the canonical implementation lives in common.io.
from common.io import save_as_json as save_as_json


def _is_valid_attribute_value(value: object) -> bool:
    """A consumer_attributes value is valid when it is a scalar (str/None/bool)
    or a list of strings.

    The open-ecommerce combined schema legitimately uses list-valued `attribute`
    fields: Layer 1 multi-select categories (e.g. signal:age-bin / income-bin /
    lifestyle / personality / benefit-sought) emit an array of values, and the
    `free_description_attributes` entry is always a (possibly empty) string list.
    Extra keys such as `pseudo_confidence` are ignored because only required_keys
    are checked below.
    """
    if isinstance(value, str) or value is None or isinstance(value, bool):
        return True
    if isinstance(value, list):
        return all(isinstance(v, str) for v in value)
    return False


def _validate_with_attribute(
    parsed_json: dict,
    required_keys: list[str],
    required_attribute_name: str,
) -> dict:
    if not isinstance(parsed_json[required_attribute_name], list):
        parsed_json["is_json"] = False
        return parsed_json

    is_valid = True
    for item in parsed_json[required_attribute_name]:
        if not isinstance(item, dict):
            is_valid = False
            break

        if not all(key in item and _is_valid_attribute_value(item[key]) for key in required_keys):
            is_valid = False
            break

    parsed_json["is_json"] = is_valid
    if is_valid:
        logger.info("✔️ Extracted JSON properly")
    else:
        logger.warning(f"❌ JSON does not match the required schema: {required_keys}")
    return parsed_json


def validate_json(
    llm_output: str,
    required_keys: list[str],
    required_attribute_name: str | None = None,
) -> dict:
    try:
        parsed_json = json.loads(llm_output)
    except (json.JSONDecodeError, Exception):
        logger.error(f"failed parse example:\n{llm_output}")
        return {"is_json": False}

    if required_attribute_name and required_attribute_name in parsed_json:
        return _validate_with_attribute(parsed_json, required_keys, required_attribute_name)

    generated_unique_keys = set(parsed_json.keys())
    required_unique_keys = set(required_keys)
    if generated_unique_keys == required_unique_keys:
        parsed_json["is_json"] = True
        logger.info("✔️ Extracted JSON properly")
    else:
        parsed_json["is_json"] = False
        logger.warning(f"❌ JSON keys {generated_unique_keys} do not match required keys {required_unique_keys}")

    return parsed_json


def load_yaml_config(config_path: str) -> dict:
    config_path_abs = Path(config_path).resolve()
    with open(str(config_path_abs)) as f:
        return yaml.safe_load(f)


def parse_config() -> dict:
    parser = argparse.ArgumentParser(description="dev")
    parser.add_argument("--config", type=str, default="./configs/base_agent.yaml")
    parser.add_argument("--begin", type=int, default=None, help="Begin index for data slicing")
    parser.add_argument("--end", type=int, default=None, help="End index for data slicing")
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Local path (or HF id) of the LLM; overrides model_name in the config",
    )
    parser.add_argument(
        "--debug-num-user",
        type=int,
        default=None,
        help="Judge only the first N users (judge_user_attribute); overrides debug_num_user in the config",
    )
    args = parser.parse_args()

    # Load ./.env (copied from .env.example) if present. Existing environment
    # variables take precedence over values in the file.
    load_dotenv(find_dotenv(usecwd=True))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    config = load_yaml_config(args.config)
    config["device"] = device
    # Search API tokens are optional: unset or empty means "use the DDGS fallback".
    config["serp_api_token"] = os.environ.get("SERP_API_TOKEN") or None
    config["serper_api_token"] = os.environ.get("SERPER_API_TOKEN") or None

    # Override config with command line arguments if provided
    if args.begin is not None:
        config["begin"] = args.begin
    if args.end is not None:
        config["end"] = args.end
    if args.debug_num_user is not None:
        config["debug_num_user"] = args.debug_num_user
    if args.model_name is not None:
        config["model_name"] = args.model_name

    os.makedirs(config["out_dir"], exist_ok=True)
    return config


def _select_search_api(config: dict) -> str:
    """Mirror WebRetriever's engine selection: serp/serper is used only when the
    configured engine has both its URL and token; otherwise DDGS is the fallback.
    """
    engine = config.get("search_engine")
    if engine == "serp" and config.get("serp_url") and config.get("serp_api_token"):
        return "serp"
    if engine == "serper" and config.get("serper_url") and config.get("serper_api_token"):
        return "serper"
    return "ddgs"


def _redact_secrets(config: dict) -> dict:
    return {key: ("***" if key.endswith("_api_token") and value else value) for key, value in config.items()}


def activate_logging(log_dir: str) -> str:
    now = datetime.datetime.now(ZoneInfo("Asia/Tokyo"))
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    logger.info(f"timestamp: {timestamp}")
    logger.add(f"{log_dir}/agent-{timestamp}.log")
    return timestamp


def load_config() -> dict:
    torch.cuda.empty_cache()
    config = parse_config()
    timestamp = activate_logging(log_dir=config["log_dir"])
    config["timestamp"] = timestamp
    logger.info(f"⛏️ Transformers version: {transformers.__version__}")
    logger.info(f"⛏️ duckduckgo version: {ddgs.__version__}")
    logger.info(f"⛏️ Python version: {sys.version}")
    logger.info(f"⛏️ PyTorch version: {torch.__version__}")
    logger.info(f"⛏️ CUDA available: {torch.cuda.is_available()}")
    logger.info(f"⛏️ MPS available: {torch.backends.mps.is_available()}")
    logger.info(f"⛏️ Device: {config['device']}")
    logger.info(f"⚙️ config: {_redact_secrets(config)}")
    logger.info(f"🔍 search API: {_select_search_api(config)}")
    return config
