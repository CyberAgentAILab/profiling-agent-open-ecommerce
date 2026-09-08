from loguru import logger

from base_agent.config import load_config

from .dataset import load_recovered_transaction
from .model import UserSubAgent


def main() -> None:
    config = load_config()

    if config["use_validation_data"]:
        path = config["validation_path"]
    else:
        path = config["query_path"]

    logger.info(f"📝 prompt_path: {config['prompt_user_attribute_path']}")

    transactions = load_recovered_transaction(
        path=path,
        is_reverse=config["is_reverse"],
        scan_output_path=config.get("scan_output_path") or None,
        frequent_patterns_path=config.get("frequent_patterns_path") or None,
        allowed_titles_csv=config.get("allowed_titles_csv") or None,
    )

    # Generate job_id from begin-end range if both are specified
    job_id = None
    if config["begin"] is not None and config["end"] is not None:
        transactions = transactions[config["begin"] : config["end"]]
        job_id = f"{config['begin']}-{config['end']}"

    user_subagent = UserSubAgent(
        config=config,
        model_name=config["model_name"],
        torch_dtype=config["torch_dtype"],
        path_prompt_template=config["prompt_user_attribute_path"],
        max_new_tokens=config["max_new_tokens"],
        max_model_len=config.get("max_model_len"),
        use_vllm=config["use_vllm"],
        tensor_parallel_size=config["tensor_parallel_size"],
        trust_remote_code=config["trust_remote_code"],
        seed=config["seed"],
        gpu_memory_utilization=config["gpu_memory_utilization"],
        enforce_eager=config["enforce_eager"],
        enable_prefix_caching=config["enable_prefix_caching"],
        out_dir=config["out_dir"],
        timestamp=config["timestamp"],
        enable_cache=config["enable_cache"],
        enable_thinking=config.get("enable_thinking"),
    )

    user_subagent.predict_user_attributes(
        transactions=transactions,
        batch_size=config["batch_size"],
        debug_num_sample=config["debug_num_sample"],
        job_id=job_id,
    )

    return


if __name__ == "__main__":
    main()
