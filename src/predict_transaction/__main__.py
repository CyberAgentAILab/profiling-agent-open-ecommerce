from base_agent.config import load_config

from .dataset import load_search_result
from .model import TransactionSubAgent


def main() -> None:
    config = load_config()

    if config["use_validation_data"]:
        path = config["validation_path"]
    else:
        path = config["query_path"]

    search_results = load_search_result(
        path=path,
        is_reverse=config["is_reverse"],
    )

    # Generate job_id from begin-end range if both are specified
    job_id = None
    if config["begin"] is not None and config["end"] is not None:
        search_results = search_results[config["begin"] : config["end"]]
        job_id = f"{config['begin']}-{config['end']}"

    transaction_subagent = TransactionSubAgent(
        config=config,
        model_name=config["model_name"],
        torch_dtype=config["torch_dtype"],
        max_new_tokens=config["max_new_tokens"],
        max_model_len=config.get("max_model_len"),
        path_prompt_template=config["prompt_recovery_transaction_path"],
        use_vllm=config["use_vllm"],
        tensor_parallel_size=config["tensor_parallel_size"],
        trust_remote_code=config["trust_remote_code"],
        seed=config["seed"],
        gpu_memory_utilization=config["gpu_memory_utilization"],
        enforce_eager=config["enforce_eager"],
        out_dir=config["out_dir"],
        timestamp=config["timestamp"],
        enable_cache=config["enable_cache"],
        temperature=config["temperature"],
        top_p=config["top_p"],
        top_k=config["top_k"],
        min_p=config["min_p"],
        enable_prefix_caching=config["enable_prefix_caching"],
        enable_thinking=config.get("enable_thinking"),
    )

    transaction_subagent.predict_transaction_attribute(
        search_results=search_results,
        batch_size=config["batch_size"],
        debug_num_sample=config["debug_num_sample"],
        job_id=job_id,
    )
    return


if __name__ == "__main__":
    main()
