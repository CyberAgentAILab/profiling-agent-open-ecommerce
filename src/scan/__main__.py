from loguru import logger

from base_agent.config import load_config

from .dataset import load_titles_for_ecommerce
from .model import ScanSubAgent


def main() -> None:
    config = load_config()

    if config["use_validation_data"]:
        path = config["validation_path"]
    else:
        path = config["query_path"]

    logger.info(f"📝 prompt_path: {config['prompt_path']}")

    noisy_queries, freqs, queries, resolved_queries = load_titles_for_ecommerce(
        path_query=path,
        is_reverse=config["is_reverse"],
    )

    # Generate job_id from begin-end range if both are specified
    job_id = None
    if config["begin"] is not None and config["end"] is not None:
        noisy_queries = noisy_queries[config["begin"] : config["end"]]
        freqs = freqs[config["begin"] : config["end"]]
        queries = queries[config["begin"] : config["end"]]
        resolved_queries = resolved_queries[config["begin"] : config["end"]]
        job_id = f"{config['begin']}-{config['end']}"

    scan_subagent = ScanSubAgent(
        model_name=config["model_name"],
        torch_dtype=config["torch_dtype"],
        path_prompt_template=config["prompt_path"],
        out_dir=config["out_dir"],
        timestamp=config["timestamp"],
        max_new_tokens=config["max_new_tokens"],
        max_model_len=config.get("max_model_len"),
        do_sample=config["do_sample"],
        temperature=config["temperature"],
        top_p=config["top_p"],
        use_vllm=config["use_vllm"],
        tensor_parallel_size=config["tensor_parallel_size"],
        trust_remote_code=config["trust_remote_code"],
        seed=config["seed"],
        gpu_memory_utilization=config["gpu_memory_utilization"],
        enforce_eager=config["enforce_eager"],
        yes_tokens=config["yes_tokens"],
        json_schema_scan=config["json_schema_scan"],
        enable_cache=config["enable_cache"],
        enable_thinking=config.get("enable_thinking"),
    )

    scan_subagent.scan_need_search(
        noisy_queries=noisy_queries,
        freqs=freqs,
        queries=queries,
        resolved_queries=resolved_queries,
        batch_size=config["batch_size"],
        debug_num_sample=config["debug_num_sample"],
        job_id=job_id,
    )


if __name__ == "__main__":
    main()
