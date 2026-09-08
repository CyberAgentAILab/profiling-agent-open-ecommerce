from pathlib import Path

from base_agent.config import load_config

from .dataset import build_attribute2pseudo_confidence, build_cluster_id2attributes, load_attribute_clusters
from .model import TaggerSubAgent


def _latest_output(dir_path: str) -> str:
    """Return the path of the most recently written output-*.json in dir_path."""
    files = sorted(Path(dir_path).glob("output-*.json"))
    if not files:
        raise FileNotFoundError(f"No output-*.json files found in {dir_path}")
    return str(files[-1])


def _resolve_sampling_method(config: dict) -> str:
    """Resolve the cluster-context sampling method from config.

    Honors the explicit `sampling_method` key; otherwise falls back to the legacy
    `use_embedding_cache` flag (kmedoids when set) for backward compatibility, and
    defaults to "kmedoids".
    """
    method = config.get("sampling_method")
    if method:
        return method
    if "use_embedding_cache" in config:
        return "kmedoids" if config["use_embedding_cache"] else "head"
    return "kmedoids"


def _run_branch(config: dict) -> None:
    query_path = config.get("query_path") or _latest_output(config["query_dir"])

    profiles, cluster_id2attributes, attribute2pseudo_confidence = load_attribute_clusters(path=query_path)

    job_id = None
    if config["begin"] is not None and config["end"] is not None:
        profiles = profiles[config["begin"] : config["end"]]
        cluster_id2attributes = build_cluster_id2attributes(profiles)
        attribute2pseudo_confidence = build_attribute2pseudo_confidence(profiles)
        job_id = f"{config['begin']}-{config['end']}"

    tagger_subagent = TaggerSubAgent(
        model_name=config["model_name"],
        torch_dtype=config["torch_dtype"],
        path_prompt_template=config["path_prompt_template"],
        out_dir=config["out_dir"],
        timestamp=config["timestamp"],
        max_new_tokens=config["max_new_tokens"],
        max_model_len=config.get("max_model_len"),
        use_vllm=config["use_vllm"],
        tensor_parallel_size=config["tensor_parallel_size"],
        trust_remote_code=config["trust_remote_code"],
        seed=config["seed"],
        gpu_memory_utilization=config["gpu_memory_utilization"],
        enforce_eager=config["enforce_eager"],
        json_schema_tag=config["json_schema_tag"],
        enable_cache=config["enable_cache"],
        sampling_method=_resolve_sampling_method(config),
        cache_embed_dir=config.get("cache_embed_dir") or config.get("embedding_cache_path", ""),
        cache_embed_model=config.get("cache_embed_model", ""),
        temperature=config.get("temperature", 0.0),
        top_p=config.get("top_p", 0.9),
        top_k=config.get("top_k", 50),
        enable_thinking=config.get("enable_thinking"),
    )

    tagger_subagent.run_tagging(
        profiles=profiles,
        cluster_id2attributes=cluster_id2attributes,
        attribute2pseudo_confidence=attribute2pseudo_confidence,
        batch_size=config["batch_size"],
        debug_num_sample=config["debug_num_sample"],
        job_id=job_id,
        cluster_id_namespace=config.get("cluster_id_namespace", "shared"),
    )


def main() -> None:
    config = load_config()
    branches = config.get("attribute_branches")
    if branches:
        for branch in branches:
            _run_branch({**config, **branch})
    else:
        _run_branch(config)


if __name__ == "__main__":
    main()
