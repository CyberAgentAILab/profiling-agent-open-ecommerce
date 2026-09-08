from loguru import logger

from base_agent.config import load_config

from .dataset import load_ecommerce_search_list, load_open_ecommerce_dataset
from .model import WebRetriever


def main() -> None:
    config = load_config()

    # Prefer scan output (for is_diagnostic gate); fall back to raw CSV when
    # use_validation_data is False — this preserves the legacy "skip-scan" path.
    if config.get("use_validation_data", True):
        scan_path = config["validation_path"]
        logger.info(f"🛒 reading scan output from {scan_path}")
        search_list = load_ecommerce_search_list(
            path=scan_path,
            filter_need_search=config.get("filter_need_search", True),
            filter_is_diagnostic=config.get("filter_is_diagnostic", True),
        )
    else:
        csv_path = config["csv_path"]
        logger.info(f"🛒 reading raw CSV from {csv_path} (skipping scan)")
        search_list = load_open_ecommerce_dataset(csv_path=csv_path)

    # Generate job_id from begin-end range if both are specified
    job_id = None
    if config["begin"] is not None and config["end"] is not None:
        search_list = search_list[config["begin"] : config["end"]]
        job_id = f"{config['begin']}-{config['end']}"

    # Use separate cache directory for each job to avoid conflicts in parallel execution
    search_cache_dir = config["search_cache_dir"]
    if job_id:
        search_cache_dir = f"{search_cache_dir}_{job_id}"

    web_retriever = WebRetriever(
        search_engine=config["search_engine"],
        region=config["ddgs_region"],
        backend=config["backend"],
        num_search=config["num_search"],
        serp_url=config["serp_url"],
        serp_api_token=config["serp_api_token"],
        serper_url=config["serper_url"],
        serper_api_token=config["serper_api_token"],
        timeout=config["timeout"],
        search_cache_dir=search_cache_dir,
        query_suffix=config.get("query_suffix", ""),
    )

    web_retriever.run_search(
        search_list=search_list,
        out_dir=config["out_dir"],
        timestamp=config["timestamp"],
        job_id=job_id,
        num_debug=config.get("num_debug"),
    )

    return


if __name__ == "__main__":
    main()
