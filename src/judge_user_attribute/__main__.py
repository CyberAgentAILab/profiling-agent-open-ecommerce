from pathlib import Path
from typing import Any

from loguru import logger

from base_agent.config import load_config

from .dataset import (
    aggregate_signals_per_user,
    aggregate_tags_per_user,
    aggregate_titles_per_user,
    filter_purchases_by_min_buyers,
    load_purchases,
    load_signal_attributes,
    load_survey,
    load_tag_db,
)
from .model import (
    JudgeUserAttributeAgent,
    attach_ground_truth,
    build_baseline_user_context,
    build_user_context_multi,
    evaluate_predictions,
    filter_users_with_any_yes_ground_truth,
    save_results,
)


def _latest_output(dir_path: str) -> str:
    """Return the path of the most recently written output-*.json in dir_path."""
    files = sorted(Path(dir_path).glob("output-*.json"))
    if not files:
        raise FileNotFoundError(f"No output-*.json files found in {dir_path}")
    return str(files[-1])


def _run_mode(
    mode: str,
    agent: JudgeUserAttributeAgent,
    user_ids: list[str],
    contexts: list[str],
    survey_df: Any,
    config: dict,
    cohort_counts: dict[str, int] | None = None,
) -> None:
    logger.info(f"🚀 [{mode}] running judgement on {len(user_ids)} users (batch_size={config['batch_size']})")
    predictions = agent.judge_users(
        user_ids=user_ids,
        contexts=contexts,
        batch_size=config["batch_size"],
    )
    predictions = attach_ground_truth(predictions=predictions, survey_df=survey_df)
    metrics = evaluate_predictions(predictions=predictions, survey_df=survey_df)
    if cohort_counts is not None:
        metrics["__cohort__"] = dict(cohort_counts)

    save_path = save_results(
        predictions=predictions,
        metrics=metrics,
        out_dir=config["out_dir"],
        timestamp=config["timestamp"],
        mode=mode,
    )
    logger.info(f"💾 [{mode}] results saved -> {save_path}")


def _run_db(
    config: dict,
    agent: JudgeUserAttributeAgent,
    purchases: Any,
    survey_df: Any,
) -> None:
    """Run db mode: load all attribute_branches tag DBs and combine into one prompt per user."""
    agent.out_dir = config["out_dir"]
    debug_num_user = config.get("debug_num_user")

    with open(config["db_prompt_path"]) as f:
        agent.prompt_template = f.read()

    branches = config.get("attribute_branches") or []
    branch_tag_data: list[tuple[str, list[str], dict[str, dict[str, int]], dict[str, dict[str, list[str]]]]] = []
    # All branches read the same merged tag DB file, so the per-product signal
    # attributes only need to be loaded once. Capture the resolved path here.
    signal_db_path: str | None = None
    for branch in branches:
        branch_cfg = {**config, **branch}
        tag_db_path = branch_cfg.get("tag_db_path") or _latest_output(branch_cfg["tag_db_dir"])
        if signal_db_path is None:
            signal_db_path = tag_db_path
        title_to_tags, all_tags = load_tag_db(
            db_path=tag_db_path,
            tag_suffix=branch_cfg.get("tag_suffix", "ecommerce"),
        )
        user_to_tags, user_to_title_tags = aggregate_tags_per_user(
            purchases=purchases,
            title_to_tags=title_to_tags,
            all_tags=all_tags,
            log_first_n=branch_cfg.get("log_first_n", 3),
            # Same product budget as the baseline so both modes see the same purchases.
            max_products=config.get("baseline_max_titles_per_user"),
        )
        branch_tag_data.append((branch["name"], all_tags, user_to_tags, user_to_title_tags))
        logger.info(f"🌿 loaded branch={branch['name']} (tag universe size={len(all_tags)})")

    all_user_sets = [set(u2t.keys()) for _, _, u2t, _ in branch_tag_data]
    user_ids = sorted(set.intersection(*all_user_sets)) if all_user_sets else []

    user_ids, cohort_counts = filter_users_with_any_yes_ground_truth(user_ids, survey_df, mode="db")
    if isinstance(debug_num_user, int) and debug_num_user > 0:
        user_ids = user_ids[:debug_num_user]
        logger.info(f"🐛 [db] debug_num_user={debug_num_user}: limiting to {len(user_ids)} users")

    # Per-product fixed-attribute signals, aggregated per user over the same
    # top-N titles the tag aggregation uses (purchase-count weighted).
    # enable_signals gates whether the signal block is appended to the prompt:
    #   True  (default) -> hybrid    = DB tags + per-product fixed-attribute signals
    #   False           -> tags_only = DB tags alone (no signal section)
    enable_signals = config.get("enable_signals", True)
    user_to_signals: dict[str, dict[str, Any]] = {}
    if enable_signals and signal_db_path is not None:
        title_to_signals = load_signal_attributes(signal_db_path)
        user_to_signals = aggregate_signals_per_user(
            purchases=purchases,
            title_to_signals=title_to_signals,
            log_first_n=config.get("log_first_n", 3),
            max_products=config.get("baseline_max_titles_per_user"),
        )
    logger.info(f"🔌 enable_signals={enable_signals} (mode={'hybrid' if enable_signals else 'tags_only'})")

    contexts = [
        build_user_context_multi(branch_tag_data, uid, user_to_signals if enable_signals else None) for uid in user_ids
    ]
    logger.info(f"🧱 [db] built {len(contexts)} multi-branch prompt contexts ({len(branch_tag_data)} branches)")
    _run_mode("db", agent, user_ids, contexts, survey_df, config, cohort_counts=cohort_counts)


def _run_baseline(
    config: dict,
    agent: JudgeUserAttributeAgent,
    purchases: Any,
    survey_df: Any,
) -> None:
    """Run baseline mode once using raw purchase titles (no tag DB)."""
    agent.out_dir = config["out_dir"]
    debug_num_user = config.get("debug_num_user")

    with open(config["prompt_path"]) as f:
        agent.prompt_template = f.read()

    user_to_titles = aggregate_titles_per_user(
        purchases=purchases,
        log_first_n=config.get("log_first_n", 3),
    )
    user_ids = list(user_to_titles.keys())
    user_ids, cohort_counts = filter_users_with_any_yes_ground_truth(user_ids, survey_df, mode="baseline")
    if isinstance(debug_num_user, int) and debug_num_user > 0:
        user_ids = user_ids[:debug_num_user]
        logger.info(f"🐛 [baseline] debug_num_user={debug_num_user}: limiting to {len(user_ids)} users")
    max_titles = config.get("baseline_max_titles_per_user")
    contexts = [build_baseline_user_context(user_to_titles[uid], max_titles=max_titles) for uid in user_ids]
    logger.info(f"🧱 [baseline] built {len(contexts)} prompt contexts (max_titles_per_user={max_titles})")
    mode = f"baseline-min{config['min_unique_buyers']}"
    _run_mode(mode, agent, user_ids, contexts, survey_df, config, cohort_counts=cohort_counts)


def main() -> None:
    config = load_config()

    enable_db = config.get("enable_db", True)
    enable_baseline = config.get("enable_baseline", False)
    if not (enable_db or enable_baseline):
        logger.warning("⚠️ both enable_db and enable_baseline are False — nothing to do.")
        return
    logger.info(f"🧭 modes: enable_db={enable_db} enable_baseline={enable_baseline}")

    purchases = load_purchases(purchases_csv=config["purchases_csv"])
    purchases = filter_purchases_by_min_buyers(purchases, min_unique_buyers=config["min_unique_buyers"])
    survey_df = load_survey(survey_csv=config["survey_csv"])

    # Load the vLLM model once; both modes share the same model.
    agent = JudgeUserAttributeAgent(
        model_name=config["model_name"],
        torch_dtype=config["torch_dtype"],
        path_prompt_template=config["prompt_path"],
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
        enable_prefix_caching=config["enable_prefix_caching"],
        enable_cache=config["enable_cache"],
        # The per-call generation cap judge_users() applies; the paper runs
        # used the 2048 default, so the committed configs pin it explicitly.
        json_max_tokens=config.get("json_max_tokens", 2048),
        enable_thinking=config.get("enable_thinking"),
    )

    # Baseline runs once using the top-level out_dir (independent of branches).
    if enable_baseline:
        _run_baseline(config, agent, purchases, survey_df)

    # DB mode loads all branches' tag DBs and combines them into one prompt per user.
    if enable_db:
        _run_db(config, agent, purchases, survey_df)


if __name__ == "__main__":
    main()
