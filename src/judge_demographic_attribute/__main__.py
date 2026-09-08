from pathlib import Path
from typing import Any

from loguru import logger

from base_agent.config import load_config

# Reuse judge_user_attribute's loaders/builders so the per-user purchase history
# is built exactly the same way (same CSV columns, same min_unique_buyers filter,
# same top-N title selection the AUC-based baseline uses).
from judge_user_attribute.dataset import (
    aggregate_titles_per_user,
    filter_purchases_by_min_buyers,
    load_purchases,
    load_survey,
)
from judge_user_attribute.model import build_baseline_user_context

from .dataset import (
    aggregate_demographics_per_user,
    aggregate_relocation_per_user,
    load_demographic_db,
    load_relocation_db,
)
from .model import (
    DemographicBaselineAgent,
    collect_income_education,
    compute_age_correlation,
    compute_education_metrics,
    compute_gender_metrics,
    compute_income_metrics,
    compute_relocation_metrics,
    fill_unknown_demographics,
    parse_baseline_predictions,
    plot_age_confusion_matrix,
    plot_age_scatter,
    plot_education_confusion_matrix,
    plot_income_confusion_matrix,
    save_results,
)


def _latest_output(dir_path: str) -> str:
    """Return the path of the most recently written output-*.json in dir_path."""
    files = sorted(Path(dir_path).glob("output-*.json"))
    if not files:
        raise FileNotFoundError(f"No output-*.json files found in {dir_path}")
    return str(files[-1])


def _evaluate_and_save(
    mode: str,
    user_demo: dict[str, dict[str, Any]],
    user_reloc: dict[str, bool],
    survey_indexed: Any,
    config: dict,
    extra_metrics: dict[str, Any],
) -> None:
    """Run the full demographic evaluation suite over a per-user prediction set
    and persist the metrics + plots. Both the DB path and the DB-free baseline
    feed the SAME `user_demo`/`user_reloc` structures here, so the age / gender /
    income / education / relocation evaluation is identical across modes.

    `mode` ('db' | 'baseline') is woven into every output filename so the two
    runs never clobber each other. `extra_metrics` carries mode-specific context
    (e.g. db_path or model_name) merged into the saved payload.
    """
    out = Path(config["out_dir"])
    ts = config["timestamp"]
    suffix = f"-{mode}"

    # Unify the evaluation cohort across methods: fill unknown/no-signal point
    # predictions (age/gender/income/education) so the only remaining drop is the
    # method-independent survey-validity filter. Default 'random' (config-driven).
    fill_strategy = config.get("unknown_fill", "random")
    fill_seed = config.get("unknown_fill_seed", 3407)
    user_demo = fill_unknown_demographics(user_demo, strategy=fill_strategy, seed=fill_seed)

    # age: Spearman correlation, extended metrics, scatter, confusion matrix.
    age = compute_age_correlation(user_demo, survey_indexed)
    plot_path = plot_age_scatter(
        survey_ages=age["survey_ages"],
        db_ages=age["db_ages"],
        out_path=str(out / f"age_scatter{suffix}-{ts}.png"),
        spearman_r=age.get("spearman_r"),
    )
    cm_plot_path = plot_age_confusion_matrix(
        survey_ages=age["survey_ages"],
        db_ages=age["db_ages"],
        out_path=str(out / f"age_confusion_matrix{suffix}-{ts}.png"),
    )

    # gender: precision / recall / F1 over {male, female, other}.
    gender = compute_gender_metrics(user_demo, survey_indexed)

    # income-bin: ordinal evaluation (Spearman, ordinal MAE, dollar MAE, weighted kappa).
    income = compute_income_metrics(user_demo, survey_indexed)
    income_cm_plot_path = plot_income_confusion_matrix(
        survey_ranks=income["survey_ranks"],
        db_ranks=income["db_ranks"],
        out_path=str(out / f"income_confusion_matrix{suffix}-{ts}.png"),
    )

    # education: ordinal evaluation (Spearman, ordinal MAE, weighted kappa).
    education = compute_education_metrics(user_demo, survey_indexed)
    education_cm_plot_path = plot_education_confusion_matrix(
        survey_ranks=education["survey_ranks"],
        db_ranks=education["db_ranks"],
        out_path=str(out / f"education_confusion_matrix{suffix}-{ts}.png"),
    )

    # relocation (life-event): binary classification with OR aggregation.
    relocation = compute_relocation_metrics(user_reloc, survey_indexed)

    # raw per-user records for income + education (kept for debugging).
    income_education = collect_income_education(user_demo, survey_indexed)

    metrics: dict[str, Any] = {
        "mode": mode,
        "n_users": len(user_demo),
        "min_unique_buyers": config["min_unique_buyers"],
        "unknown_fill": fill_strategy,
        "unknown_fill_seed": fill_seed,
        "age": age,
        "gender": gender,
        "income_education": income_education,
        "age_scatter_plot": plot_path,
        "age_confusion_matrix_plot": cm_plot_path,
        "income": income,
        "income_confusion_matrix_plot": income_cm_plot_path,
        "education": education,
        "education_confusion_matrix_plot": education_cm_plot_path,
        "relocation": relocation,
    }
    metrics.update(extra_metrics)

    save_path = save_results(metrics, out_dir=config["out_dir"], timestamp=ts, mode=mode)
    logger.info(f"💾 [{mode}] metrics saved -> {save_path}")


def _run_db(config: dict, purchases: Any, survey_indexed: Any) -> None:
    """DB-based evaluation: aggregate the per-product demographic signals in the
    tag DB into a per-user prediction, then score against the survey."""
    db_path = config.get("demographic_db_path") or _latest_output(config["demographic_db_dir"])
    logger.info(f"📥 [db] demographic DB: {db_path}")

    title_to_demo = load_demographic_db(db_path)
    title_to_reloc = load_relocation_db(db_path)

    user_demo = aggregate_demographics_per_user(
        purchases=purchases,
        title_to_demo=title_to_demo,
        log_first_n=config.get("log_first_n", 3),
    )
    user_reloc = aggregate_relocation_per_user(purchases=purchases, title_to_reloc=title_to_reloc)

    _evaluate_and_save(
        mode="db",
        user_demo=user_demo,
        user_reloc=user_reloc,
        survey_indexed=survey_indexed,
        config=config,
        extra_metrics={"db_path": db_path},
    )


def _run_baseline(config: dict, purchases: Any, survey_indexed: Any) -> None:
    """DB-free baseline: feed each user's raw purchase titles to the LLM, which
    predicts the demographic attributes directly at DB granularity, then score
    against the survey with the same metric suite as the DB path."""
    user_to_titles = aggregate_titles_per_user(
        purchases=purchases,
        log_first_n=config.get("log_first_n", 3),
    )
    user_ids = list(user_to_titles.keys())
    max_titles = config.get("baseline_max_titles_per_user")
    contexts = [build_baseline_user_context(user_to_titles[uid], max_titles=max_titles) for uid in user_ids]
    logger.info(f"🧱 [baseline] built {len(contexts)} prompt contexts (max_titles_per_user={max_titles})")

    agent = DemographicBaselineAgent(
        model_name=config["model_name"],
        torch_dtype=config["torch_dtype"],
        path_prompt_template=config["baseline_prompt_path"],
        out_dir=config["out_dir"],
        timestamp=config["timestamp"],
        max_new_tokens=config["max_new_tokens"],
        max_model_len=config.get("max_model_len"),
        use_vllm=config.get("use_vllm", True),
        tensor_parallel_size=config.get("tensor_parallel_size", 1),
        enable_cache=config.get("enable_cache", True),
        json_max_tokens=config.get("json_max_tokens", 1024),
        enable_thinking=config.get("enable_thinking"),
    )

    logger.info(f"🚀 [baseline] predicting demographics for {len(user_ids)} users (batch_size={config['batch_size']})")
    predictions = agent.predict_users(user_ids=user_ids, contexts=contexts, batch_size=config["batch_size"])
    user_demo, user_reloc = parse_baseline_predictions(predictions)

    _evaluate_and_save(
        mode="baseline",
        user_demo=user_demo,
        user_reloc=user_reloc,
        survey_indexed=survey_indexed,
        config=config,
        extra_metrics={"model_name": config["model_name"]},
    )


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
    survey_indexed = survey_df.set_index("Survey ResponseID")

    if enable_db:
        _run_db(config, purchases, survey_indexed)

    if enable_baseline:
        _run_baseline(config, purchases, survey_indexed)


if __name__ == "__main__":
    main()
