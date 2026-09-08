"""Investigate whether the LLM-emitted pseudo_confidence field is a useful,
valid signal for recovering self-reported survey attributes.

For each configured (tag -> survey attribute) pair we extract the users who
hold the tag, derive one representative pseudo_confidence per user, and measure
how well that confidence alone recovers the survey positives. See model.py for
the metric definitions. Results are written to
``{out_dir}/output-{timestamp}.json``.
"""

import json
from pathlib import Path

from loguru import logger

from base_agent.config import load_config

from .dataset import (
    build_user_tag_confidence,
    load_survey,
    load_tag_confidences,
)
from .model import evaluate_all_tags


def _latest_output(dir_path: str) -> str:
    """Return the path of the most recently written output-*.json in dir_path."""
    files = sorted(Path(dir_path).glob("output-*.json"))
    if not files:
        raise FileNotFoundError(f"No output-*.json files found in {dir_path}")
    return str(files[-1])


def main() -> None:
    config = load_config()

    # Pin a specific tag DB with tag_db_path; otherwise use the latest tag_cluster output in tag_db_dir.
    tag_db_path = config.get("tag_db_path") or _latest_output(config["tag_db_dir"])
    logger.info(f"📥 tag DB: {tag_db_path}")

    tag_specs: list[dict] = config["tag_targets"]
    tag_to_category = {s["tag"]: s["category"] for s in tag_specs}

    survey_df = load_survey(config["survey_csv"])

    # tag -> {title: max pseudo_confidence} (single streaming pass over the DB)
    tag_title_conf = load_tag_confidences(
        tag_db_path=tag_db_path,
        tag_to_category=tag_to_category,
    )

    # tag -> {user_id: representative pseudo_confidence}
    user_tag_conf = build_user_tag_confidence(
        purchases_csv=config["purchases_csv"],
        tag_title_conf=tag_title_conf,
        agg=config.get("representative_agg", "max"),
    )

    out_dir = Path(config["out_dir"])
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    results = evaluate_all_tags(
        tag_specs=tag_specs,
        user_tag_conf=user_tag_conf,
        survey_df=survey_df,
        thresholds=config.get("thresholds", [0.0, 0.3, 0.5, 0.7, 0.9]),
        ks=config.get("precision_at_ks", [10, 25, 50, 100]),
        calibration_edges=config.get("calibration_edges", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
        plot_dir=str(plot_dir),
    )

    out_path = out_dir / f"output-{config['timestamp']}.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "config": {
                    "representative_agg": config.get("representative_agg", "max"),
                    "population": "all_survey_respondents",
                    "m_definition": "all_survey_positives",
                },
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info(f"💾 saved feasibility report to {out_path}")
    logger.info(f"\n{json.dumps(results, ensure_ascii=False, indent=2)}")


if __name__ == "__main__":
    main()
