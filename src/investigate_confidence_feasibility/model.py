"""Feasibility metrics for the LLM-emitted pseudo_confidence field.

For a target tag we have:
  - the set of users who hold the tag (bought >=1 product carrying it), each
    with a representative pseudo_confidence;
  - a survey ground-truth label per user for the corresponding attribute.

We answer: *using only pseudo_confidence among the tagged users, how well can
we recover the m survey positives, and does pseudo_confidence add anything
over plain tag membership?* The report therefore contains, per tag:

  - cohort sizes: n (tagged), m (all survey positives), overlap;
  - a tag-only (confidence-agnostic) baseline = predict every tagged user
    positive;
  - ranking quality: AUC of pseudo_confidence vs the label within the cohort;
  - a confidence threshold sweep (precision / recall-of-m / F1 / selected);
  - precision@k / recall@k for top-confidence users;
  - a calibration view: mean confidence of positives vs negatives, plus the
    positive rate per confidence bin (is higher confidence actually truer?).
"""

from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger
from scipy.stats import ks_2samp, mannwhitneyu
from sklearn.metrics import roc_auc_score

from .dataset import label_for
from .plot import plot_confidence_distributions


def _prf(tp: int, selected: int, m: int) -> dict[str, float]:
    """precision / recall(of m) / f1 for a hard selection of `selected` users
    of which `tp` are true positives, against a positive universe of size m."""
    precision = tp / selected if selected else 0.0
    recall = tp / m if m else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall_of_m": recall, "f1": f1}


def _threshold_sweep(
    scored: list[tuple[float, int]],
    m: int,
    thresholds: list[float],
) -> list[dict[str, Any]]:
    """For each threshold, select tagged users with conf >= thr and score the
    selection against the m-sized positive universe. `scored` is the list of
    (confidence, label) over tagged users that have a valid label."""
    rows: list[dict[str, Any]] = []
    for thr in thresholds:
        sel = [(c, y) for c, y in scored if c >= thr]
        tp = sum(y for _, y in sel)
        rows.append({"threshold": thr, "selected": len(sel), "tp": tp, **_prf(tp, len(sel), m)})
    return rows


def _precision_recall_at_k(
    scored: list[tuple[float, int]],
    m: int,
    ks: list[int],
) -> list[dict[str, Any]]:
    """Rank tagged users by confidence (desc) and report precision@k /
    recall@k(of m) for each k. Ties keep input order, which is stable."""
    ranked = sorted(scored, key=lambda x: x[0], reverse=True)
    rows: list[dict[str, Any]] = []
    for k in ks:
        top = ranked[:k]
        tp = sum(y for _, y in top)
        rows.append(
            {
                "k": k,
                "n_available": len(top),
                "tp": tp,
                "precision_at_k": (tp / len(top) if top else 0.0),
                "recall_at_k_of_m": (tp / m if m else 0.0),
            }
        )
    return rows


def _distribution_separation(
    pos_conf: list[float],
    neg_conf: list[float],
    auc: float,
) -> dict[str, Any]:
    """Quantify how separated the positive- vs negative-label pseudo_confidence
    distributions are, and whether that separation is statistically credible.

    Unequal sample sizes (typically n_pos << n_neg) are not a problem here:
    both tests rank-/ECDF-normalize internally, so the differing denominators
    only affect statistical *power*, not validity.

      - Mann-Whitney U (one-sided, positives > negatives): the significance
        test FOR the AUC already reported, since AUC == U / (n_pos * n_neg).
        Answers "is this separation distinguishable from chance at this cohort
        size?". scipy switches to the exact distribution for small samples.
      - Cliff's delta (= 2*AUC - 1): a sample-size-independent effect size for
        the separation magnitude; comparable across tags.
      - KS two-sample: detects any distributional difference (shape, not just
        location), complementing the mean-of-positive vs mean-of-negative view.
    """
    mw = mannwhitneyu(pos_conf, neg_conf, alternative="greater")
    ks = ks_2samp(pos_conf, neg_conf)
    return {
        "mann_whitney_u": {
            "u_statistic": float(mw.statistic),
            "p_value": float(mw.pvalue),
            "alternative": "positives_greater",
            "note": "significance of the reported AUC; AUC == U / (n_pos * n_neg)",
        },
        "cliffs_delta": 2.0 * auc - 1.0,
        "ks_2samp": {
            "statistic": float(ks.statistic),
            "p_value": float(ks.pvalue),
        },
    }


def _calibration_bins(scored: list[tuple[float, int]], edges: list[float]) -> list[dict[str, Any]]:
    """Positive rate per confidence bin: checks whether higher pseudo_confidence
    corresponds to a higher true-positive rate (monotonicity = well-behaved)."""
    bins: list[dict[str, Any]] = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        # last bin is closed on the right so conf == 1.0 is included
        is_last = hi == edges[-1]
        members = [y for c, y in scored if (lo <= c < hi) or (is_last and c == hi)]
        n = len(members)
        bins.append(
            {
                "range": f"[{lo:.1f},{hi:.1f}{']' if is_last else ')'}",
                "n": n,
                "n_pos": sum(members),
                "positive_rate": (sum(members) / n if n else None),
            }
        )
    return bins


def evaluate_tag(
    spec: dict[str, Any],
    user_conf: dict[str, float],
    survey_indexed: pd.DataFrame,
    thresholds: list[float],
    ks: list[int],
    calibration_edges: list[float],
    plot_dir: str | None = None,
) -> dict[str, Any]:
    """Compute the full feasibility block for one tag spec.

    `spec` carries: tag, category, name, survey_col, match, positive_value.
    `user_conf` is {user_id: representative_confidence} for tag holders.
    """
    col = spec["survey_col"]
    match = spec["match"]
    pos_val = spec["positive_value"]

    # m = positives over ALL survey respondents (confirmed scope: full survey).
    labels_all = survey_indexed[col].map(lambda v: label_for(v, match, pos_val))
    m = int((labels_all == 1).sum())

    # Tagged users that are survey respondents with a usable label.
    scored: list[tuple[float, int]] = []
    n_tagged = 0
    n_tagged_no_label = 0
    for user, conf in user_conf.items():
        n_tagged += 1
        if user not in survey_indexed.index:
            n_tagged_no_label += 1
            continue
        y = label_for(survey_indexed.at[user, col], match, pos_val)
        if y is None:
            n_tagged_no_label += 1
            continue
        scored.append((conf, y))

    n_eval = len(scored)
    overlap = sum(y for _, y in scored)  # true positives captured by the tag

    block: dict[str, Any] = {
        "tag": spec["tag"],
        "attribute": spec["name"],
        "survey_col": col,
        "positive_value": pos_val,
        "n_tagged": n_tagged,
        "n_tagged_evaluable": n_eval,
        "n_tagged_unlabeled": n_tagged_no_label,
        "m_survey_positives": m,
        "tag_true_positives": overlap,
        # Tag-only (confidence-agnostic) baseline: predict every tagged user 1.
        "baseline_tag_only": {
            "tp": overlap,
            "selected": n_eval,
            **_prf(overlap, n_eval, m),
            "note": "predict-positive for every tag holder; recall_of_m here is the tag's recall ceiling",
        },
    }

    if overlap == 0 or overlap == n_eval:
        block["auc"] = None
        block["note"] = "AUC undefined (tagged cohort is single-class on the label)"
    else:
        scores = [c for c, _ in scored]
        ys = [y for _, y in scored]
        block["auc"] = float(roc_auc_score(ys, scores))
        pos_conf = [c for c, y in scored if y == 1]
        neg_conf = [c for c, y in scored if y == 0]
        separation = _distribution_separation(pos_conf, neg_conf, block["auc"])
        block["calibration"] = {
            "mean_conf_positive": sum(pos_conf) / len(pos_conf),
            "mean_conf_negative": sum(neg_conf) / len(neg_conf),
            "distribution_separation": separation,
            "bins": _calibration_bins(scored, calibration_edges),
        }
        if plot_dir is not None:
            plot_path = plot_confidence_distributions(
                pos_conf=pos_conf,
                neg_conf=neg_conf,
                attribute=spec["name"],
                auc=block["auc"],
                separation=separation,
                out_path=str(Path(plot_dir) / f"dist-{spec['name']}.png"),
            )
            if plot_path is not None:
                block["distribution_plot"] = plot_path

    block["threshold_sweep"] = _threshold_sweep(scored, m, thresholds)
    block["precision_recall_at_k"] = _precision_recall_at_k(scored, m, ks)

    # Best-F1 operating point over the sweep, as a single headline number.
    best = max(block["threshold_sweep"], key=lambda r: r["f1"], default=None)
    block["best_f1_operating_point"] = best

    logger.info(f"📊 {spec['name']}: n_tagged={n_tagged} n_eval={n_eval} m={m} tag_TP={overlap} AUC={block['auc']}")
    return block


def evaluate_all_tags(
    tag_specs: list[dict[str, Any]],
    user_tag_conf: dict[str, dict[str, float]],
    survey_df: pd.DataFrame,
    thresholds: list[float],
    ks: list[int],
    calibration_edges: list[float],
    plot_dir: str | None = None,
) -> dict[str, Any]:
    survey_indexed = survey_df.set_index("Survey ResponseID")
    results: dict[str, Any] = {}
    for spec in tag_specs:
        user_conf = user_tag_conf.get(spec["tag"], {})
        results[spec["name"]] = evaluate_tag(
            spec=spec,
            user_conf=user_conf,
            survey_indexed=survey_indexed,
            thresholds=thresholds,
            ks=ks,
            calibration_edges=calibration_edges,
            plot_dir=plot_dir,
        )
    return results
