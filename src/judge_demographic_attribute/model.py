import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score, precision_score, recall_score
from tqdm import tqdm
from vllm import SamplingParams

from base_agent.model import BaseAgent

# Survey columns holding the self-reported ground truth for each signal.
SIGNAL_TO_SURVEY = {
    "signal:age-bin": "Q-demos-age",
    "signal:gender": "Q-demos-gender",
    "signal:income-bin": "Q-demos-income",
    "signal:education": "Q-demos-education",
}

# Canonical gender classes for the 3-way precision/recall/F1 report.
_GENDER_CLASSES = ["female", "male", "other"]

# Allowed age bins for the baseline (matches the demographic signal taxonomy).
_BASELINE_AGE_BINS = ("18-24", "25-34", "35-44", "45-54", "55-64", "65+")

# Representative decade value (int, from age_representative) → 0-based ordinal
# rank, ascending age. Parallels the ordinal evaluation used for income/education.
# The 6 age bins are evenly spaced at 10-year intervals, so:
#   18-24→20→0, 25-34→30→1, 35-44→40→2, 45-54→50→3, 55-64→60→4, 65+→70→5
_AGE_DECADE_RANK: dict[int, int] = {20: 0, 30: 1, 40: 2, 50: 3, 60: 4, 70: 5}


# ── DB-free baseline: predict demographics directly from purchase titles ─────


def _extract_json_object(raw_text: str) -> dict | None:
    """Best-effort recovery of the JSON object the LLM was asked to emit.

    Tries: (1) raw parse, (2) parse after stripping ``` fences, (3) parse the
    substring between the first '{' and the last '}'. Returns None on failure.
    """
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


class DemographicBaselineAgent(BaseAgent):
    """LLM that predicts a user's demographic attributes directly from raw
    purchase titles (no tag DB). One prompt per user; the model returns a single
    JSON object whose values are constrained to the DB-granularity option sets.
    """

    def __init__(
        self,
        model_name: str,
        torch_dtype: str,
        path_prompt_template: str,
        out_dir: str,
        timestamp: str,
        max_new_tokens: int,
        max_model_len: int | None = None,
        use_vllm: bool = True,
        tensor_parallel_size: int = 1,
        enable_cache: bool = True,
        json_max_tokens: int = 1024,
        enable_thinking: bool | None = None,
    ):
        super().__init__(
            model_name=model_name,
            torch_dtype=torch_dtype,
            path_prompt_template=path_prompt_template,
            out_dir=out_dir,
            timestamp=timestamp,
            max_new_tokens=max_new_tokens,
            max_model_len=max_model_len,
            use_vllm=use_vllm,
            tensor_parallel_size=tensor_parallel_size,
            enable_cache=enable_cache,
            enable_thinking=enable_thinking,
        )
        self._json_max_tokens = json_max_tokens
        return

    def _render_prompt(self, user_context: str) -> str:
        body = self.prompt_template.replace("{query}", user_context)
        chat = [{"role": "user", "content": body}]
        enable_thinking_flag = self.enable_thinking if self.enable_thinking is not None else False
        return self.tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking_flag,
        )

    def predict_users(
        self,
        user_ids: list[str],
        contexts: list[str],
        batch_size: int,
    ) -> list[dict[str, Any]]:
        """Run vLLM inference one batch at a time (one prompt per user) and
        return one record per user with the parsed JSON object under `parsed`
        (None on parse failure) plus the raw output for debugging.
        """
        sampling_params = SamplingParams(temperature=0.0, max_tokens=self._json_max_tokens)

        results: list[dict[str, Any]] = []
        n_total = len(contexts)
        num_total_batch = max(1, (n_total + batch_size - 1) // batch_size)
        pbar = tqdm(range(0, n_total, batch_size), total=num_total_batch, desc="judge_demographic_baseline")

        n_parse_fail = 0
        for step in pbar:
            batch_user_ids = user_ids[step : step + batch_size]
            batch_contexts = contexts[step : step + batch_size]
            prompts = [self._render_prompt(ctx) for ctx in batch_contexts]

            vllm_outputs = self.model.generate(prompts=prompts, sampling_params=sampling_params)

            for user_id, ctx, out in zip(batch_user_ids, batch_contexts, vllm_outputs, strict=False):
                generated = out.outputs[0]
                raw_content = self.tokenizer.decode(list(generated.token_ids), skip_special_tokens=True)
                parsed = _extract_json_object(raw_content)
                if parsed is None:
                    n_parse_fail += 1
                results.append(
                    {
                        "user_id": user_id,
                        "context": ctx,
                        "parsed": parsed,
                        "raw_content": raw_content,
                        "input_tokens": len(out.prompt_token_ids),
                        "output_tokens": len(generated.token_ids),
                    }
                )

        if n_parse_fail:
            logger.warning(
                f"⚠️ JSON parse failed for {n_parse_fail}/{len(results)} baseline responses — "
                f"those users contribute no demographic prediction (treated as no signal)"
            )
        return results


# ── shared helpers ───────────────────────────────────────────────────────────


def _mode_value(counter: Counter) -> str | None:
    """Return the most frequently voted value, or None if the counter is empty.

    Ties are broken by Counter.most_common ordering (insertion order), which is
    deterministic for a fixed aggregation pass.
    """
    if not counter:
        return None
    return counter.most_common(1)[0][0]


# ── age: representative value + Spearman correlation ─────────────────────────


def age_representative(text: object) -> int | None:
    """Collapse an age range/bin to a representative decade value (int).

    Extracts the first number from the text and rounds its units digit to the
    nearest 10 using round-half-up. For a range like "25-34", either
    boundary rounds to the same decade (25 → 30, 34 → 30), so the first is used.
    Examples:
      "25-34"         → 25 → 30
      "65+"           → 65 → 70
      "25 - 34 years" → 25 → 30
      "35-44"         → 35 → 40
    Returns None when no digits are found or input is not a string.
    """
    if not isinstance(text, str):
        return None
    nums = [int(n) for n in re.findall(r"\d+", text)]
    if not nums:
        return None
    # Round the first number to the nearest 10 with half-up.
    return int(math.floor(nums[0] / 10 + 0.5)) * 10


def db_age_representative(age_bin_counter: Counter) -> int | None:
    """Per-user DB age: convert each purchased-product age bin to an int decade
    value, then reduce to one value using mode-with-median-fallback.

    Per product, a bin string (e.g. "25-34") is converted to an int decade via
    age_representative. Per user, the resulting int decades (weighted by
    purchase count) are aggregated as follows:
      1. If one int age has the strictly highest frequency (unique mode), return it.
      2. If multiple int ages share the highest frequency (tie), return the
         weighted median as a rounded int.
    Returns None if the counter is empty or no bins parse to a valid int age.

    # TODO: to inspect the LLM's predicted age distribution across users,
    # collect int_age_counter per user (after the bin-to-int conversion step
    # below) and aggregate them into a population-level Counter keyed by decade.
    # This reveals whether the model systematically over- or under-predicts
    # certain age groups before the mode/median reduction is applied.
    # Example: pass int_age_counter back to the caller and accumulate into
    # a shared Counter in compute_age_correlation, then save alongside metrics.
    """
    if not age_bin_counter:
        return None

    # Convert bin strings to int decade values, accumulating purchase-count weights.
    int_age_counter: Counter = Counter()
    for bin_str, count in age_bin_counter.items():
        age_val = age_representative(str(bin_str))
        if age_val is not None:
            int_age_counter[age_val] += count

    if not int_age_counter:
        return None

    max_count = max(int_age_counter.values())
    modes = [age for age, cnt in int_age_counter.items() if cnt == max_count]

    if len(modes) == 1:
        return modes[0]

    # Tie — fall back to weighted median of all int age values.
    weighted: list[int] = []
    for age_val, count in int_age_counter.items():
        weighted.extend([age_val] * count)
    weighted.sort()
    n = len(weighted)
    mid = n // 2
    if n % 2 == 1:
        return weighted[mid]
    return int(round((weighted[mid - 1] + weighted[mid]) / 2))


def compute_age_correlation(
    user_demo: dict[str, dict[str, Counter]],
    survey_indexed: pd.DataFrame,
) -> dict[str, Any]:
    """Pair each user's self-reported age (survey) with the DB-predicted age and
    compute Spearman rank correlation and MAE between the two int decade series.

    Both the survey and DB ages are reduced to a single int decade per user via
    age_representative / db_age_representative before comparison. Users without
    any DB age signal are counted and excluded from all metrics.

    Returns a dict containing:
      - n_users_paired           : number of users used in Spearman / MAE
      - n_users_skipped_no_db_age: users excluded (no age signal in DB)
      - spearman_r / p_value     : Spearman correlation (None if n < 2)
      - mean_absolute_error      : mean |db_age - survey_age| in years (None if n < 2)
      - ordinal_mae              : mean |db_rank - survey_rank| in rank steps,
                                   where each age bin is one rank (None if n < 2)
      - n_users_ordinal          : users used for ordinal_mae (both decades in
                                   the 6-bin rank table)
      - survey_ages / db_ages    : raw paired int lists for scatter plot

    Interpretation: `mean_absolute_error` measures the average miss in real years
    (a representative-value MAE — each bin is collapsed to its decade midpoint),
    while `ordinal_mae` measures the average miss in number of age bands. Because
    the 6 bins are uniformly 10 years wide, the two are tied by ordinal_mae ≈
    mean_absolute_error / 10 (e.g. ordinal_mae = 1.0 ⇔ off by one band ⇔ ~10 yr).
    Reporting both keeps age comparable to the ordinal-MAE-based income/education
    metrics while preserving the intuitive "years off" reading.
    """
    survey_col = SIGNAL_TO_SURVEY["signal:age-bin"]
    survey_ages: list[int] = []
    db_ages: list[int] = []
    n_no_db_age = 0

    for user_id, per_category in user_demo.items():
        age_counter = per_category.get("signal:age-bin")
        if not age_counter:
            n_no_db_age += 1
            continue
        db_age = db_age_representative(age_counter)
        if db_age is None:
            n_no_db_age += 1
            continue
        if user_id not in survey_indexed.index:
            continue
        survey_raw = survey_indexed.loc[user_id, survey_col]
        if not isinstance(survey_raw, str) and pd.isna(survey_raw):
            continue
        survey_age = age_representative(str(survey_raw))
        if survey_age is None:
            continue
        survey_ages.append(survey_age)
        db_ages.append(db_age)

    n = len(survey_ages)
    logger.info(f"📊 age evaluation: {n} users paired, {n_no_db_age} users skipped (no DB age signal)")

    result: dict[str, Any] = {
        "n_users_paired": n,
        "n_users_skipped_no_db_age": n_no_db_age,
        "survey_ages": survey_ages,
        "db_ages": db_ages,
        "spearman_r": None,
        "p_value": None,
        "mean_absolute_error": None,
        "ordinal_mae": None,
        "n_users_ordinal": 0,
    }

    if n < 2:
        logger.warning(f"⚠️ age correlation: only {n} paired users — Spearman and MAE skipped")
        return result

    rho, p_value = spearmanr(survey_ages, db_ages)
    mae = float(np.mean(np.abs(np.array(db_ages, dtype=float) - np.array(survey_ages, dtype=float))))

    # Ordinal MAE (in age-band steps), parallel to the income/education ordinal
    # eval. Pairs whose decade falls outside the 6-bin rank table are skipped.
    ordinal_pairs = [
        (_AGE_DECADE_RANK[s], _AGE_DECADE_RANK[d])
        for s, d in zip(survey_ages, db_ages, strict=True)
        if s in _AGE_DECADE_RANK and d in _AGE_DECADE_RANK
    ]
    n_ordinal = len(ordinal_pairs)
    if n_ordinal:
        sr = np.array([p[0] for p in ordinal_pairs], dtype=float)
        dr = np.array([p[1] for p in ordinal_pairs], dtype=float)
        ordinal_mae = float(np.mean(np.abs(dr - sr)))
    else:
        ordinal_mae = None

    result.update(
        {
            "spearman_r": float(rho),
            "p_value": float(p_value),
            "mean_absolute_error": mae,
            "ordinal_mae": ordinal_mae,
            "n_users_ordinal": n_ordinal,
        }
    )

    extended = _age_extended_metrics(survey_ages, db_ages)
    result.update(extended)

    acc = extended["accuracy_tiers"]
    ordinal_str = f"{ordinal_mae:.3f} band steps (n={n_ordinal})" if ordinal_mae is not None else "n/a"
    logger.info(
        f"📋 age evaluation summary:\n"
        f"    paired users             : {n}\n"
        f"    skipped (no DB age)      : {n_no_db_age}\n"
        f"    Spearman ρ               : {rho:.4f}\n"
        f"    p-value                  : {p_value:.3e}\n"
        f"    mean absolute error      : {mae:.2f} years\n"
        f"    ordinal MAE              : {ordinal_str}\n"
        f"    bias (mean signed error) : {extended['bias_mean']:+.2f} years "
        f"(σ={extended['bias_std']:.2f})\n"
        f"    exact match              : {acc['exact_match']:.1%} "
        f"(n={acc['n_exact']})\n"
        f"    within ±10 yr            : {acc['within_10yr']:.1%} "
        f"(n={acc['n_within_10yr']})\n"
        f"    within ±20 yr            : {acc['within_20yr']:.1%} "
        f"(n={acc['n_within_20yr']})\n"
        f"    weighted κ (linear)      : {extended['weighted_kappa_linear']}\n"
        f"    weighted κ (quadratic)   : {extended['weighted_kappa_quadratic']}"
    )
    return result


def plot_age_scatter(
    survey_ages: list[int],
    db_ages: list[int],
    out_path: str,
    spearman_r: float | None = None,
) -> str | None:
    """Scatter of survey age (x) vs DB-predicted age (y) with a linear fit line.

    Both axes are discrete representative values, so a small deterministic jitter
    is added to the displayed points to reveal overlapping mass; the fit line is
    computed on the un-jittered values. Returns the path, or None if matplotlib
    is unavailable or there is nothing to plot.
    """
    if len(survey_ages) < 2:
        logger.warning("Not enough points to plot age scatter")
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover - environment-dependent
        logger.warning(f"matplotlib unavailable, skipping plot: {e}")
        return None

    x = np.asarray(survey_ages, dtype=float)
    y = np.asarray(db_ages, dtype=float)

    rng = np.random.default_rng(3407)
    jx = x + rng.uniform(-1.5, 1.5, size=x.shape)
    jy = y + rng.uniform(-1.5, 1.5, size=y.shape)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(jx, jy, s=12, alpha=0.25, color="#0072B2", edgecolor="none", label=f"users (n={len(x)})")

    # Linear fit (degree 1) on the raw representative values.
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, slope * xs + intercept, color="#D55E00", linewidth=2, label=f"fit: y={slope:.2f}x+{intercept:.1f}")

    lo = min(x.min(), y.min()) - 5
    hi = max(x.max(), y.max()) + 5
    ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--", linewidth=1, label="y=x")

    title = "Self-reported vs DB-predicted age"
    if spearman_r is not None:
        title += f"  (Spearman ρ={spearman_r:.3f})"
    ax.set_title(title)
    ax.set_xlabel("survey age (representative)")
    ax.set_ylabel("DB-predicted age (representative)")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"🖼️ Wrote age scatter plot: {out_path}")
    return out_path


# ── age: extended metrics (bias, accuracy tiers, weighted kappa, confusion matrix) ──


def _age_extended_metrics(
    survey_ages: list[int],
    db_ages: list[int],
) -> dict[str, Any]:
    """Bias, accuracy tiers, and weighted Cohen's κ for the paired age lists.

    Returns a flat dict merged into the main age result by compute_age_correlation.
    """
    err = np.array(db_ages, dtype=float) - np.array(survey_ages, dtype=float)
    abs_err = np.abs(err)
    n = len(survey_ages)

    exact = int(np.sum(abs_err == 0))
    within_10 = int(np.sum(abs_err <= 10))
    within_20 = int(np.sum(abs_err <= 20))

    try:
        kappa_linear = float(cohen_kappa_score(survey_ages, db_ages, weights="linear"))
        kappa_quadratic = float(cohen_kappa_score(survey_ages, db_ages, weights="quadratic"))
    except Exception as e:
        logger.warning(f"⚠️ weighted kappa computation failed: {e}")
        kappa_linear = None
        kappa_quadratic = None

    return {
        "bias_mean": float(np.mean(err)),
        "bias_std": float(np.std(err)),
        "accuracy_tiers": {
            "exact_match": exact / n,
            "within_10yr": within_10 / n,
            "within_20yr": within_20 / n,
            "n_exact": exact,
            "n_within_10yr": within_10,
            "n_within_20yr": within_20,
        },
        "weighted_kappa_linear": kappa_linear,
        "weighted_kappa_quadratic": kappa_quadratic,
    }


def plot_age_confusion_matrix(
    survey_ages: list[int],
    db_ages: list[int],
    out_path: str,
) -> str | None:
    """Heatmap of the confusion matrix: survey age (true, rows) vs DB-predicted
    age (columns). Each cell shows the user count; the diagonal is correct
    predictions. Returns the path, or None if matplotlib is unavailable or there
    is insufficient data.
    """
    if len(survey_ages) < 2:
        logger.warning("Not enough data to plot age confusion matrix")
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover - environment-dependent
        logger.warning(f"matplotlib unavailable, skipping confusion matrix: {e}")
        return None

    labels = sorted(set(survey_ages) | set(db_ages))
    cm = confusion_matrix(survey_ages, db_ages, labels=labels)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, label="user count")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("DB-predicted age")
    ax.set_ylabel("survey age (true)")
    ax.set_title(f"Age confusion matrix  (n={len(survey_ages)})")

    thresh = cm.max() / 2
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=9,
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"🖼️ Wrote age confusion matrix: {out_path}")
    return out_path


# ── gender: precision / recall / F1 over {male, female, other} ───────────────


def normalize_survey_gender(text: object) -> str | None:
    """Map a survey gender label to a canonical class, or None to skip.

    'Female'/'Male'/'Other' → lowercase class; 'Prefer not to say' (and anything
    unrecognized) → None so it is excluded from the metric.
    """
    if not isinstance(text, str):
        return None
    v = text.strip().lower()
    if v in _GENDER_CLASSES:
        return v
    return None


def _gender_score_block(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
) -> dict[str, Any]:
    """Per-class + macro/micro precision/recall/F1 over the given label set."""
    n = len(y_true)
    if n == 0:
        return {"n": 0}

    per_class_p = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class_r = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class_f = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class = {
        cls: {
            "precision": float(per_class_p[i]),
            "recall": float(per_class_r[i]),
            "f1": float(per_class_f[i]),
            "support_true": int(sum(1 for t in y_true if t == cls)),
            "n_pred": int(sum(1 for p in y_pred if p == cls)),
        }
        for i, cls in enumerate(labels)
    }
    return {
        "n": n,
        "labels": list(labels),
        "accuracy": float(sum(t == p for t, p in zip(y_true, y_pred, strict=True)) / n),
        "per_class": per_class,
        "macro": {
            "precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        },
        "micro": {
            "precision": float(precision_score(y_true, y_pred, labels=labels, average="micro", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, labels=labels, average="micro", zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, labels=labels, average="micro", zero_division=0)),
        },
    }


def compute_gender_metrics(
    user_demo: dict[str, dict[str, Counter]],
    survey_indexed: pd.DataFrame,
) -> dict[str, Any]:
    """Precision / recall / F1 for the gender prediction, reported two ways.

    DB prediction is the majority-voted gender across the user's purchases;
    ground truth is the (normalized) survey answer. The rare `other` class is
    poorly represented in the DB, so two scoring blocks are returned:
      - `with_other`: full 3-way eval over {female, male, other}.
      - `without_other`: binary eval over {female, male} only — any pair whose
        survey label OR DB prediction is `other` is dropped first.
    """
    survey_col = SIGNAL_TO_SURVEY["signal:gender"]
    y_true: list[str] = []
    y_pred: list[str] = []

    for user_id, per_category in user_demo.items():
        gender_counter = per_category.get("signal:gender")
        if not gender_counter:
            continue
        db_gender = _mode_value(gender_counter)
        if db_gender not in _GENDER_CLASSES:
            continue
        if user_id not in survey_indexed.index:
            continue
        true_gender = normalize_survey_gender(survey_indexed.loc[user_id, survey_col])
        if true_gender is None:
            continue
        y_true.append(true_gender)
        y_pred.append(db_gender)

    if not y_true:
        logger.warning("⚠️ gender metrics: no paired users")
        return {"n_total_pairs": 0, "with_other": {"n": 0}, "without_other": {"n": 0}}

    binary_classes = [c for c in _GENDER_CLASSES if c != "other"]
    bin_true: list[str] = []
    bin_pred: list[str] = []
    for t, p in zip(y_true, y_pred, strict=True):
        if t == "other" or p == "other":
            continue
        bin_true.append(t)
        bin_pred.append(p)

    metrics: dict[str, Any] = {
        "n_total_pairs": len(y_true),
        "with_other": _gender_score_block(y_true, y_pred, _GENDER_CLASSES),
        "without_other": _gender_score_block(bin_true, bin_pred, binary_classes),
    }
    logger.info(
        f"🚻 gender with_other (n={metrics['with_other']['n']}) "
        f"macro-F1={metrics['with_other']['macro']['f1']:.4f} "
        f"accuracy={metrics['with_other']['accuracy']:.4f}"
    )
    logger.info(
        f"🚻 gender without_other (n={metrics['without_other']['n']}) "
        f"macro-F1={metrics['without_other']['macro']['f1']:.4f} "
        f"accuracy={metrics['without_other']['accuracy']:.4f}"
    )
    return metrics


# ── income / education: data collection only (evaluation TBD) ────────────────


def collect_income_education(
    user_demo: dict[str, dict[str, Counter]],
    survey_indexed: pd.DataFrame,
) -> dict[str, Any]:
    """Collect, per user, the DB majority vote and the self-reported survey value
    for income-bin and education. No scoring yet — the evaluation method for
    these two is decided after the age/gender pass (spec step 9).

    Returns {category -> {"survey_col", "records": [...], "db_value_counts",
    "survey_value_counts"}} where each record is a per-user pairing.
    """
    out: dict[str, Any] = {}
    for category in ("signal:income-bin", "signal:education"):
        survey_col = SIGNAL_TO_SURVEY[category]
        records: list[dict[str, Any]] = []
        db_value_counts: Counter = Counter()
        survey_value_counts: Counter = Counter()

        for user_id, per_category in user_demo.items():
            counter = per_category.get(category)
            if not counter:
                continue
            db_value = _mode_value(counter)
            survey_value = survey_indexed.loc[user_id, survey_col] if user_id in survey_indexed.index else None
            if isinstance(survey_value, float) and pd.isna(survey_value):
                survey_value = None
            records.append(
                {
                    "user_id": user_id,
                    "db_mode_value": db_value,
                    "db_value_distribution": dict(counter),
                    "survey_value": survey_value,
                }
            )
            if db_value is not None:
                db_value_counts[db_value] += 1
            if survey_value is not None:
                survey_value_counts[str(survey_value)] += 1

        out[category] = {
            "survey_col": survey_col,
            "n": len(records),
            "db_value_counts": dict(db_value_counts),
            "survey_value_counts": dict(survey_value_counts),
            "records": records,
        }
        logger.info(
            f"🗃️ collected {category}: {len(records)} users with a DB value "
            f"(DB distinct={len(db_value_counts)}, survey distinct={len(survey_value_counts)})"
        )
    return out


# ── income-bin: ordinal evaluation ───────────────────────────────────────────

# DB and survey both span 6 income bins, mapped to 0-based ordinal ranks
# (ascending income). "Prefer not to say" is excluded. (The prediction prompt
# emits 1-3 adjacent bins per user; the per-user mode is ranked here.)
_INCOME_DB_RANK: dict[str, int] = {
    "<25k": 0,
    "25-49k": 1,
    "50-74k": 2,
    "75-99k": 3,
    "100-149k": 4,
    "150k+": 5,
}

_INCOME_SURVEY_RANK: dict[str, int] = {
    "Less than $25,000": 0,
    "$25,000 - $49,999": 1,
    "$50,000 - $74,999": 2,
    "$75,000 - $99,999": 3,
    "$100,000 - $149,999": 4,
    "$150,000 or more": 5,
}

_INCOME_RANK_LABEL: dict[int, str] = {
    0: "<$25k",
    1: "$25-49k",
    2: "$50-74k",
    3: "$75-99k",
    4: "$100-149k",
    5: "$150k+",
}

# Representative annual-income value (USD) per ordinal rank, used to compute a
# dollar-scale MAE alongside the ordinal (rank-step) MAE. Each closed band uses
# its midpoint from the band labels:
#   <25k → (0+25)/2 ≈ 12k,  25-49k → (25+49)/2 = 37k,  50-74k → 62k,
#   75-99k → 87k,  100-149k → (100+149)/2 ≈ 124k.
# The open top band (150k+) has no upper bound; we extrapolate one band-width
# (≈25k, half of the 50k-wide band below) above the threshold → 175k.
_INCOME_RANK_REPRESENTATIVE: dict[int, float] = {
    0: 12_000.0,
    1: 37_000.0,
    2: 62_000.0,
    3: 87_000.0,
    4: 124_000.0,
    5: 175_000.0,
}


def compute_income_metrics(  # noqa: C901
    user_demo: dict[str, dict[str, Counter]],
    survey_indexed: pd.DataFrame,
) -> dict[str, Any]:
    """Ordinal evaluation of income-bin predictions.

    DB and survey labels are mapped to ordinal ranks (0-based, ascending
    income). Spearman rank correlation, ordinal MAE (in rank steps), weighted
    Cohen's κ, and a dollar-scale MAE are computed over the paired sample.

    The dollar MAE collapses each band to its representative income value
    (_INCOME_RANK_REPRESENTATIVE) and averages |db_value - survey_value|. Whereas
    `ordinal_mae` treats every adjacent band as one equal step, `dollar_mae`
    reflects that the bands are not equally wide in dollars (e.g. the 100-149k
    band is twice as wide as 25-49k, and the top band is open-ended), giving a
    more interpretable "average miss in USD". Reporting both mirrors the age
    metric, which carries a years-scale MAE plus an ordinal (band-step) MAE.

    Returns a dict with Spearman r/p, ordinal MAE, dollar MAE, weighted kappa,
    sample sizes, and the raw paired rank lists for the confusion matrix plot.
    """
    survey_col = SIGNAL_TO_SURVEY["signal:income-bin"]
    db_ranks: list[int] = []
    survey_ranks: list[int] = []
    n_no_db_income = 0
    n_prefer_not_to_say = 0
    n_survey_above_db_max = 0
    db_max_rank = max(_INCOME_DB_RANK.values())

    for user_id, per_category in user_demo.items():
        income_counter = per_category.get("signal:income-bin")
        if not income_counter:
            n_no_db_income += 1
            continue
        db_mode = _mode_value(income_counter)
        db_rank = _INCOME_DB_RANK.get(str(db_mode)) if db_mode else None
        if db_rank is None:
            n_no_db_income += 1
            continue
        if user_id not in survey_indexed.index:
            continue
        survey_raw = survey_indexed.loc[user_id, survey_col]
        if not isinstance(survey_raw, str):
            if pd.isna(survey_raw):
                continue
            survey_raw = str(survey_raw)
        survey_rank = _INCOME_SURVEY_RANK.get(survey_raw.strip())
        if survey_rank is None:
            n_prefer_not_to_say += 1
            continue
        if survey_rank > db_max_rank:
            n_survey_above_db_max += 1
        db_ranks.append(db_rank)
        survey_ranks.append(survey_rank)

    n = len(db_ranks)
    logger.info(
        f"💰 income evaluation: {n} users paired, "
        f"{n_no_db_income} skipped (no DB income signal), "
        f"{n_prefer_not_to_say} excluded (Prefer not to say), "
        f"{n_survey_above_db_max} survey above DB max rank ($100k+)"
    )

    result: dict[str, Any] = {
        "n_users_paired": n,
        "n_users_skipped_no_db_income": n_no_db_income,
        "n_users_excluded_prefer_not_to_say": n_prefer_not_to_say,
        "n_survey_above_db_max": n_survey_above_db_max,
        "db_ranks": db_ranks,
        "survey_ranks": survey_ranks,
        "spearman_r": None,
        "p_value": None,
        "ordinal_mae": None,
        "dollar_mae": None,
        "weighted_kappa_linear": None,
        "weighted_kappa_quadratic": None,
    }

    if n < 2:
        logger.warning(f"⚠️ income metrics: only {n} paired users — metrics skipped")
        return result

    rho, p_value = spearmanr(survey_ranks, db_ranks)
    mae = float(np.mean(np.abs(np.array(db_ranks, dtype=float) - np.array(survey_ranks, dtype=float))))

    # Dollar-scale MAE: map each rank to its representative income value, then
    # average the absolute USD difference.
    db_dollars = np.array([_INCOME_RANK_REPRESENTATIVE[r] for r in db_ranks], dtype=float)
    survey_dollars = np.array([_INCOME_RANK_REPRESENTATIVE[r] for r in survey_ranks], dtype=float)
    dollar_mae = float(np.mean(np.abs(db_dollars - survey_dollars)))

    try:
        kappa_linear = float(cohen_kappa_score(survey_ranks, db_ranks, weights="linear"))
        kappa_quadratic = float(cohen_kappa_score(survey_ranks, db_ranks, weights="quadratic"))
    except Exception as e:
        logger.warning(f"⚠️ income weighted kappa failed: {e}")
        kappa_linear = None
        kappa_quadratic = None

    result.update(
        {
            "spearman_r": float(rho),
            "p_value": float(p_value),
            "ordinal_mae": mae,
            "dollar_mae": dollar_mae,
            "weighted_kappa_linear": kappa_linear,
            "weighted_kappa_quadratic": kappa_quadratic,
        }
    )

    logger.info(
        f"📋 income evaluation summary:\n"
        f"    paired users             : {n}\n"
        f"    skipped (no DB income)   : {n_no_db_income}\n"
        f"    excluded (prefer not say): {n_prefer_not_to_say}\n"
        f"    survey above DB max      : {n_survey_above_db_max} ($100k+)\n"
        f"    Spearman ρ               : {rho:.4f}\n"
        f"    p-value                  : {p_value:.3e}\n"
        f"    ordinal MAE              : {mae:.3f} rank steps\n"
        f"    dollar MAE               : ${dollar_mae:,.0f}\n"
        f"    weighted κ (linear)      : {kappa_linear}\n"
        f"    weighted κ (quadratic)   : {kappa_quadratic}"
    )
    return result


def plot_income_confusion_matrix(
    survey_ranks: list[int],
    db_ranks: list[int],
    out_path: str,
) -> str | None:
    """Heatmap of income confusion matrix: survey income rank (true, rows) vs
    DB-predicted income rank (columns). Axes are annotated with income labels.
    Returns the path, or None if matplotlib is unavailable or data is insufficient.
    """
    if len(survey_ranks) < 2:
        logger.warning("Not enough data to plot income confusion matrix")
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover - environment-dependent
        logger.warning(f"matplotlib unavailable, skipping income confusion matrix: {e}")
        return None

    all_ranks = sorted(set(survey_ranks) | set(db_ranks))
    cm = confusion_matrix(survey_ranks, db_ranks, labels=all_ranks)
    tick_labels = [_INCOME_RANK_LABEL.get(r, str(r)) for r in all_ranks]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, label="user count")

    ax.set_xticks(range(len(all_ranks)))
    ax.set_yticks(range(len(all_ranks)))
    ax.set_xticklabels(tick_labels, rotation=30, ha="right")
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel("DB-predicted income")
    ax.set_ylabel("survey income (true)")
    ax.set_title(f"Income confusion matrix  (n={len(survey_ranks)})")

    thresh = cm.max() / 2
    for i in range(len(all_ranks)):
        for j in range(len(all_ranks)):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=9,
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"🖼️ Wrote income confusion matrix: {out_path}")
    return out_path


# ── education: ordinal evaluation ────────────────────────────────────────────

# DB and survey both span 4 education levels, mapped to 0-based ordinal ranks
# (ascending education). DB `<high-school` ↔ survey "Some high school or less"
# (rank 0), `high-school` ↔ "High school diploma or GED" (rank 1). The DB-only
# `student` value stays unmapped (dropped). "Prefer not to say" is excluded.
_EDUCATION_DB_RANK: dict[str, int] = {
    "<high-school": 0,
    "high-school": 1,
    "bachelor": 2,
    "graduate": 3,
}

_EDUCATION_SURVEY_RANK: dict[str, int] = {
    "Some high school or less": 0,
    "High school diploma or GED": 1,
    "Bachelor's degree": 2,
    "Graduate or professional degree (MA, MS, MBA, PhD, JD, MD, DDS, etc)": 3,
}

_EDUCATION_RANK_LABEL: dict[int, str] = {
    0: "< High school",
    1: "High school",
    2: "Bachelor's",
    3: "Graduate",
}


def compute_education_metrics(  # noqa: C901
    user_demo: dict[str, dict[str, Counter]],
    survey_indexed: pd.DataFrame,
) -> dict[str, Any]:
    """Ordinal evaluation of education predictions.

    DB and survey labels are mapped to ordinal ranks (0-based, ascending
    education level). Spearman rank correlation, ordinal MAE (in rank steps),
    and weighted Cohen's κ are computed over the paired sample.

    Note: the DB has 3 levels while the survey has 4. DB `high-school` (rank 0)
    conflates survey ranks 0 and 1 — n_survey_some_high_school counts users
    for whom this granularity gap applies.
    """
    survey_col = SIGNAL_TO_SURVEY["signal:education"]
    db_ranks: list[int] = []
    survey_ranks: list[int] = []
    n_no_db_education = 0
    n_prefer_not_to_say = 0
    n_survey_some_high_school = 0

    for user_id, per_category in user_demo.items():
        edu_counter = per_category.get("signal:education")
        if not edu_counter:
            n_no_db_education += 1
            continue
        db_mode = _mode_value(edu_counter)
        db_rank = _EDUCATION_DB_RANK.get(str(db_mode)) if db_mode else None
        if db_rank is None:
            n_no_db_education += 1
            continue
        if user_id not in survey_indexed.index:
            continue
        survey_raw = survey_indexed.loc[user_id, survey_col]
        if not isinstance(survey_raw, str):
            if pd.isna(survey_raw):
                continue
            survey_raw = str(survey_raw)
        survey_rank = _EDUCATION_SURVEY_RANK.get(survey_raw.strip())
        if survey_rank is None:
            n_prefer_not_to_say += 1
            continue
        if survey_rank == 0:
            n_survey_some_high_school += 1
        db_ranks.append(db_rank)
        survey_ranks.append(survey_rank)

    n = len(db_ranks)
    logger.info(
        f"🎓 education evaluation: {n} users paired, "
        f"{n_no_db_education} skipped (no DB education signal), "
        f"{n_prefer_not_to_say} excluded (Prefer not to say), "
        f"{n_survey_some_high_school} survey 'Some high school' (below DB granularity)"
    )

    result: dict[str, Any] = {
        "n_users_paired": n,
        "n_users_skipped_no_db_education": n_no_db_education,
        "n_users_excluded_prefer_not_to_say": n_prefer_not_to_say,
        "n_survey_some_high_school": n_survey_some_high_school,
        "db_ranks": db_ranks,
        "survey_ranks": survey_ranks,
        "spearman_r": None,
        "p_value": None,
        "ordinal_mae": None,
        "weighted_kappa_linear": None,
        "weighted_kappa_quadratic": None,
    }

    if n < 2:
        logger.warning(f"⚠️ education metrics: only {n} paired users — metrics skipped")
        return result

    rho, p_value = spearmanr(survey_ranks, db_ranks)
    mae = float(np.mean(np.abs(np.array(db_ranks, dtype=float) - np.array(survey_ranks, dtype=float))))

    try:
        kappa_linear = float(cohen_kappa_score(survey_ranks, db_ranks, weights="linear"))
        kappa_quadratic = float(cohen_kappa_score(survey_ranks, db_ranks, weights="quadratic"))
    except Exception as e:
        logger.warning(f"⚠️ education weighted kappa failed: {e}")
        kappa_linear = None
        kappa_quadratic = None

    result.update(
        {
            "spearman_r": float(rho),
            "p_value": float(p_value),
            "ordinal_mae": mae,
            "weighted_kappa_linear": kappa_linear,
            "weighted_kappa_quadratic": kappa_quadratic,
        }
    )

    logger.info(
        f"📋 education evaluation summary:\n"
        f"    paired users             : {n}\n"
        f"    skipped (no DB education): {n_no_db_education}\n"
        f"    excluded (prefer not say): {n_prefer_not_to_say}\n"
        f"    survey 'Some high school': {n_survey_some_high_school} (below DB granularity)\n"
        f"    Spearman ρ               : {rho:.4f}\n"
        f"    p-value                  : {p_value:.3e}\n"
        f"    ordinal MAE              : {mae:.3f} rank steps\n"
        f"    weighted κ (linear)      : {kappa_linear}\n"
        f"    weighted κ (quadratic)   : {kappa_quadratic}"
    )
    return result


def plot_education_confusion_matrix(
    survey_ranks: list[int],
    db_ranks: list[int],
    out_path: str,
) -> str | None:
    """Heatmap of education confusion matrix: survey education rank (true, rows)
    vs DB-predicted education rank (columns). Axes are annotated with labels.
    Returns the path, or None if matplotlib is unavailable or data is insufficient.
    """
    if len(survey_ranks) < 2:
        logger.warning("Not enough data to plot education confusion matrix")
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover - environment-dependent
        logger.warning(f"matplotlib unavailable, skipping education confusion matrix: {e}")
        return None

    all_ranks = sorted(set(survey_ranks) | set(db_ranks))
    cm = confusion_matrix(survey_ranks, db_ranks, labels=all_ranks)
    tick_labels = [_EDUCATION_RANK_LABEL.get(r, str(r)) for r in all_ranks]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, label="user count")

    ax.set_xticks(range(len(all_ranks)))
    ax.set_yticks(range(len(all_ranks)))
    ax.set_xticklabels(tick_labels, rotation=30, ha="right")
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel("DB-predicted education")
    ax.set_ylabel("survey education (true)")
    ax.set_title(f"Education confusion matrix  (n={len(survey_ranks)})")

    thresh = cm.max() / 2
    for i in range(len(all_ranks)):
        for j in range(len(all_ranks)):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=9,
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"🖼️ Wrote education confusion matrix: {out_path}")
    return out_path


# ── relocation: binary classification ────────────────────────────────────────

_SURVEY_RELOCATION_COL = "Q-life-changes"
_SURVEY_RELOCATION_VALUE = "Moved place of residence"


def compute_relocation_metrics(
    user_reloc: dict[str, bool],
    survey_indexed: pd.DataFrame,
) -> dict[str, Any]:
    """Binary classification metrics for the relocation signal.

    Prediction (positive): user has ≥1 purchased product with
    `signal:recent-relocation = 'positive'` in the DB (OR aggregation).
    Ground truth (positive): user's Q-life-changes survey answer contains
    'Moved place of residence'.

    Returns Precision, Recall, F1, and raw confusion matrix counts (TP/FP/FN/TN).
    """
    y_true: list[int] = []
    y_pred: list[int] = []
    n_no_survey = 0

    for user_id, predicted_moved in user_reloc.items():
        if user_id not in survey_indexed.index:
            n_no_survey += 1
            continue
        survey_raw = survey_indexed.loc[user_id, _SURVEY_RELOCATION_COL]
        if not isinstance(survey_raw, str) and pd.isna(survey_raw):
            true_moved = False
        else:
            true_moved = _SURVEY_RELOCATION_VALUE in str(survey_raw)
        y_true.append(int(true_moved))
        y_pred.append(int(predicted_moved))

    n = len(y_true)
    n_true_moved = sum(y_true)
    n_predicted_moved = sum(y_pred)

    logger.info(
        f"🏠 relocation pairing: {n} users, "
        f"{n_no_survey} skipped (no survey), "
        f"survey positive={n_true_moved}, DB predicted positive={n_predicted_moved}"
    )

    result: dict[str, Any] = {
        "n_users_paired": n,
        "n_no_survey": n_no_survey,
        "n_true_moved_survey": n_true_moved,
        "n_predicted_moved_db": n_predicted_moved,
        "precision": None,
        "recall": None,
        "f1": None,
        "tp": None,
        "fp": None,
        "fn": None,
        "tn": None,
    }

    if n == 0:
        logger.warning("⚠️ relocation metrics: no paired users — metrics skipped")
        return result
    if n_predicted_moved == 0:
        # zero_division=0 keeps precision/recall/F1 well-defined (0.0), so a
        # method that never predicts relocation still gets comparable scores.
        logger.warning("⚠️ relocation metrics: no positive DB predictions (scores fall back to 0.0)")

    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    result.update(
        {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        }
    )

    logger.info(
        f"📋 relocation evaluation summary:\n"
        f"    paired users             : {n}\n"
        f"    no survey data           : {n_no_survey}\n"
        f"    true moved (survey)      : {n_true_moved}\n"
        f"    predicted moved (DB)     : {n_predicted_moved}\n"
        f"    TP={tp}  FP={fp}  FN={fn}  TN={tn}\n"
        f"    precision                : {prec:.4f}\n"
        f"    recall                   : {rec:.4f}\n"
        f"    F1                       : {f1:.4f}"
    )
    return result


# ── baseline output → evaluation structures ──────────────────────────────────

# JSON key (in the baseline prompt output) → demographic category used by the
# DB-evaluation metric functions.
_BASELINE_KEY_TO_CATEGORY: dict[str, str] = {
    "age_bin": "signal:age-bin",
    "gender": "signal:gender",
    "income_bin": "signal:income-bin",
    "education": "signal:education",
}


def parse_baseline_predictions(
    predictions: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Counter]], dict[str, bool]]:
    """Convert raw baseline LLM outputs into the same structures the DB path
    produces, so every compute_*/plot_* function is reused unchanged:

      - user_demo : {user_id -> {category -> Counter(value -> 1)}}
        The baseline emits one value per attribute per user, so each Counter has
        a single entry. Values outside the DB-granularity option set (or
        "unknown"/missing/parse-fail) yield an absent category — exactly how the
        DB path represents "no signal".
      - user_reloc: {user_id -> bool}  (relocation == "positive")

    Values are lower-cased and matched against the allowed sets:
      age-bin ∈ _BASELINE_AGE_BINS, gender ∈ _GENDER_CLASSES,
      income-bin ∈ _INCOME_DB_RANK keys, education ∈ _EDUCATION_DB_RANK keys.
    """
    allowed: dict[str, set[str]] = {
        "signal:age-bin": set(_BASELINE_AGE_BINS),
        "signal:gender": set(_GENDER_CLASSES),
        "signal:income-bin": set(_INCOME_DB_RANK),
        "signal:education": set(_EDUCATION_DB_RANK),
    }

    user_demo: dict[str, dict[str, Counter]] = {}
    user_reloc: dict[str, bool] = {}
    coverage: Counter = Counter()
    n_reloc_positive = 0

    for record in predictions:
        user_id = str(record["user_id"])
        data = record.get("parsed") or {}

        per_category: dict[str, Counter] = {}
        for json_key, category in _BASELINE_KEY_TO_CATEGORY.items():
            raw = data.get(json_key)
            # age_bin / income_bin are arrays of 1-N adjacent bins; gender /
            # education are single strings. Normalize both to a list, lower-case,
            # and keep only values inside the DB-granularity set (each counted
            # once, mirroring the DB path's per-product tally).
            values = raw if isinstance(raw, list) else [raw]
            counter: Counter = Counter()
            for v in values:
                if not isinstance(v, str):
                    continue
                normalized = v.strip().lower()
                if normalized in allowed[category]:
                    counter[normalized] += 1
            if counter:
                per_category[category] = counter
                coverage[category] += 1
        user_demo[user_id] = per_category

        reloc = data.get("relocation")
        is_positive = isinstance(reloc, str) and reloc.strip().lower() == "positive"
        user_reloc[user_id] = is_positive
        if is_positive:
            n_reloc_positive += 1

    logger.info(
        f"🧮 parsed baseline predictions for {len(user_demo)} users; "
        f"per-category coverage={dict(coverage)}; relocation positive={n_reloc_positive}"
    )
    return user_demo, user_reloc


# ── unknown_fill: unify the evaluation cohort across methods ─────────────────

# The four point-prediction demographic categories that can be "unknown". Their
# DB-granularity value spaces are reused from the metric mappings above so a
# filled value is always a valid, scorable label. (relocation has no 'unknown'
# state — both methods assign every user a positive/negative, so it is never
# filled and stays unified at the full cohort.)
_FILL_CATEGORIES = ("signal:age-bin", "signal:gender", "signal:income-bin", "signal:education")


def _fill_value_space() -> dict[str, list[str]]:
    """Per-category allowed value set a missing prediction can be filled with."""
    return {
        "signal:age-bin": list(_BASELINE_AGE_BINS),
        "signal:gender": list(_GENDER_CLASSES),  # {female, male, other} — gender eval is unchanged
        "signal:income-bin": list(_INCOME_DB_RANK),
        "signal:education": list(_EDUCATION_DB_RANK),
    }


def fill_unknown_demographics(  # noqa: C901
    user_demo: dict[str, dict[str, Counter]],
    strategy: str = "random",
    seed: int = 3407,
) -> dict[str, dict[str, Counter]]:
    """Fill missing (unknown / no-signal) per-attribute predictions so every user
    carries a value for all four point-prediction categories, unifying the
    evaluation cohort across methods (the only remaining drop is the
    method-independent survey-validity filter).

    This trades the per-method coverage difference for injected noise: a method
    that abstains more gets more filled (≈chance) predictions, so its metrics are
    penalised accordingly. That is the intended "force-predict / abstention =
    guess" framing — distinct from an intersection-cohort comparison.

    strategy:
      - "drop"     : no fill (legacy behaviour; cohorts may differ across methods)
      - "random"   : uniform random pick from the category's DB-granularity values
      - "prior"    : sample from the empirical distribution of observed predictions
      - "majority" : the single most frequent observed value
    Deterministic given `seed`. gender is filled over {female, male, other}; the
    with_other / without_other gender evaluation is left unchanged.
    """
    if strategy == "drop":
        return user_demo
    if strategy not in ("random", "prior", "majority"):
        raise ValueError(f"unknown unknown_fill strategy: {strategy!r}")

    rng = np.random.default_rng(seed)
    spaces = _fill_value_space()

    # Empirical distribution of observed (non-filled) predictions, for prior/majority.
    observed: dict[str, Counter] = {cat: Counter() for cat in _FILL_CATEGORIES}
    for per_category in user_demo.values():
        for cat in _FILL_CATEGORIES:
            counter = per_category.get(cat)
            if counter:
                observed[cat][_mode_value(counter)] += 1

    prior_choices: dict[str, tuple[list[str], Any]] = {}
    majority_value: dict[str, str] = {}
    for cat in _FILL_CATEGORIES:
        values = spaces[cat]
        counts = np.array([observed[cat].get(v, 0) for v in values], dtype=float)
        if counts.sum() > 0:
            prior_choices[cat] = (values, counts / counts.sum())
            majority_value[cat] = values[int(counts.argmax())]
        else:
            prior_choices[cat] = (values, np.ones(len(values)) / len(values))
            majority_value[cat] = values[0]

    n_filled: Counter = Counter()
    for user_id in sorted(user_demo.keys()):  # sorted → deterministic fill order
        per_category = user_demo[user_id]
        for cat in _FILL_CATEGORIES:
            if per_category.get(cat):
                continue
            values = spaces[cat]
            if strategy == "random":
                value = values[int(rng.integers(len(values)))]
            elif strategy == "prior":
                vals, probs = prior_choices[cat]
                value = vals[int(rng.choice(len(vals), p=probs))]
            else:  # majority
                value = majority_value[cat]
            per_category[cat] = Counter({value: 1})
            n_filled[cat] += 1

    logger.info(f"🎲 unknown_fill[{strategy}] filled missing predictions: {dict(n_filled)} (seed={seed})")
    return user_demo


# ── persistence ──────────────────────────────────────────────────────────────


def save_results(
    metrics: dict[str, Any],
    out_dir: str,
    timestamp: str,
    mode: str | None = None,
) -> str:
    """Persist the full metrics/collection payload as output-<timestamp>.json.

    When `mode` is given (e.g. 'db' or 'baseline') it is appended to the filename
    so the two evaluation modes don't clobber each other in the same invocation.
    The large per-user `survey_ages`/`db_ages` arrays under `age` are kept so the
    scatter can be re-plotted without re-running the aggregation.
    """
    os.makedirs(out_dir, exist_ok=True)
    suffix = f"-{mode}" if mode else ""
    save_path = str(Path(f"{out_dir}/output-{timestamp}{suffix}.json").resolve())
    with open(save_path, "w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info(f"💾 [{mode or 'default'}] saved demographic-attribute metrics to {save_path}")
    return save_path
