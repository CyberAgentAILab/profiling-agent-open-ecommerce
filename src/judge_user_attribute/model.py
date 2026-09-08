import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm
from vllm import SamplingParams

from base_agent.model import BaseAgent

# Order is fixed so prompt and evaluation stay aligned.
ATTRIBUTE_KEYS: list[str] = [
    "cigarettes",
    "marijuana",
    "alcohol",
    "diabetes",
    "wheelchair",
    "had_child",
    "became_pregnant",
    "moved",
    "lost_job",
    "divorce",
]

# Map our prompt-output keys onto the survey CSV columns / class labels.
BINARY_TARGETS: dict[str, str] = {
    "cigarettes": "Q-substance-use-cigarettes",
    "marijuana": "Q-substance-use-marijuana",
    "alcohol": "Q-substance-use-alcohol",
    "diabetes": "Q-personal-diabetes",
    "wheelchair": "Q-personal-wheelchair",
}

LIFE_CHANGE_LABELS: dict[str, str] = {
    "had_child": "Had a child",
    "became_pregnant": "Became pregnant",
    "moved": "Moved place of residence",
    "lost_job": "Lost a job",
    "divorce": "Divorce",
}


def _binary_yes(value: Any) -> int | None:
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    if v == "yes" or "stopped" in v:
        return 1
    if v == "no":
        return 0
    return None


def _pred_to_int(value: Any) -> int:
    """Map a per-attribute prediction (bool from the LLM JSON) to {0, 1}.
    Tolerates the legacy 'yes'/'no' string form so older output files can still
    be re-scored without re-running inference.
    """
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, str):
        return 1 if value.strip().lower() in ("yes", "true", "1") else 0
    return 0


def build_baseline_user_context(
    user_title_freq: dict[str, int],
    max_titles: int | None = None,
) -> str:
    """Render a single user's prompt-input block for the baseline mode.

    The block is just a ranked frequency list of the user's Amazon Titles —
    no tag DB, no aggregation across users. Long histories are truncated at
    `max_titles` unique entries (top by count) to fit the model context.
    """
    items = list(user_title_freq.items())
    n_unique_total = len(items)
    total_count = sum(c for _, c in items)

    if max_titles is not None and n_unique_total > max_titles:
        items = items[:max_titles]
        truncation_note = (
            f" (showing top {max_titles} unique titles by count; "
            f"{n_unique_total - max_titles} less-frequent titles omitted)"
        )
    else:
        truncation_note = ""

    if items:
        lines = [f"- {title} (x{count})" for title, count in items]
        body = "\n".join(lines)
    else:
        body = "(no purchases recorded for this user)"

    return (
        f"This user's Amazon purchase history "
        f"({n_unique_total} unique titles, {total_count} total purchases){truncation_note}:\n"
        f"{body}"
    )


def build_user_context(all_tags: list[str], user_tag_freq: dict[str, int]) -> str:
    """Render a single user's prompt-input block.

    The block lists EVERY tag in the DB universe alongside its per-user count.
    Tags the user never matched appear with count 0 — that absence is itself a
    signal we want the LLM to consider. Tags are ordered by count desc, ties
    broken alphabetically, so high-evidence tags come first.
    """
    n_total = len(all_tags)
    n_nonzero = sum(1 for v in user_tag_freq.values() if v > 0)
    total_count = sum(user_tag_freq.values())

    freq_lines = [f"- {tag}: {user_tag_freq.get(tag, 0)}" for tag in all_tags]
    # `all_tags` is alphabetical from the DB loader; re-sort here by count desc
    # then by tag name so the user-specific salient tags surface first while
    # zero-count tags still appear (after the nonzero ones).
    freq_lines = sorted(
        freq_lines,
        key=lambda line: (-int(line.rsplit(": ", 1)[-1]), line),
    )
    freq_block = "\n".join(freq_lines)

    return (
        f"This user's tag frequency profile (covers all {n_total} tags in the DB; "
        f"{n_nonzero} tags have count > 0; total tagged purchases = {total_count}):\n"
        f"{freq_block}"
    )


_BRANCH_SECTION_LABELS: dict[str, str] = {
    "demographic": "Demographic tags",
    "psycho_behavioral": "Psycho-behavioral tags",
    "life_event": "Life event tags",
}

# Fixed product-level signal categories, grouped by domain to mirror the tag
# sections. Order here is the display order; values within a category are sorted
# by count desc at render time.
_SIGNAL_DOMAINS: list[tuple[str, list[str]]] = [
    (
        "Demographic signals",
        [
            "signal:age-bin",
            "signal:gender",
            "signal:income-bin",
            "signal:education",
            "signal:occupation-class",
            "signal:ethnicity",
            "signal:location-type",
        ],
    ),
    (
        "Psycho-behavioral signals",
        [
            "signal:lifestyle",
            "signal:personality",
            "signal:purchase-occasion",
            "signal:benefit-sought",
        ],
    ),
    (
        "Life-event signals",
        [
            "signal:recent-marriage",
            "signal:recent-childbirth",
            "signal:recent-relocation",
            "signal:recent-retirement",
            "signal:recent-school-enrollment",
            "signal:recent-empty-nest",
            "signal:recent-job-change",
            "signal:recent-bereavement",
        ],
    ),
]


def build_signal_section(user_signals: dict[str, Counter]) -> str:
    """Render one user's fixed-attribute signal block.

    `user_signals` is {signal-category -> Counter(value -> count)} for one user.
    Categories with no matched values are skipped; within a category, values are
    listed "value (count)" by count desc (ties alphabetical). Returns "" when the
    user has no signal predictions at all, so the caller can omit the section.
    """
    domain_blocks: list[str] = []
    for label, categories in _SIGNAL_DOMAINS:
        lines: list[str] = []
        for category in categories:
            counter = user_signals.get(category)
            if not counter:
                continue
            ordered = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
            dist = ", ".join(f"{value} ({count})" for value, count in ordered)
            lines.append(f"- {category}: {dist}")
        if lines:
            domain_blocks.append(f"### {label}\n" + "\n".join(lines))

    if not domain_blocks:
        return ""

    header = (
        "## Fixed product-level attribute signals (aggregated over this user's purchases)\n"
        "These are independent product-level predictions of a fixed attribute taxonomy, "
        "aggregated across the products this user bought (each predicted value weighted by "
        'purchase count; "unknown" predictions omitted). Treat them as soft, noisy priors — '
        "NOT ground truth — and weigh them together with the tag evidence above. The "
        "life-event signals (positive/negative) are related to but distinct from the target "
        "attributes, so use them as hints rather than answers."
    )
    return header + "\n\n" + "\n\n".join(domain_blocks)


def build_user_context_multi(
    branch_tag_data: list[tuple[str, list[str], dict[str, dict[str, int]], dict[str, dict[str, list[str]]]]],
    user_id: str,
    user_to_signals: dict[str, dict[str, Counter]] | None = None,
) -> str:
    """Render a single user's prompt-input block for multi-branch db mode.

    Each entry in branch_tag_data is
        (branch_name, all_tags, user_to_tags, user_to_title_tags).
    Produces one labeled section per branch (demographic / psycho-behavioral /
    life-event) so the LLM reasons over all three domains in a single call.

    The product set is already restricted upstream (aggregate_tags_per_user with
    max_products = baseline_max_titles_per_user) to the same top-N titles the
    baseline sees, so this function renders everything it is given. Each section
    shows two views, limited to tags the user matched (count > 0):
      1. Tag frequency: "- tag: count", sorted by count desc.
      2. Products and their tags: each selected title mapped to its tag(s).
    """
    sections: list[str] = []
    for branch_name, all_tags, user_to_tags, user_to_title_tags in branch_tag_data:
        label = _BRANCH_SECTION_LABELS.get(branch_name, branch_name)
        user_tag_freq = user_to_tags.get(user_id, {})

        # keep only tags the user matched (count > 0)
        nonzero = {tag: cnt for tag, cnt in user_tag_freq.items() if cnt > 0}
        nonzero = dict(sorted(nonzero.items(), key=lambda kv: (-kv[1], kv[0])))
        total_count = sum(nonzero.values())

        freq_block = "\n".join(f"- {tag}: {cnt}" for tag, cnt in nonzero.items()) or "(no tags matched)"

        title_tags = user_to_title_tags.get(user_id, {})
        product_lines = (
            "\n".join(f'- "{title}" -> {", ".join(tags)}' for title, tags in title_tags.items())
            or "(no tagged products)"
        )

        sections.append(
            f"## {label} "
            f"({len(nonzero)} of {len(all_tags)} DB tags matched; total tagged purchases = {total_count}):\n"
            f"### Tag frequency (count > 0):\n"
            f"{freq_block}\n\n"
            f"### Tagged products and their {label.lower()}:\n"
            f"{product_lines}"
        )

    block = "\n\n".join(sections)
    if user_to_signals is not None:
        signal_section = build_signal_section(user_to_signals.get(user_id, {}))
        if signal_section:
            block = f"{block}\n\n{signal_section}"
    return block


def filter_users_with_any_yes_ground_truth(
    user_ids: list[str],
    survey_df: pd.DataFrame,
    mode: str | None = None,
) -> tuple[list[str], dict[str, int]]:
    """Keep only users whose survey ground truth contains at least one 'yes'
    across the binary targets and life-change labels.

    Users absent from the survey, or whose every target resolves to 'no' /
    'unknown', carry no positive signal and are dropped from the evaluation
    cohort. Returns the retained user_ids and a cohort-counts dict
    (valid_user / invalid_user / n_missing_from_survey) so downstream code can
    persist the cohort composition alongside the metrics.
    """
    survey_indexed = survey_df.set_index("Survey ResponseID")
    kept: list[str] = []
    n_invalid = 0
    n_missing_from_survey = 0

    for user_id in user_ids:
        if user_id not in survey_indexed.index:
            n_missing_from_survey += 1
            n_invalid += 1
            continue
        row = survey_indexed.loc[user_id]
        has_yes = False
        for survey_col in BINARY_TARGETS.values():
            if _binary_yes(row.get(survey_col)) == 1:
                has_yes = True
                break
        if not has_yes:
            life_change_value = row.get("Q-life-changes")
            for class_label in LIFE_CHANGE_LABELS.values():
                if _life_change_gt_bool(life_change_value, class_label):
                    has_yes = True
                    break
        if has_yes:
            kept.append(user_id)
        else:
            n_invalid += 1

    counts = {
        "valid_user": len(kept),
        "invalid_user": n_invalid,
        "n_input_users": len(user_ids),
        "n_missing_from_survey": n_missing_from_survey,
    }
    tag = f"[{mode}] " if mode else ""
    logger.info(
        f"🧹 {tag}all-no GT filter: valid_user={counts['valid_user']} invalid_user={counts['invalid_user']} "
        f"(of which {n_missing_from_survey} absent from the survey) / total_input={counts['n_input_users']}"
    )
    return kept, counts


class JudgeUserAttributeAgent(BaseAgent):
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
        trust_remote_code: bool = True,
        seed: int = 3407,
        gpu_memory_utilization: float = 0.9,
        enforce_eager: bool = True,
        enable_prefix_caching: bool = False,
        enable_cache: bool = True,
        json_max_tokens: int = 2048,
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
            trust_remote_code=trust_remote_code,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            enable_prefix_caching=enable_prefix_caching,
            enable_cache=enable_cache,
            enable_thinking=enable_thinking,
        )
        # Cap on generated tokens for a single user's JSON output. The model's
        # max_model_len (which may diverge from max_new_tokens) still bounds
        # input+output, but per-call we only need enough room for the JSON.
        self._json_max_tokens = json_max_tokens
        return

    def _render_prompt(self, user_context: str) -> str:
        """Render the per-user prompt (single shot for all 10 attributes).

        We do template substitution ourselves so the chat-template wrap and
        the `enable_thinking` flag are explicit at the call site — the LLM
        must emit the JSON directly, not via a <think> preamble unless
        thinking is explicitly enabled via config.
        """
        body = self.prompt_template.replace("{query}", user_context)
        chat = [{"role": "user", "content": body}]
        enable_thinking_flag = self.enable_thinking if self.enable_thinking is not None else False
        return self.tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking_flag,
        )

    @staticmethod
    def _extract_json_object(raw_text: str) -> dict | None:
        """Best-effort recovery of the JSON object the LLM was instructed to
        emit. Tries: (1) raw parse, (2) parse after stripping ``` fences,
        (3) parse the substring between the first '{' and the matching '}'.
        Returns None when no valid object is found.
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

    @staticmethod
    def _coerce_attribute_entry(entry: Any) -> tuple[bool, float]:
        """Coerce a single `attributes[key]` value into (decision, confidence).

        The prompt emits each attribute value as a single float in [0, 1].
        Clamp to [0, 1], default 0.5 on miss, and derive decision from
        confidence (>= 0.5 → True, else False).

        Non-numeric values (including numeric strings like "0.9") DELIBERATELY
        fall back to the neutral 0.5: the paper runs scored with this exact
        behaviour, so parsing strings here would change reproduced results.
        """
        if isinstance(entry, (int, float)):
            try:
                confidence = float(entry)
            except (TypeError, ValueError):
                confidence = 0.5
        else:
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))
        return confidence >= 0.5, confidence

    def _parse_response(
        self,
        raw_text: str,
    ) -> tuple[dict[str, bool], dict[str, float], str, dict[str, Any]]:
        """Parse the LLM's free-form JSON answer for all 10 attributes.

        Returns (predicted, predicted_proba, reasoning, attributes_raw) where:
          - predicted[attr]       : bool (True = positive match)
          - predicted_proba[attr] : float in [0, 1] (= confidence = P(positive))
          - reasoning             : the model's free-text reasoning, or "" on parse miss
          - attributes_raw        : the raw per-attribute dict (kept for audit)

        A parse miss for the whole response collapses every attribute to the
        neutral default (False, 0.5) so downstream evaluation never crashes —
        the per-record raw_content preserves the original output for debugging.
        """
        predicted: dict[str, bool] = dict.fromkeys(ATTRIBUTE_KEYS, False)
        predicted_proba: dict[str, float] = dict.fromkeys(ATTRIBUTE_KEYS, 0.5)

        data = self._extract_json_object(raw_text)
        if data is None:
            return predicted, predicted_proba, "", {}

        reasoning_val = data.get("reasoning")
        reasoning = reasoning_val if isinstance(reasoning_val, str) else ""

        attrs = data.get("attributes")
        if not isinstance(attrs, dict):
            return predicted, predicted_proba, reasoning, {}

        for key in ATTRIBUTE_KEYS:
            decision, confidence = self._coerce_attribute_entry(attrs.get(key))
            predicted[key] = decision
            predicted_proba[key] = confidence

        return predicted, predicted_proba, reasoning, attrs

    def judge_users(
        self,
        user_ids: list[str],
        contexts: list[str],
        batch_size: int,
    ) -> list[dict[str, Any]]:
        """Run vLLM inference one batch at a time. One prompt per user — the
        prompt asks for all 10 attributes in a single JSON response.

        Returns one record per user with:
            predicted:       {attr: bool}  # True = positive match
            predicted_proba: {attr: float in [0, 1]}  # = confidence, P(positive)
            reasoning:       free-text rationale produced by the model
            raw_content:     the full string the model emitted (for debugging)
        """
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=self._json_max_tokens,
        )

        results: list[dict[str, Any]] = []
        n_total = len(contexts)
        num_total_batch = max(1, (n_total + batch_size - 1) // batch_size)
        pbar = tqdm(range(0, n_total, batch_size), total=num_total_batch, desc="judge_user_attribute")

        n_parse_fail = 0
        for step in pbar:
            batch_user_ids = user_ids[step : step + batch_size]
            batch_contexts = contexts[step : step + batch_size]
            n_users_in_batch = len(batch_contexts)

            prompts = [self._render_prompt(ctx) for ctx in batch_contexts]

            t0 = time.perf_counter()
            vllm_outputs = self.model.generate(prompts=prompts, sampling_params=sampling_params)
            duration = time.perf_counter() - t0
            duration_per_user = duration / max(1, n_users_in_batch)

            for user_id, ctx, out in zip(batch_user_ids, batch_contexts, vllm_outputs, strict=False):
                generated = out.outputs[0]
                raw_content = self.tokenizer.decode(list(generated.token_ids), skip_special_tokens=True)

                predicted, predicted_proba, reasoning, attrs_raw = self._parse_response(raw_content)
                if not attrs_raw:
                    n_parse_fail += 1

                results.append(
                    {
                        "user_id": user_id,
                        "context": ctx,
                        "predicted": predicted,
                        "predicted_proba": predicted_proba,
                        "reasoning": reasoning,
                        "attributes_raw": attrs_raw,
                        "raw_content": raw_content,
                        "thinking": None,
                        "input_tokens": len(out.prompt_token_ids),
                        "output_tokens": len(generated.token_ids),
                        "duration_per_sample": duration_per_user,
                    }
                )

        if n_parse_fail:
            logger.warning(
                f"⚠️ JSON parse failed for {n_parse_fail}/{len(results)} user responses — "
                f"those entries fell back to neutral defaults (predicted=False, proba=0.5)"
            )
        return results


def _binary_gt_bool(value: Any) -> bool | None:
    """Render a binary-target survey answer as bool | None. None means the
    survey cell was missing / refused (treated as "unknown" downstream).
    """
    gt = _binary_yes(value)
    if gt == 1:
        return True
    if gt == 0:
        return False
    return None


def _life_change_gt_bool(value: Any, class_label: str) -> bool:
    # Match evaluate_predictions semantics: a missing Q-life-changes cell
    # is treated as "the user had none of the listed life changes" → False.
    text = value if isinstance(value, str) else ""
    return class_label.lower() in text.lower()


def attach_ground_truth(
    predictions: list[dict[str, Any]],
    survey_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Annotate each prediction record with a `ground_truth` field shaped
    identically to `predicted` ({attribute_key: bool | None}). `None` denotes
    "unknown" (survey cell missing or refused); users not present in the
    survey CSV get all-None ground truth.

    Also attaches a per-attribute `correctness` map ('correct' / 'wrong' /
    'unknown') and a sample-level summary so each saved record makes the
    sample's prediction outcome inspectable on its own.
    """
    survey_indexed = survey_df.set_index("Survey ResponseID")

    for record in predictions:
        user_id = record.get("user_id")
        gt: dict[str, bool | None] = dict.fromkeys(ATTRIBUTE_KEYS, None)

        if user_id in survey_indexed.index:
            row = survey_indexed.loc[user_id]
            for key, survey_col in BINARY_TARGETS.items():
                gt[key] = _binary_gt_bool(row.get(survey_col))
            life_change_value = row.get("Q-life-changes")
            if isinstance(life_change_value, str):
                for key, class_label in LIFE_CHANGE_LABELS.items():
                    gt[key] = _life_change_gt_bool(life_change_value, class_label)
            # else: gt[key] remains None — consistent with binary target NaN handling

        record["ground_truth"] = gt

        predicted = record.get("predicted") or {}
        correctness: dict[str, str] = {}
        n_correct = 0
        n_scored = 0
        for key in ATTRIBUTE_KEYS:
            gt_v = gt.get(key)
            pred_v = bool(predicted.get(key, False))
            if gt_v is None:
                correctness[key] = "unknown"
                continue
            n_scored += 1
            if pred_v == gt_v:
                correctness[key] = "correct"
                n_correct += 1
            else:
                correctness[key] = "wrong"
        record["correctness"] = correctness
        record["correctness_summary"] = {
            "n_scored": n_scored,
            "n_correct": n_correct,
            "accuracy": (n_correct / n_scored) if n_scored else None,
        }

    n_with_gt = sum(1 for r in predictions if any(v is not None for v in r["ground_truth"].values()))
    logger.info(f"📎 attached ground_truth + correctness for {n_with_gt}/{len(predictions)} users (rest are 'unknown')")
    return predictions


def evaluate_predictions(
    predictions: list[dict[str, Any]],
    survey_df: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """Compare bool predictions against survey ground truth and report
    per-attribute accuracy / precision / recall / F1 / AUC, plus an overall
    macro-F1 and macro-AUC. AUC uses `predicted_proba[attr]` directly as
    P(positive); the proba is the LLM-emitted confidence (P(positive) by
    prompt contract).
    """
    survey_df = survey_df.set_index("Survey ResponseID")
    user_to_pred = {p["user_id"]: p["predicted"] for p in predictions}
    user_to_proba = {p["user_id"]: p.get("predicted_proba") or {} for p in predictions}

    metrics: dict[str, dict[str, Any]] = {}

    for key, survey_col in BINARY_TARGETS.items():
        y_true: list[int] = []
        y_pred: list[int] = []
        y_score: list[float] = []
        for user_id, pred in user_to_pred.items():
            if user_id not in survey_df.index:
                continue
            gt = _binary_yes(survey_df.loc[user_id, survey_col])
            if gt is None:
                continue
            y_true.append(gt)
            y_pred.append(_pred_to_int(pred.get(key)))
            y_score.append(float(user_to_proba.get(user_id, {}).get(key, 0.0)))
        metrics[key] = _score_block(y_true, y_pred, y_score=y_score)

    for key, class_label in LIFE_CHANGE_LABELS.items():
        y_true = []
        y_pred = []
        y_score = []
        for user_id, pred in user_to_pred.items():
            if user_id not in survey_df.index:
                continue
            gt_str = survey_df.loc[user_id, "Q-life-changes"]
            if not isinstance(gt_str, str):
                continue
            y_true.append(int(class_label.lower() in gt_str.lower()))
            y_pred.append(_pred_to_int(pred.get(key)))
            y_score.append(float(user_to_proba.get(user_id, {}).get(key, 0.0)))
        metrics[key] = _score_block(y_true, y_pred, y_score=y_score)

    def _macro(field: str) -> float | None:
        values = [m[field] for m in metrics.values() if isinstance(m.get(field), float)]
        return float(sum(values) / len(values)) if values else None

    macro_f1 = _macro("f1") or 0.0
    macro_auc = _macro("auc")
    macro_pr_auc = _macro("pr_auc")
    macro_f1_optimal = _macro("f1_optimal")
    macro_f1_optimal_lift = _macro("f1_optimal_lift_over_baseline")
    metrics["__overall__"] = {
        "macro_f1": float(macro_f1),
        "macro_auc": macro_auc,
        "macro_pr_auc": macro_pr_auc,
        "macro_f1_optimal": macro_f1_optimal,
        "macro_f1_optimal_lift": macro_f1_optimal_lift,
        "n_users": len(user_to_pred),
    }

    _log_auc_summary(metrics)
    return metrics


def _log_auc_summary(metrics: dict[str, dict[str, Any]]) -> None:
    """Log only the AUC per category and the macro AUC; every other metric lives in the saved results JSON."""

    def _fmt(value: Any) -> str:
        return f"{value:.4f}" if isinstance(value, float) else "n/a"

    overall = metrics.get("__overall__", {})
    per_category = " | ".join(f"{key}={_fmt(m.get('auc'))}" for key, m in metrics.items() if key != "__overall__")
    logger.info(f"📊 AUC per category ({overall.get('n_users', 0)} users): {per_category}")
    logger.info(f"📊 macro_auc: {_fmt(overall.get('macro_auc'))}")


def _score_block(
    y_true: list[int],
    y_pred: list[int],
    y_score: list[float] | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Per-attribute metric block. Threshold-independent metrics (auc, pr_auc,
    f1_optimal) are included only when y_score is supplied and y_true contains
    both classes; otherwise they are omitted.
    """
    if not y_true:
        return {"n": 0}
    n = len(y_true)
    n_pos = int(sum(y_true))
    prevalence = n_pos / n if n else 0.0
    baseline_f1 = 2 * prevalence / (1 + prevalence) if prevalence > 0 else 0.0
    block: dict[str, Any] = {
        "n": n,
        "n_pos_true": n_pos,
        "n_pos_pred": int(sum(y_pred)),
        "prevalence": float(prevalence),
        "threshold_current": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "baseline_f1_all_positive": float(baseline_f1),
    }
    if y_score is not None and len(set(y_true)) > 1:
        y_true_arr = np.asarray(y_true)
        y_score_arr = np.asarray(y_score, dtype=float)
        block["auc"] = float(roc_auc_score(y_true_arr, y_score_arr))
        pr_auc = float(average_precision_score(y_true_arr, y_score_arr))
        block["pr_auc"] = pr_auc
        block["pr_auc_lift"] = float(pr_auc / prevalence) if prevalence > 0 else None

        precision_arr, recall_arr, thresholds_arr = precision_recall_curve(y_true_arr, y_score_arr)
        # precision_recall_curve appends a (precision=1, recall=0) sentinel
        # without a matching threshold; drop it to keep arrays aligned.
        p = precision_arr[:-1]
        r = recall_arr[:-1]
        denom = p + r
        f1_arr = np.where(denom > 0, 2 * p * r / (denom + 1e-12), 0.0)
        if f1_arr.size > 0:
            best_idx = int(np.argmax(f1_arr))
            block["f1_optimal"] = float(f1_arr[best_idx])
            block["threshold_optimal"] = float(thresholds_arr[best_idx])
            block["precision_optimal"] = float(p[best_idx])
            block["recall_optimal"] = float(r[best_idx])
            block["f1_optimal_lift_over_baseline"] = (
                float(block["f1_optimal"] / baseline_f1) if baseline_f1 > 0 else None
            )
        else:
            block["f1_optimal"] = None
            block["threshold_optimal"] = None
            block["precision_optimal"] = None
            block["recall_optimal"] = None
            block["f1_optimal_lift_over_baseline"] = None

        block["score_n_unique"] = int(np.unique(y_score_arr).size)
        block["score_min"] = float(y_score_arr.min())
        block["score_max"] = float(y_score_arr.max())
        block["score_mean"] = float(y_score_arr.mean())
        block["score_median"] = float(np.median(y_score_arr))
        score_counter = Counter(np.round(y_score_arr, 2).tolist())
        block["score_top5_modes"] = [(float(v), int(c)) for v, c in score_counter.most_common(5)]
    return block


def save_results(
    predictions: list[dict[str, Any]],
    metrics: dict[str, dict[str, Any]],
    out_dir: str,
    timestamp: str,
    mode: str | None = None,
) -> str:
    """Persist predictions + metrics. When `mode` is provided (e.g. 'db' or
    'baseline') it is appended to the filename so multiple runs in the same
    invocation don't clobber each other.
    """
    os.makedirs(out_dir, exist_ok=True)
    suffix = f"-{mode}" if mode else ""
    save_path = str(Path(f"{out_dir}/output-{timestamp}{suffix}.json").resolve())
    payload = {"mode": mode, "predictions": predictions, "metrics": metrics}
    with open(save_path, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info(f"💾 [{mode or 'default'}] saved {len(predictions)} predictions + metrics to {save_path}")
    return save_path
