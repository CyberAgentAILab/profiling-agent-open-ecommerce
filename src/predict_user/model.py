import json
import time
from typing import Any

from loguru import logger
from tqdm import tqdm

from base_agent.config import save_as_json, validate_json
from base_agent.model import BaseAgent

ECOMMERCE_ATTRIBUTE_KEYS = (
    "user_attribute_demographic",
    "user_attribute_psycho_behavioral",
    "user_attribute_life_event",
)


class UserSubAgent(BaseAgent):
    def __init__(
        self,
        config: dict,
        model_name: str,
        torch_dtype: str,
        path_prompt_template: str,
        out_dir: str,
        timestamp: str,
        max_new_tokens: int,
        max_model_len: int | None = None,
        use_vllm: bool = False,
        tensor_parallel_size: int = 1,
        trust_remote_code: bool = True,
        seed: int = 3407,
        gpu_memory_utilization: float = 0.9,
        enforce_eager: bool = True,
        enable_prefix_caching: bool = False,
        enable_cache: bool = True,
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
        self.json_schema_parent = config["json_schema_parent"]
        self.json_schema_child = config["json_schema_child"]

        return

    def _build_context(self, transaction: dict[str, Any]) -> str:
        items = transaction.get("items") or []
        return json.dumps({"items": items}, ensure_ascii=False, indent=2)

    @staticmethod
    def _invalid_groups() -> dict[str, dict[str, Any]]:
        """One fresh {"is_json": False} dict per group — never share instances."""
        return {key: {"is_json": False} for key in ECOMMERCE_ATTRIBUTE_KEYS}

    def _parse_combined_attributes(self, content: str) -> dict[str, dict[str, Any]]:
        """Parse the combined LLM JSON into the 3 per-group attribute dicts.

        Each returned value is the same shape that `validate_json` would have
        produced for a stand-alone single-group prompt — i.e. a dict containing
        `consumer_attributes` plus an `is_json` flag. A parse failure at any
        level falls back to `{"is_json": False}` for the affected key(s).
        """
        try:
            parsed = json.loads(content)
        except (json.JSONDecodeError, Exception):
            logger.error(f"failed parse example:\n{content}")
            return self._invalid_groups()

        if not isinstance(parsed, dict):
            logger.error(f"combined output is not an object:\n{content}")
            return self._invalid_groups()

        result: dict[str, dict[str, Any]] = {}
        for key in ECOMMERCE_ATTRIBUTE_KEYS:
            sub = parsed.get(key)
            if not isinstance(sub, dict):
                result[key] = {"is_json": False}
                continue
            result[key] = validate_json(
                json.dumps(sub),
                required_keys=self.json_schema_child,
                required_attribute_name=self.json_schema_parent,
            )
        return result

    def predict_ecommerce_attribute(
        self,
        batch_transaction: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Single-prompt prediction that returns all 3 attribute groups at once.

        The combined prompt asks the model to emit one JSON containing
        `user_attribute_demographic`, `user_attribute_psycho_behavioral`, and
        `user_attribute_life_event`, each shaped like the legacy per-group
        outputs. This keeps the downstream consumer schema unchanged while
        guaranteeing that the LLM sees full context for all three groups in a
        single inference pass.
        """
        batch_context = list(map(self._build_context, batch_transaction))

        # Use the actual batch length so the final (smaller) batch is not misreported.
        time_scale = 1 / max(len(batch_context), 1)
        begin = time.perf_counter()
        batch_user_attribute = self.generate(prompt=batch_context, use_extra_prompt=False)
        duration_per_sample = time_scale * (time.perf_counter() - begin)

        results: list[dict[str, Any]] = []
        for user_attribute, transaction in zip(batch_user_attribute, batch_transaction, strict=False):
            attributes = self._parse_combined_attributes(user_attribute["content"])
            success_predict = any(len(attr.get(self.json_schema_parent, [])) > 0 for attr in attributes.values())

            user_dict: dict[str, Any] = {
                "run_predict": True,
                "duration_predict": duration_per_sample,
            }
            for key in ECOMMERCE_ATTRIBUTE_KEYS:
                user_dict[key] = attributes[key]
            user_dict["input_tokens"] = user_attribute["input_tokens"]
            user_dict["output_tokens"] = user_attribute["output_tokens"]
            user_dict["success_predict"] = success_predict

            transaction.update({"user": user_dict})
            results.append(transaction)
        return results

    def predict_user_attributes(
        self,
        transactions: list[dict[str, Any]],
        batch_size: int,
        debug_num_sample: int | None,
        job_id: str | None = None,
    ) -> str:
        if debug_num_sample is not None:
            transactions = transactions[:debug_num_sample]

        if not transactions:
            logger.warning(
                "⚠️ no transactions to predict (input length 0). "
                "Check upstream out_dir matches predict_user.validation_path / query_path."
            )

        user_attributes: list[dict[str, Any]] = []
        # Initial save keeps save_path defined (and the output file present)
        # even when transactions is empty; each batch below overwrites it as a checkpoint.
        save_path = save_as_json(
            responses=user_attributes,
            out_dir=self.out_dir,
            timestamp=self.timestamp,
            job_id=job_id,
        )

        num_total_batch = (len(transactions) + batch_size - 1) // batch_size
        pbar = tqdm(range(0, len(transactions), batch_size), total=num_total_batch)
        for step in pbar:
            batch_transaction = transactions[step : step + batch_size]

            batch_attribute = self.predict_ecommerce_attribute(
                batch_transaction=batch_transaction,
            )
            user_attributes.extend(batch_attribute)

            save_path = save_as_json(
                responses=user_attributes,
                out_dir=self.out_dir,
                timestamp=self.timestamp,
                job_id=job_id,
            )

        logger.info(f"💾 user-attribute predictions: {len(user_attributes)} records -> {save_path}")
        return save_path
