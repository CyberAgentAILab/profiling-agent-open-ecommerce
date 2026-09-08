import time
from typing import Any

from loguru import logger
from tqdm import tqdm

from base_agent.config import save_as_json, validate_json
from base_agent.model import BaseAgent


class TransactionSubAgent(BaseAgent):
    def __init__(
        self,
        config: dict,
        model_name: str,
        torch_dtype: str,
        path_prompt_template: str,
        out_dir: str,
        timestamp: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        enable_prefix_caching: bool,
        use_vllm: bool = False,
        tensor_parallel_size: int = 1,
        trust_remote_code: bool = True,
        seed: int = 3407,
        gpu_memory_utilization: float = 0.9,
        enforce_eager: bool = True,
        enable_cache: bool = True,
        enable_thinking: bool | None = None,
        max_model_len: int | None = None,
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
            enable_cache=enable_cache,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            enable_prefix_caching=enable_prefix_caching,
            enable_thinking=enable_thinking,
        )

        self.json_schema_transaction = config["json_schema_transaction"]
        logger.info(f"📝 prompt_path: {path_prompt_template}")
        return

    def _build_context(self, search_result: dict[str, Any]) -> str:
        query = search_result["resolved_query"]

        search_data = search_result.get("search")
        if search_data is None:
            retrieval_context = ""
        else:
            search_results_list = search_data.get("search_result", [])
            retrieval_contexts = [
                item.get("retrieval_context", "") for item in search_results_list if isinstance(item, dict)
            ]
            retrieval_context = "\n".join(retrieval_contexts)

        return f"Product title\n\n{query}\n\nSearch results\n\n{retrieval_context}"

    def recovery_transaction_attribute(
        self,
        batch_search_result: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        batch_context = list(map(self._build_context, batch_search_result))

        # Use the actual batch length so the final (smaller) batch is not misreported.
        time_scale = 1 / max(len(batch_context), 1)
        begin = time.perf_counter()
        batch_transaction_attribute = self.generate(prompt=batch_context)
        duration_per_sample = time_scale * (time.perf_counter() - begin)

        results = []
        for transaction_attribute, search_result in zip(batch_transaction_attribute, batch_search_result, strict=False):
            success_recovery = False
            attribute = validate_json(transaction_attribute["content"], required_keys=self.json_schema_transaction)
            run_recovery = True
            name = attribute.get("canonical_name")
            if isinstance(name, str) and name.strip():
                success_recovery = True

            search_result.update(
                {
                    "transaction": {
                        "prediction": attribute,
                        "run_recovery": run_recovery,
                        "success_recovery": success_recovery,
                        "duration_recovery": float(f"{duration_per_sample:.04f}"),
                        "input_tokens": transaction_attribute["input_tokens"],
                        "output_tokens": transaction_attribute["output_tokens"],
                        "thinking": transaction_attribute.get("thinking", None),
                    }
                }
            )
            results.append(search_result)
        return results

    def predict_transaction_attribute(
        self,
        search_results: list[dict[str, Any]],
        batch_size: int,
        debug_num_sample: int | None,
        job_id: str | None = None,
    ) -> str:
        if debug_num_sample is not None:
            search_results = search_results[:debug_num_sample]

        transaction_attributes: list[dict[str, Any]] = []
        # Initial save keeps save_path defined (and the output file present)
        # even when search_results is empty; each batch below overwrites it as a checkpoint.
        save_path = save_as_json(
            responses=transaction_attributes,
            out_dir=self.out_dir,
            timestamp=self.timestamp,
            job_id=job_id,
        )

        num_total_batch = (len(search_results) + batch_size - 1) // batch_size
        pbar = tqdm(range(0, len(search_results), batch_size), total=num_total_batch)
        for step in pbar:
            batch_search_results = search_results[step : step + batch_size]
            batch_attribute = self.recovery_transaction_attribute(
                batch_search_result=batch_search_results,
            )
            transaction_attributes.extend(batch_attribute)

            save_path = save_as_json(
                responses=transaction_attributes,
                out_dir=self.out_dir,
                timestamp=self.timestamp,
                job_id=job_id,
            )

        logger.info(f"💾 transaction predictions: {len(transaction_attributes)} records -> {save_path}")
        return save_path
