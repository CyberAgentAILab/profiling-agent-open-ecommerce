import time
from typing import Any

from loguru import logger
from tqdm import tqdm

from base_agent.config import save_as_json, validate_json
from base_agent.model import BaseAgent


class ScanSubAgent(BaseAgent):
    def __init__(
        self,
        model_name: str,
        torch_dtype: str,
        path_prompt_template: str,
        top_p: float,
        out_dir: str,
        timestamp: str,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        use_vllm: bool,
        tensor_parallel_size: int,
        trust_remote_code: bool,
        seed: int,
        gpu_memory_utilization: float,
        enforce_eager: bool,
        yes_tokens: list,
        json_schema_scan: list,
        enable_cache: bool = True,
        enable_thinking: bool | None = None,
        max_model_len: int | None = None,
    ) -> None:
        super().__init__(
            model_name=model_name,
            torch_dtype=torch_dtype,
            path_prompt_template=path_prompt_template,
            out_dir=out_dir,
            timestamp=timestamp,
            max_new_tokens=max_new_tokens,
            max_model_len=max_model_len,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            use_vllm=use_vllm,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=trust_remote_code,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            enable_cache=enable_cache,
            enable_thinking=enable_thinking,
        )
        self.json_schema_scan = json_schema_scan
        self.yes_tokens = yes_tokens
        logger.info("⚡ initialized ScanSubAgent")
        return

    def need_search(
        self,
        noisy_queries: list[str],
        freqs: list[int],
        queries: list[str],
        resolved_queries: list[dict[str, Any]],
        batch_size: int,
    ) -> list[dict[str, Any]]:
        time_scale = 1 / batch_size
        begin = time.perf_counter()
        batch_response = self.generate(prompt=queries)
        duration_per_sample = time_scale * (time.perf_counter() - begin)

        enable_logging = True
        results: list[dict[str, Any]] = []
        for step, (_noisy_query, _freq, _query, resolved_query, response) in enumerate(
            zip(noisy_queries, freqs, queries, resolved_queries, batch_response, strict=False)
        ):
            result: dict[str, Any] = {
                "resolved_query": resolved_query.get("resolved_query"),
                "raw_query_mapping": resolved_query.get("noisy_queries"),
                "freq_mapping": resolved_query.get("freqs"),
                "query_mapping": resolved_query.get("queries"),
                "direction": resolved_query.get("directions"),
            }
            does_search_need = validate_json(
                llm_output=response["content"],
                required_keys=self.json_schema_scan,
            )
            result.update({"scan": does_search_need})
            result.update({"duration_per_sample": float(f"{duration_per_sample:.04f}")})
            results.append(result)
            if enable_logging and step < 5:
                logger.info(f"llm_output\n{response['content']}")
        return results

    def scan_need_search(
        self,
        noisy_queries: list[str],
        freqs: list[int],
        queries: list[str],
        resolved_queries: list[dict[str, Any]],
        batch_size: int,
        debug_num_sample: int | None = 10,
        job_id: str | None = None,
    ) -> str:
        if debug_num_sample is not None:
            noisy_queries = noisy_queries[:debug_num_sample]
            freqs = freqs[:debug_num_sample]
            queries = queries[:debug_num_sample]
            resolved_queries = resolved_queries[:debug_num_sample]

        search_list: list[dict[str, Any]] = []
        # Initial save keeps save_path defined (and the output file present)
        # even when queries is empty; each batch below overwrites it as a checkpoint.
        save_path = save_as_json(
            responses=search_list,
            out_dir=self.out_dir,
            timestamp=self.timestamp,
            job_id=job_id,
        )

        num_total_batch = (len(queries) + batch_size - 1) // batch_size
        pbar = tqdm(range(0, len(queries), batch_size), total=num_total_batch)
        for step in pbar:
            noisy_query = noisy_queries[step : step + batch_size]
            query = queries[step : step + batch_size]
            freqs_batch = freqs[step : step + batch_size]
            resolved_queries_batch = resolved_queries[step : step + batch_size]

            batch_does_need_search = self.need_search(
                noisy_queries=noisy_query,
                freqs=freqs_batch,
                queries=query,
                resolved_queries=resolved_queries_batch,
                batch_size=batch_size,
            )
            search_list.extend(batch_does_need_search)

            save_path = save_as_json(
                responses=search_list,
                out_dir=self.out_dir,
                timestamp=self.timestamp,
                job_id=job_id,
            )

        logger.info(f"💾 scan results: {len(search_list)} records -> {save_path}")
        return save_path
