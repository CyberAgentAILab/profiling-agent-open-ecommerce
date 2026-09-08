from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import torch
from loguru import logger
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams


class BaseAgent:
    def __init__(
        self,
        model_name: str,
        torch_dtype: str,
        path_prompt_template: str,
        out_dir: str,
        timestamp: str,
        max_new_tokens: int = 1024,
        max_model_len: int | None = None,
        do_sample: bool = False,
        temperature: float = 0.0,
        top_p: float = 0.9,
        top_k: int = 50,
        min_p: float = 0.0,
        use_sentence_transformer: bool = False,
        use_qwen3_embedding: bool = False,
        use_vllm: bool = False,
        tensor_parallel_size: int | None = 1,
        trust_remote_code: bool | None = True,
        seed: int | None = 3407,
        gpu_memory_utilization: float | None = 0.9,
        enforce_eager: bool | None = True,
        enable_cache: bool = True,
        enable_prefix_caching: bool = False,
        path_prompt_template_extra: str | None = None,
        enable_thinking: bool | None = None,
    ):
        self.use_vllm = use_vllm
        self.tensor_parallel_size = tensor_parallel_size
        self.trust_remote_code = trust_remote_code
        self.seed = seed
        self.gpu_memory_utilization = gpu_memory_utilization
        self.enforce_eager = enforce_eager

        self.do_sample = do_sample
        # max_new_tokens caps the *response* length (SamplingParams.max_tokens / HF generate).
        # max_model_len caps the *prompt + response* total (vLLM LLM(max_model_len=...)).
        # If max_model_len is not given, fall back to max_new_tokens for backward compatibility
        # with configs that only define a single value.
        self.max_new_tokens = max_new_tokens
        self.max_model_len = max_model_len if max_model_len is not None else max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.enable_prefix_caching = enable_prefix_caching
        self.enable_cache = enable_cache
        self.enable_thinking = enable_thinking

        self.out_dir = out_dir
        self.timestamp = timestamp
        self.path_prompt_template_extra = path_prompt_template_extra

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        if path_prompt_template != "":
            with open(path_prompt_template) as f:
                self.prompt_template = f.read()
        else:
            self.prompt_template = "{query}"

        if path_prompt_template_extra is not None:
            with open(path_prompt_template_extra) as f:
                self.prompt_template_extra = f.read()
        else:
            self.prompt_template_extra = self.prompt_template

        if use_sentence_transformer:
            if self.enable_cache:
                pre_downloaded_dir = self.find_config_dir(model_name)
                if pre_downloaded_dir is None:
                    logger.error(f"didn't find any config.json at {model_name} dir")
                    raise FileNotFoundError(f"didn't find any config.json at {model_name} dir")

                logger.info(f"📦 found pre-downloaded folder which contains config.json at {pre_downloaded_dir}")
                self.model = SentenceTransformer(
                    pre_downloaded_dir,
                    trust_remote_code=self.trust_remote_code,
                ).to(self.device)
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=self.trust_remote_code,
                )
                logger.info(f"🪢 {model_name} loaded successfully from {pre_downloaded_dir}")

            else:
                logger.info("🪢 use SentenceTransformer")
                self.model = SentenceTransformer(
                    model_name,
                    trust_remote_code=self.trust_remote_code,
                ).to(self.device)
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=self.trust_remote_code,
                )
                logger.info(f"🪢 {model_name} loaded successfully")
        elif use_qwen3_embedding:
            # Qwen3-Embedding is a decoder-style embedding model: we run AutoModel
            # forward and last-token-pool the hidden states ourselves (see
            # ClusterSubAgent._encode_qwen3), so it loads via AutoModel rather than
            # SentenceTransformer/AutoModelForCausalLM. padding_side="left" is
            # required for the last-token pooling to read the final real token.
            # https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
            model_source = model_name
            if self.enable_cache:
                pre_downloaded_dir = self.find_config_dir(model_name)
                if pre_downloaded_dir is None:
                    logger.error(f"didn't find any config.json at {model_name} dir")
                    raise FileNotFoundError(f"didn't find any config.json at {model_name} dir")
                logger.info(f"📦 found pre-downloaded folder which contains config.json at {pre_downloaded_dir}")
                model_source = str(pre_downloaded_dir)
            dtype = torch.bfloat16 if torch_dtype == "bfloat16" else torch.float16
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_source,
                padding_side="left",
                trust_remote_code=self.trust_remote_code,
            )
            self.model = AutoModel.from_pretrained(
                model_source,
                torch_dtype=dtype,
                trust_remote_code=self.trust_remote_code,
            ).to(self.device)
            self.model.eval()
            logger.info(f"🪢 Qwen3-Embedding model {model_name} loaded successfully from {model_source}")
        else:
            self.model, self.tokenizer = self.load_model_and_tokenizer(
                model_name=model_name,
                torch_dtype=torch_dtype,
            )

        if self.tokenizer is not None:
            self.sampling_params = SamplingParams(
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                min_p=self.min_p,
                max_tokens=self.max_new_tokens,
                stop=[self.tokenizer.eos_token],
            )
        else:
            self.sampling_params = cast(Any, None)
        return

    def find_config_dir(self, root: str) -> Path | None:
        root_path = Path(root)
        for p in root_path.rglob("config.json"):
            return p.parent
        return None

    def load_model_and_tokenizer(
        self,
        model_name: str,
        torch_dtype: str,
    ) -> tuple[Any, Any]:
        if self.use_vllm:
            if self.enable_cache:
                pre_downloaded_dir = self.find_config_dir(model_name)
                if pre_downloaded_dir is None:
                    logger.error(f"didn't find any config.json at {model_name} dir")
                    raise FileNotFoundError(f"didn't find any config.json at {model_name} dir")

                logger.info(f"📦 found pre-downloaded folder which contains config.json at {pre_downloaded_dir}")
                model = LLM(
                    model=str(pre_downloaded_dir),
                    tensor_parallel_size=self.tensor_parallel_size,
                    max_model_len=self.max_model_len,
                    seed=self.seed,
                    gpu_memory_utilization=self.gpu_memory_utilization,
                    trust_remote_code=self.trust_remote_code,
                    enforce_eager=self.enforce_eager,
                    enable_prefix_caching=self.enable_prefix_caching,
                )
                tokenizer = AutoTokenizer.from_pretrained(pre_downloaded_dir, padding_side="left")
            else:
                logger.info(f"load model from : {model_name}")
                model = LLM(
                    model=model_name,
                    tensor_parallel_size=self.tensor_parallel_size,
                    max_model_len=self.max_model_len,
                    seed=self.seed,
                    gpu_memory_utilization=self.gpu_memory_utilization,
                    trust_remote_code=self.trust_remote_code,
                    enforce_eager=self.enforce_eager,
                    enable_prefix_caching=self.enable_prefix_caching,
                )
                tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

            logger.info(f"✅ load {model_name}")
            logger.info(f"✅ activate vllm ? {self.use_vllm}")
        else:
            if self.enable_cache:
                pre_downloaded_dir = self.find_config_dir(model_name)
                if pre_downloaded_dir is None:
                    logger.error(f"didn't find any config.json at {model_name} dir")
                    raise FileNotFoundError(f"didn't find any config.json at {model_name} dir")

                logger.info(f"📦 found pre-downloaded folder which contains config.json at {pre_downloaded_dir}")
                model_source = str(pre_downloaded_dir)
            else:
                model_source = model_name

            if "plamo-embedding-1b" in model_name:
                tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
                model = AutoModel.from_pretrained(model_source, trust_remote_code=True)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = model.to(device)
            else:
                dtype = torch.bfloat16 if torch_dtype == "bfloat16" else torch.float16
                tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    device_map="auto",
                )
            logger.info(f"✅ load {model_name}")
            logger.info(f"✅ activate vllm ? {self.use_vllm}")
            logger.info(f"📊 model size: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")
        return model, tokenizer

    def __build_message(
        self,
        prompt: str | list[str],
        mode: str,
        use_extra_prompt: bool,
    ) -> list[str]:
        enable_thinking_flag = self.enable_thinking if self.enable_thinking is not None else (mode == "thinking")
        if isinstance(prompt, str):
            if use_extra_prompt:
                prompt_template = str(deepcopy(self.prompt_template_extra))
            else:
                prompt_template = str(deepcopy(self.prompt_template))

            self.prompt = prompt_template.replace("{query}", prompt)
            chat = [{"role": "user", "content": self.prompt}]
            messages = self.tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking_flag,
            )
            return [messages]
        else:
            messages = []
            for sentence in prompt:
                if use_extra_prompt:
                    prompt_template = str(deepcopy(self.prompt_template_extra))
                else:
                    prompt_template = str(deepcopy(self.prompt_template))

                self.prompt = prompt_template.replace("{query}", sentence)
                chat = [{"role": "user", "content": self.prompt}]
                message = self.tokenizer.apply_chat_template(
                    chat,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking_flag,
                )
                messages.append(message)
            return messages

    def _batchify(
        self,
        prompt: str | list[str],
        mode: str,
        use_extra_prompt: bool,
    ) -> tuple[list[str], Any]:
        messages = self.__build_message(prompt=prompt, mode=mode, use_extra_prompt=use_extra_prompt)
        tokenized_inputs = self.tokenizer(
            messages,
            return_tensors="pt",
            max_length=self.max_model_len,
            padding="max_length",
            truncation=True,
        ).to(self.device)
        return messages, tokenized_inputs

    def _find_token_position(
        self,
        arr: list[int],
        token_id: int,
    ) -> int:
        if token_id in arr:
            return len(arr) - arr[::-1].index(token_id)
        else:
            return 0

    def generate(
        self,
        prompt: str | list[str],
        mode: str = "thinking",
        end_think_id: int = 151668,  # </think> token id of Qwen3 series
        use_extra_prompt: bool = False,
        return_logprobs: bool = False,
    ) -> list[dict]:
        """Generate text from prompts.

        When `return_logprobs=True` and vLLM is in use, each output dict gains
        a `tokens` field containing per-token records for the content portion
        (post-thinking): `{"token_id": int, "text": str, "logprob": float}`.
        The `logprob` is the natural-log probability that the model assigned
        to the sampled token at that position. Caller can `exp()` to recover
        the probability. No-op outside of vLLM.
        """
        messages, tokenized_inputs = self._batchify(prompt=prompt, mode=mode, use_extra_prompt=use_extra_prompt)

        if self.use_vllm:
            if return_logprobs:
                sampling_params = SamplingParams(
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    min_p=self.min_p,
                    max_tokens=self.max_new_tokens,
                    stop=[self.tokenizer.eos_token],
                    logprobs=1,
                )
            else:
                sampling_params = self.sampling_params
            outputs = self.model.generate(
                prompts=messages,
                sampling_params=sampling_params,
            )
        else:
            with torch.no_grad():
                outputs = self.model.generate(
                    **tokenized_inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.do_sample,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )

        batch_output = []
        input_ids = tokenized_inputs["input_ids"]
        for tokenized_input, output in zip(input_ids, outputs, strict=False):
            input_length = tokenized_input.shape[0]

            if self.use_vllm:
                generated_ids = list(output.outputs[0].token_ids)
                input_tokens = len(output.prompt_token_ids)
                output_tokens = len(output.outputs[0].token_ids)
                vllm_logprobs = output.outputs[0].logprobs if return_logprobs else None
            else:
                generated_ids = output[input_length:].tolist()
                input_tokens = int((tokenized_input != self.tokenizer.pad_token_id).sum())
                output_tokens = len(generated_ids)
                vllm_logprobs = None

            end_think_idx = self._find_token_position(generated_ids, token_id=end_think_id)
            thinking = self.tokenizer.decode(generated_ids[:end_think_idx], skip_special_tokens=True)
            content = self.tokenizer.decode(generated_ids[end_think_idx:], skip_special_tokens=True)
            record: dict = {
                "thinking": thinking,
                "content": content,
                "full_text": thinking + content,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }

            if return_logprobs and self.use_vllm and vllm_logprobs is not None:
                content_ids = generated_ids[end_think_idx:]
                content_lps = vllm_logprobs[end_think_idx:]
                tokens: list[dict] = []
                for tid, lp_dict in zip(content_ids, content_lps, strict=False):
                    lp_entry = lp_dict.get(tid) if isinstance(lp_dict, dict) else None
                    lp_val = float(lp_entry.logprob) if lp_entry is not None else 0.0
                    tokens.append(
                        {
                            "token_id": int(tid),
                            "text": self.tokenizer.decode([tid], skip_special_tokens=False),
                            "logprob": lp_val,
                        }
                    )
                record["tokens"] = tokens

            batch_output.append(record)
        return batch_output
