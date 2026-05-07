"""vLLM inference backend with optional PEFT LoRA adapter.

vLLM gives us continuous batching + paged-attention KV cache, which is
~10–50× faster than HF Transformers' naive autoregressive generate on
benchmark workloads. Use this backend when running > a few-hundred
prompts through a single model — the speedup is dominated by batching,
which the default HF backend can't do.

Config (``configs/models/<m>.yaml``)::

    backend: vllm
    model_id: Qwen/Qwen3.5-27B
    adapter_path: /path/to/peft-adapter   # optional; enables LoRA at runtime
    dtype: bfloat16                       # bfloat16 | float16 | auto
    tensor_parallel_size: 4               # GPUs to shard across
    max_lora_rank: 64                     # must match training rank
    gpu_memory_utilization: 0.9           # 0.0-1.0; 0.9 leaves room for activations
    strip_thinking: true                  # drop <think>...</think> from completion
    generation:
      max_new_tokens: 512                 # smaller is faster; pick per task
      do_sample: false
      temperature: 0.0
      top_p: 1.0

Adapter format
--------------
``adapter_path`` must point at a directory with PEFT-format files
(``adapter_config.json`` + ``adapter_model.safetensors``). The HF Trainer
FSDP checkpoint format is **not** directly loadable by vLLM; convert
it first via ``tools/fsdp_to_peft_adapter.py``.
"""

from __future__ import annotations

import os
from typing import Any

from .base import InferenceBackend, register_backend


@register_backend("vllm")
class VLLMBackend(InferenceBackend):
    """vLLM batched inference with optional PEFT LoRA loading."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        # Lazy imports — vllm pulls in CUDA toolchain + xformers; keep the
        # registry import safe in envs without the `vllm` extra installed.
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        from transformers import AutoTokenizer

        model_id = config["model_id"]
        adapter_path = config.get("adapter_path") or None
        adapter_exists = bool(adapter_path) and os.path.isdir(adapter_path)

        # Tokenizer is used for chat-template rendering before vllm.generate
        # so the prompt string we hand to the model matches what HF would
        # produce. trust_remote_code=True for Qwen3.5's custom layers.
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True
        )

        gen_cfg = config.get("generation", {}) or {}
        do_sample = bool(gen_cfg.get("do_sample", False))
        self._sampling = SamplingParams(
            max_tokens=gen_cfg.get("max_new_tokens", 512),
            temperature=gen_cfg.get("temperature", 0.0) if do_sample else 0.0,
            top_p=gen_cfg.get("top_p", 1.0),
        )

        self._llm = LLM(
            model=model_id,
            tensor_parallel_size=int(config.get("tensor_parallel_size", 4)),
            dtype=config.get("dtype", "bfloat16"),
            trust_remote_code=True,
            gpu_memory_utilization=float(config.get("gpu_memory_utilization", 0.9)),
            enable_lora=adapter_exists,
            max_lora_rank=int(config.get("max_lora_rank", 64)) if adapter_exists else 16,
            max_loras=1 if adapter_exists else 0,
        )

        self._lora_request = (
            LoRARequest("eval-adapter", 1, adapter_path) if adapter_exists else None
        )
        self._strip_thinking = bool(config.get("strip_thinking", True))

    def _render_chat(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _post(self, text: str) -> str:
        if self._strip_thinking and "</think>" in text:
            text = text.split("</think>", 1)[1]
        return text.strip()

    def generate(self, prompt: str) -> str:
        return self.generate_many([prompt])[0]

    def generate_many(self, prompts: list[str]) -> list[str]:
        rendered = [self._render_chat(p) for p in prompts]
        kwargs: dict[str, Any] = {}
        if self._lora_request is not None:
            kwargs["lora_request"] = self._lora_request
        outputs = self._llm.generate(rendered, self._sampling, **kwargs)
        return [
            self._post(o.outputs[0].text if o.outputs else "")
            for o in outputs
        ]
