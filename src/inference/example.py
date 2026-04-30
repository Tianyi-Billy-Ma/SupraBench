"""Example inference backend — HuggingFace Transformers with multi-GPU.

Reference implementation for a SupraBench inference backend. Uses
``device_map="auto"`` so on a single node with multiple GPUs the model is
sharded across devices by Accelerate (naive model parallel). Set
``CUDA_VISIBLE_DEVICES`` to choose which GPUs participate.

Requires the ``hf`` extras::

    uv sync --extra hf

Config (from ``configs/models/<model>.yaml``)::

    backend: example
    model_id: Qwen/Qwen3-8B
    dtype: auto              # "auto" | "bfloat16" | "float16" | ...
    device_map: auto         # "auto" | "balanced" | "cuda:0" | dict
    system_prompt: null      # overrides models.qwen3.DEFAULT_SYSTEM_PROMPT
    enable_thinking: false   # Qwen3 thinking mode
    strip_thinking: true     # drop <think>...</think> from the stored prediction
    generation:
      max_new_tokens: 1024
      do_sample: false
      temperature: 0.0
      top_p: 1.0

Scaling notes
-------------
``device_map="auto"`` is trivially easy but GPUs idle during each other's
forward pass. For higher throughput on larger Qwen3 variants (32B /
235B-MoE) switch to a vLLM-based backend that uses tensor parallelism —
register a separate class under ``@register_backend("vllm")`` that
constructs ``vllm.LLM(model=..., tensor_parallel_size=N)``. Don't force
every task through one backend; pick the right tool per model size.
"""

from __future__ import annotations

from typing import Any

from models import qwen3

from .base import InferenceBackend, register_backend


@register_backend("example")
class ExampleBackend(InferenceBackend):
    """HF Transformers backend with single-node multi-GPU via device_map='auto'."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)

        # Lazy import so this module still parses with only the base deps
        # installed. Heavy imports only happen when someone actually builds
        # the backend.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = config["model_id"]
        dtype_str = config.get("dtype", "auto")
        dtype = dtype_str if dtype_str == "auto" else getattr(torch, dtype_str)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            device_map=config.get("device_map", "auto"),
            trust_remote_code=True,
        )
        self.model.eval()

        gen_cfg = config.get("generation", {}) or {}
        do_sample = bool(gen_cfg.get("do_sample", False))
        self.generation_kwargs: dict[str, Any] = {
            "max_new_tokens": gen_cfg.get("max_new_tokens", 1024),
            "do_sample": do_sample,
        }
        # temperature / top_p / top_k only apply when sampling; passing them
        # under greedy decoding triggers a transformers warning.
        if do_sample:
            for key in ("temperature", "top_p", "top_k"):
                if key in gen_cfg:
                    self.generation_kwargs[key] = gen_cfg[key]

        self.system_prompt = config.get("system_prompt", qwen3.DEFAULT_SYSTEM_PROMPT)
        self.enable_thinking = bool(config.get("enable_thinking", False))
        self.strip_thinking_blocks = bool(config.get("strip_thinking", True))

    def generate(self, prompt: str) -> str:
        import torch

        messages = qwen3.build_messages(prompt, system_prompt=self.system_prompt)
        chat_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        inputs = self.tokenizer([chat_text], return_tensors="pt").to(self.model.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                pad_token_id=self.tokenizer.eos_token_id,
                **self.generation_kwargs,
            )

        # Strip the prompt prefix so the returned string is only the completion.
        prompt_len = inputs["input_ids"].shape[-1]
        completion_ids = outputs[0][prompt_len:]
        text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)

        if self.strip_thinking_blocks:
            text = qwen3.strip_thinking(text)
        return text
