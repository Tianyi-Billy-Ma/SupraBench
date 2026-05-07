"""PEFT-aware HuggingFace inference backend for SupraBench.

Supports base models loaded via ``AutoModelForImageTextToText`` (required for
Qwen3.5-27B and other vision-language models used text-only) and optionally
wraps them with a LoRA adapter via ``PeftModel.from_pretrained``.

Requires the ``hf`` extras and ``peft``::

    uv sync --extra hf
    uv add peft

Config (from ``configs/models/<model>.yaml``)::

    backend: hf_peft
    model_id: Qwen/Qwen3.5-27B
    adapter_path: /path/to/checkpoint-1302   # omit to run base model only
    dtype: bfloat16          # "auto" | "bfloat16" | "float16" | "float32"
    device_map: auto         # "auto" | "balanced" | "cuda:0" | dict
    merge_adapter: false     # true → merge_and_unload() after loading (faster
                             # inference, higher RAM usage); default false
    system_prompt: null      # null → models.qwen3.DEFAULT_SYSTEM_PROMPT
    strip_thinking: true     # drop <think>...</think> from stored prediction
    batch_size: 8            # batched generation (generate_many) — bigger is
                             # faster up to GPU memory limits
    generation:
      max_new_tokens: 1024
      do_sample: false
      temperature: 0.0
      top_p: 1.0

FSDP checkpoint note
--------------------
HF Trainer + Accelerate FSDP writes a consolidated ``adapter_model.safetensors``
and ``adapter_config.json`` alongside the sharded weight files. PEFT's
``PeftModel.from_pretrained`` resolves these files automatically, so passing
the checkpoint directory directly works without any manual shard consolidation.
"""

from __future__ import annotations

import os
from typing import Any

from models import qwen3

from .base import InferenceBackend, register_backend


@register_backend("hf_peft")
class HFPeftBackend(InferenceBackend):
    """HF Transformers + PEFT backend for LoRA-adapted vision-language models.

    Loads the base model via ``AutoModelForImageTextToText`` (suitable for
    Qwen3.5-27B and similar VLMs used in text-only mode). If ``adapter_path``
    is provided and resolves to an existing directory, wraps the base model
    with ``PeftModel.from_pretrained``.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)

        # Lazy imports: heavy deps only load when the backend is actually built.
        import torch
        from transformers import AutoModelForImageTextToText, AutoTokenizer

        model_id: str = config["model_id"]
        dtype_str: str = config.get("dtype", "bfloat16")
        dtype = dtype_str if dtype_str == "auto" else getattr(torch, dtype_str)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
        )

        base_model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=config.get("device_map", "auto"),
            trust_remote_code=True,
        )

        adapter_path: str | None = config.get("adapter_path")
        if adapter_path and os.path.isdir(adapter_path):
            from peft import PeftModel

            base_model = PeftModel.from_pretrained(base_model, adapter_path)
            if config.get("merge_adapter", False):
                base_model = base_model.merge_and_unload()

        base_model.eval()
        self.model = base_model

        gen_cfg: dict[str, Any] = config.get("generation", {}) or {}
        do_sample = bool(gen_cfg.get("do_sample", False))
        self.generation_kwargs: dict[str, Any] = {
            "max_new_tokens": gen_cfg.get("max_new_tokens", 1024),
            "do_sample": do_sample,
        }
        # Sampling-only params: omit under greedy decoding to avoid warnings.
        if do_sample:
            for key in ("temperature", "top_p", "top_k"):
                if key in gen_cfg:
                    self.generation_kwargs[key] = gen_cfg[key]

        self.system_prompt: str | None = config.get(
            "system_prompt", qwen3.DEFAULT_SYSTEM_PROMPT
        )
        self.strip_thinking_blocks: bool = bool(config.get("strip_thinking", True))
        self.batch_size: int = int(config.get("batch_size", 8))

        # Causal LMs need left-padding so generated tokens align with the end
        # of each sequence; right-padding produces nonsense for batches.
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

    def _render(self, prompt: str) -> str:
        # Qwen3.5 uses the same apply_chat_template interface as Qwen3.
        # enable_thinking is omitted (defaults to off) since the LoRA adapter
        # was trained without thinking mode; strip_thinking still runs to
        # catch any residual <think> blocks in completions.
        messages = qwen3.build_messages(prompt, system_prompt=self.system_prompt)
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _post(self, text: str) -> str:
        if self.strip_thinking_blocks:
            text = qwen3.strip_thinking(text)
        return text

    def generate(self, prompt: str) -> str:
        return self.generate_many([prompt])[0]

    def generate_many(self, prompts: list[str]) -> list[str]:
        import torch

        rendered = [self._render(p) for p in prompts]
        results: list[str] = []
        bs = max(1, self.batch_size)
        n = len(rendered)
        for i in range(0, n, bs):
            chunk = rendered[i:i + bs]
            inputs = self.tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.get("max_input_length", 4096),
            ).to(self.model.device)

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    pad_token_id=self.tokenizer.pad_token_id,
                    **self.generation_kwargs,
                )

            # Strip the (left-padded) prompt prefix: each row's prompt is the
            # first ``input_len`` tokens of inputs, so completion is everything
            # after that. With left-padding, all rows share the same
            # ``input_ids.shape[-1]``.
            prompt_len = inputs["input_ids"].shape[-1]
            completion_ids = outputs[:, prompt_len:]
            texts = self.tokenizer.batch_decode(
                completion_ids, skip_special_tokens=True
            )
            results.extend(self._post(t) for t in texts)
            print(f"  hf_peft: {min(i + bs, n)}/{n}", flush=True)
        return results
