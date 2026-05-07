"""vLLM inference backend with multimodal support.

Use this backend for high-throughput local inference with vLLM, including
vision-language models like Qwen3-VL. Reads PIL images from
``example.images`` and forwards them to vLLM's ``multi_modal_data``.

When ``example.fewshot_demos`` is set (multimodal in-context fewshot),
this backend interleaves the demos as alternating user/assistant turns
in the chat template before the final query turn. ``limit_mm_per_prompt``
in the model YAML must be at least ``k * imgs_per_demo + imgs_per_query``.

Requires the ``vllm`` and ``hf`` extras::

    uv sync --extra vllm --extra hf

Config (from ``configs/models/<model>.yaml``)::

    backend: vllm
    model_id: /path/to/Qwen3-VL-4B-Instruct        # local path or HF id
    dtype: bfloat16
    gpu_memory_utilization: 0.88
    max_model_len: 8192
    tensor_parallel_size: 1
    limit_mm_per_prompt:
      image: 8           # bumped to fit fewshot demos (k=3 + 2 query images = 8 for logka)
    generation:
      max_new_tokens: 256
      temperature: 0.0

Generation is greedy by default (temperature 0).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import InferenceBackend, register_backend

if TYPE_CHECKING:
    from datasets.base import Example


@register_backend("vllm")
class VLLMBackend(InferenceBackend):
    """vLLM backend; supports text-only, image+text, and multimodal fewshot."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)

        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams

        model_id = config["model_id"]
        gen_cfg = config.get("generation", {}) or {}

        llm_kwargs: dict[str, Any] = {
            "model": model_id,
            "trust_remote_code": True,
            "dtype": config.get("dtype", "bfloat16"),
            "gpu_memory_utilization": config.get("gpu_memory_utilization", 0.88),
            "max_model_len": config.get("max_model_len", 8192),
        }
        if "tensor_parallel_size" in config:
            llm_kwargs["tensor_parallel_size"] = config["tensor_parallel_size"]
        if "limit_mm_per_prompt" in config:
            llm_kwargs["limit_mm_per_prompt"] = config["limit_mm_per_prompt"]

        self.llm = LLM(**llm_kwargs)
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        sampling_kwargs: dict[str, Any] = {
            "max_tokens": gen_cfg.get("max_new_tokens", 256),
            "temperature": gen_cfg.get("temperature", 0.0),
        }
        for key in ("top_p", "top_k"):
            if key in gen_cfg:
                sampling_kwargs[key] = gen_cfg[key]
        self.sampling = SamplingParams(**sampling_kwargs)

    def generate(self, example: "Example") -> str:
        messages, all_images = self._build_chat(example)
        prompt_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        request: dict[str, Any] = {"prompt": prompt_text}
        if all_images:
            # vLLM accepts a single image directly or a list — pass list when >1.
            request["multi_modal_data"] = {
                "image": all_images if len(all_images) > 1 else all_images[0]
            }

        outputs = self.llm.generate([request], self.sampling)
        if not outputs or not outputs[0].outputs:
            return ""
        return outputs[0].outputs[0].text

    @staticmethod
    def _build_chat(example: "Example") -> tuple[list[dict[str, Any]], list[Any]]:
        """Build (messages, flat_image_list) for the chat template.

        Demos appear as alternating user/assistant turns first; then the
        final user turn carries the query image(s). The flat image list is
        in the same order vLLM consumes — concat of every demo's images
        followed by the query images — so it lines up with the
        ``{"type": "image"}`` placeholders in the messages.
        """
        messages: list[dict[str, Any]] = []
        flat_images: list[Any] = []

        for demo in (example.fewshot_demos or []):
            demo_imgs = list(demo.get("images") or [])
            user_content: list[dict[str, Any]] = [{"type": "image"} for _ in demo_imgs]
            user_content.append({"type": "text", "text": example.prompt})
            messages.append({"role": "user", "content": user_content})
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": f"<answer>{demo['answer']}</answer>"}],
            })
            flat_images.extend(demo_imgs)

        query_imgs = list(example.images or [])
        query_content: list[dict[str, Any]] = [{"type": "image"} for _ in query_imgs]
        query_content.append({"type": "text", "text": example.prompt})
        messages.append({"role": "user", "content": query_content})
        flat_images.extend(query_imgs)

        return messages, flat_images
