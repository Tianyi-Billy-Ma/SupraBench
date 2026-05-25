"""Inference backends for SupraBench.

:mod:`inference.hf_peft` is the canonical local backend (HuggingFace
Transformers with optional PEFT adapters, single-node multi-GPU sharding
via ``device_map="auto"``).
:mod:`inference.openrouter` is the hosted backend for closed and open
models served behind the OpenRouter chat-completions API.
:mod:`inference.vllm_backend` is the vLLM-based local backend for
high-throughput serving.
New backends register themselves via :func:`register_backend`.
"""

from .base import InferenceBackend, build_inference_backend, register_backend

# Side-effect imports — populate the backend registry.
from . import hf_peft        # noqa: F401  (key: "hf_peft")
from . import openrouter     # noqa: F401  (key: "openrouter")
from . import vllm_backend   # noqa: F401  (key: "vllm")

__all__ = ["InferenceBackend", "build_inference_backend", "register_backend"]
