"""Inference backends for SupraBench.

:mod:`inference.example` is the canonical local-HF backend (single-node
multi-GPU sharding via ``device_map="auto"``).
:mod:`inference.openrouter` is the hosted backend for closed and open
models served behind the OpenRouter chat-completions API.
New backends (vLLM, Anthropic-direct, …) go in sibling modules and
register themselves via :func:`register_backend`.
"""

from .base import InferenceBackend, build_inference_backend, register_backend

# Side-effect imports — populate the backend registry.
from . import example        # noqa: F401  (key: "example")
from . import hf_peft        # noqa: F401  (key: "hf_peft")
from . import openrouter     # noqa: F401  (key: "openrouter")
from . import vllm_backend   # noqa: F401  (key: "vllm")

__all__ = ["InferenceBackend", "build_inference_backend", "register_backend"]
