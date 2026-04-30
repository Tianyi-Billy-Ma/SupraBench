"""Inference backends for SupraBench.

:mod:`inference.example` is the canonical reference backend — HF
Transformers with single-node multi-GPU sharding via
``device_map="auto"``. New backends (vLLM, OpenAI, Anthropic, ...) go in
sibling modules and register themselves via
:func:`register_backend`.
"""

from .base import InferenceBackend, build_inference_backend, register_backend

# Side-effect import — populates the backend registry with "example".
from . import example  # noqa: F401

__all__ = ["InferenceBackend", "build_inference_backend", "register_backend"]
