"""Inference backend base class and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable


class InferenceBackend(ABC):
    """Abstract base class every inference backend must subclass."""

    name: str = ""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Return the model's completion for a single prompt."""

    def generate_many(self, prompts: list[str]) -> list[str]:
        """Return completions for a batch of prompts.

        Default implementation is a serial loop over :meth:`generate` so any
        backend gets correct (if slow) batched behavior for free. Backends
        with native batched APIs (vLLM continuous batching, HF Pipeline,
        TGI streams) should override this to keep the GPU saturated across
        the whole list.
        """
        return [self.generate(p) for p in prompts]


_REGISTRY: dict[str, type[InferenceBackend]] = {}


def register_backend(name: str) -> Callable[[type[InferenceBackend]], type[InferenceBackend]]:
    def _wrap(cls: type[InferenceBackend]) -> type[InferenceBackend]:
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return _wrap


def build_inference_backend(config: dict[str, Any]) -> InferenceBackend:
    """Instantiate the backend named by ``config['backend']``."""
    key = config.get("backend")
    if key is None:
        raise KeyError("model config is missing required field 'backend'")
    if key not in _REGISTRY:
        raise KeyError(
            f"unknown inference backend '{key}'. Registered: {sorted(_REGISTRY)}."
        )
    return _REGISTRY[key](config)
