"""Inference backend base class and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from datasets.base import Example


class InferenceBackend(ABC):
    """Abstract base class every inference backend must subclass."""

    name: str = ""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    @abstractmethod
    def generate(self, example: "Example") -> str:
        """Return the model's completion for a single example.

        Text-only backends should read ``example.prompt``. Multimodal
        backends additionally consume ``example.images`` (a list of PIL
        images, or ``None`` for text-only inputs).
        """


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
