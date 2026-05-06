"""Dataset base classes and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator


@dataclass
class Example:
    """A single benchmark example handed to an inference backend.

    ``prompt`` is the **final, rendered** string that goes to the model. It
    must be produced through :mod:`templates` (``generate_prompt`` /
    ``generate_options``) — never hand-format it here. Divergence in prompt
    layout across tasks or models defeats the point of a shared benchmark.
    """

    id: str
    prompt: str
    reference: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


class SupraDataset(ABC):
    """Abstract base class every task-specific dataset must subclass.

    Implementations **must** render each example's prompt via the shared
    :mod:`templates` module (``generate_prompt``, optionally composed with
    ``generate_options`` for multiple-choice). This keeps every task /
    model comparison apples-to-apples. See :mod:`datasets.example` for the
    canonical integration.
    """

    name: str = ""

    def __init__(self, config: dict[str, Any], limit: int | None = None) -> None:
        self.config = config
        self.limit = limit

    @abstractmethod
    def __iter__(self) -> Iterator[Example]:
        """Yield :class:`Example` objects, honouring ``self.limit`` if set.

        The ``prompt`` field of each yielded :class:`Example` must be the
        return value of :func:`templates.generate_prompt`.
        """


_REGISTRY: dict[str, type[SupraDataset]] = {}


def register_dataset(name: str) -> Callable[[type[SupraDataset]], type[SupraDataset]]:
    """Decorator to register a dataset class under a string key."""

    def _wrap(cls: type[SupraDataset]) -> type[SupraDataset]:
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return _wrap


def build_dataset(config: dict[str, Any], limit: int | None = None) -> SupraDataset:
    """Instantiate the dataset named by ``config['dataset']``."""
    key = config.get("dataset")
    if key is None:
        raise KeyError("task config is missing required field 'dataset'")
    if key not in _REGISTRY:
        raise KeyError(
            f"unknown dataset '{key}'. Registered: {sorted(_REGISTRY)}. "
            "Decorate your dataset class with @register_dataset('<name>')."
        )
    return _REGISTRY[key](config, limit=limit)
