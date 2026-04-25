"""Evaluator base class and registry."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Iterable


class Evaluator(ABC):
    """Abstract base class for task-specific evaluators."""

    name: str = ""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def _load_predictions(self, path: Path) -> Iterable[dict[str, Any]]:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    @abstractmethod
    def evaluate(self, predictions_path: Path) -> dict[str, Any]:
        """Consume a JSONL file of predictions and return a metrics dict."""


_REGISTRY: dict[str, type[Evaluator]] = {}


def register_evaluator(name: str) -> Callable[[type[Evaluator]], type[Evaluator]]:
    def _wrap(cls: type[Evaluator]) -> type[Evaluator]:
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return _wrap


def build_evaluator(config: dict[str, Any]) -> Evaluator:
    """Instantiate the evaluator named by ``config['evaluator']``."""
    key = config.get("evaluator")
    if key is None:
        raise KeyError("task config is missing required field 'evaluator'")
    if key not in _REGISTRY:
        raise KeyError(
            f"unknown evaluator '{key}'. Registered: {sorted(_REGISTRY)}."
        )
    return _REGISTRY[key](config)
