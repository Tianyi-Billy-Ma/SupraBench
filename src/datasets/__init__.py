"""Dataset loading for SupraBench tasks.

Each task ships a YAML config under ``configs/tasks/`` describing which
dataset loader to use and where the raw data lives. ``build_dataset`` reads
that config and returns an iterable of :class:`Example` objects ready for
inference.

Task-specific loaders (``task1`` … ``task7``) are imported here so their
``@register_dataset`` decorators run at package-import time.
:mod:`datasets.example` is the canonical reference implementation — copy
it when filling in a task loader.
"""

from .base import Example, SupraDataset, build_dataset, register_dataset

# Side-effect imports — these populate the dataset registry.
from . import example, task1, task2, task3, task4, task5, task6, task7  # noqa: F401
from . import vqa_identification, vqa_logka  # noqa: F401

__all__ = ["Example", "SupraDataset", "build_dataset", "register_dataset"]
