"""Dataset loading for SupraBench tasks.

Each task ships a YAML config under ``configs/tasks/`` describing which
dataset loader to use and where the raw data lives. ``build_dataset`` reads
that config and returns an iterable of :class:`Example` objects ready for
inference.

Task-specific loaders (``bap``, ``tbs``, ``hgd``, ``mi``, ``sid``) are
imported here so their ``@register_dataset`` decorators run at
package-import time. :mod:`datasets.example` is the canonical reference
implementation — copy it when filling in a task loader.
"""

from .base import Example, SupraDataset, build_dataset, register_dataset

# Side-effect imports — these populate the dataset registry.
from . import example, bap, tbs, hgd, mi, sid  # noqa: F401

__all__ = ["Example", "SupraDataset", "build_dataset", "register_dataset"]
