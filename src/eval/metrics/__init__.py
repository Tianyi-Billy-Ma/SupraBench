"""Reusable metric functions shared across SupraBench evaluators.

Every metric exposes a single ``compute_<name>(predictions, references,
**kwargs)`` callable that returns a ``dict[str, float]``. Evaluators in
``src/eval/task*.py`` compose one or more of these to produce their final
metrics dict.

The heavy learned metrics (ROUGE, BERTScore) defer their imports inside
``compute_<name>`` so the base install stays lightweight.
"""

from .acc import compute_acc
from .bertscore import compute_bertscore
from .em import compute_em
from .f1 import compute_f1
from .mae import compute_mae
from .rmse import compute_rmse
from .rouge import compute_rouge

__all__ = [
    "compute_acc",
    "compute_bertscore",
    "compute_em",
    "compute_f1",
    "compute_mae",
    "compute_rmse",
    "compute_rouge",
]
