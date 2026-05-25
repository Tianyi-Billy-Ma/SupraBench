# `src/eval/metrics/`

Reusable metric functions shared across every task evaluator. Each module
exposes a single `compute_<metric>(predictions, references, **kwargs) -> dict[str, float]`
callable so task evaluators can pick and compose them freely.

| Module | Function | Inputs | External dep |
| --- | --- | --- | --- |
| `acc.py` | `compute_acc` — normalised string accuracy (multiple-choice, classification). | strings | — |
| `em.py` | `compute_em` — exact match with SQuAD-style normalisation. | strings | — |
| `f1.py` | `compute_f1` — token-level F1 with SQuAD-style normalisation (macro-averaged). | strings | — |
| `mae.py` | `compute_mae` — mean absolute error for regression tasks. | floats (None entries skipped) | — |
| `rmse.py` | `compute_rmse` — root-mean-square error for regression tasks. | floats (None entries skipped) | — |
| `rouge.py` | `compute_rouge` — ROUGE-1 / ROUGE-2 / ROUGE-L F-measure. | strings | `rouge_score` |
| `bertscore.py` | `compute_bertscore` — BERTScore precision / recall / F1. | strings | `bert_score` + torch |

**Regression metrics** accept ``Sequence[float | None]`` for predictions
so the evaluator can pass ``None`` (or ``float('nan')``) for examples
whose model output failed to parse. The metric reports both ``n_total``
and ``n_parsed`` so partial-failure rates stay visible.

Heavy learned metrics defer their third-party imports inside their
`compute_*` body so the base `uv sync` stays lightweight. Install them
ad-hoc:

```bash
uv pip install rouge-score
uv pip install bert-score
```

## Using a metric in a task evaluator

```python
from eval import register_evaluator, Evaluator
from eval.metrics import compute_em, compute_f1

@register_evaluator("bap")
class Task1Evaluator(Evaluator):
    def evaluate(self, predictions_path):
        rows = list(self._load_predictions(predictions_path))
        preds = [r["prediction"] for r in rows]
        refs  = [r["reference"]  for r in rows]
        return {**compute_em(preds, refs), **compute_f1(preds, refs)}
```

## Adding a new metric

1. Drop a new module `<metric>.py` alongside the others.
2. Expose `compute_<metric>(predictions, references, **kwargs) -> dict[str, float]`.
3. Validate lengths and handle the empty-input case.
4. Re-export the `compute_<metric>` name from `__init__.py`.
