# `src/eval/`

Evaluation for the seven SupraBench tasks. Layout:

```
src/eval/
├── base.py         # Evaluator ABC + registry
├── task1.py … task7.py   # one evaluator class per task
└── metrics/        # shared metric functions (acc, em, f1, rouge, bertscore)
```

## Contract

Each task evaluator subclasses `Evaluator` and is registered under a key
matching the task's YAML (`configs/tasks/<task>.yaml` → `evaluator: task1`).

```python
from eval import register_evaluator, Evaluator
from eval.metrics import compute_acc, compute_em, compute_f1

@register_evaluator("task1")
class Task1Evaluator(Evaluator):
    def evaluate(self, predictions_path):
        rows = list(self._load_predictions(predictions_path))
        preds = [str(r["prediction"]) for r in rows]
        refs  = [str(r["reference"])  for r in rows]
        return {**compute_acc(preds, refs),
                **compute_em(preds, refs),
                **compute_f1(preds, refs)}
```

`build_evaluator(config)` returns the instance matching
`config["evaluator"]`. The evaluator reads the JSONL written by
`src/main.py` (fields: `id`, `prompt`, `prediction`, `reference`) and
returns a metrics dict; `main.py` writes that dict to
`outputs/<task>_<model>/metrics.json`.

## Shared metrics

Put reusable metric functions under `metrics/` — each exposes a single
`compute_<metric>(predictions, references, **kwargs) -> dict[str, float]`.
See [`metrics/README.md`](./metrics/README.md) for the current catalog
(`compute_acc`, `compute_em`, `compute_f1`, `compute_rouge`,
`compute_bertscore`) and conventions for adding new metrics.

## Adding a new task evaluator

1. Drop `src/eval/task<N>.py`, subclass `Evaluator`, decorate with
   `@register_evaluator("task<N>")`.
2. Compose metric functions from `eval.metrics` inside `evaluate`.
3. The import in `src/eval/__init__.py` already covers `task1` … `task7`;
   extend it when adding further tasks so the registration fires on
   import.
