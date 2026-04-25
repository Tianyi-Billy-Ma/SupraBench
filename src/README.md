# `src/`

SupraBench source layout. `src/` is the Python import root — run everything
through `uv run python src/main.py ...`, or add `src/` to `PYTHONPATH` if
you need to import from a notebook.

| Directory | Purpose |
| --- | --- |
| `datasets/` | Task-specific dataset loaders (one registered class per task). |
| `eval/` | Task-specific evaluators that consume prediction JSONL. |
| `inference/` | Backends that call a model (OpenAI, Anthropic, HF, vLLM). |
| `models/` | Model-specific glue (chat templates, stop tokens, response cleanup). |
| `train/` | Fine-tuning pipelines. Placeholder — empty at init. |
| `extras/` | Code-level constants shared across modules. |
| `templates/` | Shared prompt-rendering helpers (`generate_prompt`, `generate_options`). |
| `main.py` | Entry point: load configs, run inference, run evaluation. |

Each subdirectory has its own `README.md` describing its contract and how
to extend it.

## Execution model

```
configs/tasks/<task>.yaml  ─┐
                             ├─▶  src/main.py  ─▶  outputs/<task>_<model>/
configs/models/<model>.yaml ─┘
```

`main.py` resolves a dataset (from the task config), a backend (from the
model config), and an evaluator (from the task config) via string keys
registered through the `@register_*` decorators in each subpackage. Adding
a new task or model therefore never requires editing `main.py`.
