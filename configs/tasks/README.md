# `configs/tasks/`

One YAML per benchmark task. Each file is consumed by `src/main.py` and
wires together:

- which dataset loader to instantiate (key in `src/datasets/`)
- which evaluator to instantiate (key in `src/eval/`)
- prompt-rendering options (few-shot, chain-of-thought, multiple-choice)
- any per-task data paths or generation limits

## Schema (minimum viable)

```yaml
name: task1                   # logical name; used in outputs/<task>_<model>/
dataset: <dataset_key>        # registered via @register_dataset in src/datasets/
evaluator: <evaluator_key>    # registered via @register_evaluator in src/eval/

data_path: data/task1.jsonl   # resolved relative to the repo root
prompt:
  fewshot_k: 0                # number of few-shot examples
  thinking: false             # chain-of-thought cue on/off
```

Add whatever extra fields a specific task needs; the loader and evaluator
read the same config dict, so new fields simply flow through.

## Adding a task

1. Create `configs/tasks/<task>.yaml`.
2. Implement the loader in `src/datasets/<task>.py` (decorated with
   `@register_dataset("<dataset_key>")`).
3. Implement the evaluator in `src/eval/<task>.py` (decorated with
   `@register_evaluator("<evaluator_key>")`).
4. Import both modules from their package `__init__.py` so the
   registration runs on import.
