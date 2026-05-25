# `configs/`

All runtime configuration for SupraBench. Every run is driven by exactly
two YAML files: one from `tasks/` and one from `models/`.

```
configs/
├── tasks/     # task-specific settings (dataset, prompts, metric)
└── models/    # model-specific settings (backend, model id, decoding kwargs)
```

## Launching a run

```bash
uv run python src/main.py \
    --task-config configs/tasks/bap.yaml \
    --model-config configs/models/qwen3.yaml
```

`main.py` resolves the dataset + evaluator from the **task** config and
the inference backend from the **model** config, then writes results to
`outputs/<task>_<model>/`.

## Editing conventions

- One file per task and one file per model — do not cram multiple tasks or
  models into a single YAML.
- Name files after the identifier referenced from the code
  (`configs/tasks/<task>.yaml`, `configs/models/<model>.yaml`).
- Keep per-task / per-model files in the same directory and do not nest.
- Comment every non-obvious field; other collaborators read these more
  often than the Python code.
