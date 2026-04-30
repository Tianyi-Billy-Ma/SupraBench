# `outputs/`

Per-run artifacts from `src/main.py`. Layout is flat — one directory per
(task, model) pair:

```
outputs/
└── <task_name>_<model_name>/
    ├── predictions.jsonl   # one line per example (id, prompt, prediction, reference)
    └── metrics.json        # metrics dict returned by the task's evaluator
```

**Nothing under this directory is committed.** The directory itself is
tracked via `outputs/.gitignore`, which ignores everything except its own
ignore file and this README.
