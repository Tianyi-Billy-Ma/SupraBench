# `src/datasets/`

Dataset loaders for the seven SupraBench tasks.

## Contract

Every loader subclasses `SupraDataset` (see `base.py`) and is registered
under a string key via the `@register_dataset("<name>")` decorator. A task
YAML selects its loader by setting the `dataset` field to that key.

**Prompt rendering is not optional.** Every loader **must** produce the
`Example.prompt` field through the shared prompt template in
[`src/templates/`](../templates/) — specifically `templates.generate_prompt`,
optionally composed with `templates.generate_options` for multiple-choice
questions. Hand-formatting prompts inside a loader defeats the benchmark:
different layouts make cross-model comparisons meaningless.

```python
from templates import generate_options, generate_prompt

from datasets import register_dataset, SupraDataset, Example


@register_dataset("binding_affinity")
class BindingAffinityDataset(SupraDataset):
    def __iter__(self):
        for row in self._rows():
            query = row["question"]
            if row.get("options"):
                query = f"{query}\n{generate_options(row['options'])}"

            prompt = generate_prompt(
                query,
                fewshot_examples=row.get("fewshot"),
                thinking=self.config.get("prompt", {}).get("thinking", False),
            )
            yield Example(id=row["id"], prompt=prompt, reference=row["answer"])
```

`build_dataset(config, limit=...)` reads `config["dataset"]`, instantiates
the matching loader, and returns an iterable of `Example` objects. A full
working reference lives in [`example.py`](./example.py); the
`task1`…`task7` files are deliberate stubs that raise `NotImplementedError`
until a teammate fills them in.

## Adding a new task

1. Start from [`example.py`](./example.py) — copy it into the matching
   `task<N>.py` stub and adapt the row-parsing block.
2. Keep the prompt-rendering call to `generate_prompt(...)` intact; only
   change how you assemble the query string that feeds into it.
3. Make sure the `@register_dataset("task<N>")` decorator's key matches
   the `dataset:` field in `configs/tasks/task<N>.yaml`.
4. The module is already imported from `__init__.py`, so registration
   fires automatically at package-import time.
