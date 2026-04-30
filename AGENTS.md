# SupraBench — Agent / Collaborator Guide

This file is the canonical contributor handbook for SupraBench, shared by
human collaborators and by AI coding agents (Claude Code reads it via the
top-level `CLAUDE.md` which points here).

## 1. What we are building

SupraBench is a benchmark of **seven supramolecular-chemistry tasks** used
to evaluate LLM performance (Qwen3, ChatGPT, and others). It is a
**multi-contributor project** — treat every file as something a teammate
will read next week.

## 2. Repository map

```
SupraBench/
├── configs/
│   ├── tasks/           # one YAML per benchmark task
│   └── models/          # one YAML per evaluated model
├── src/
│   ├── datasets/        # task-specific dataset loaders
│   ├── eval/            # task-specific evaluators
│   ├── inference/       # inference backends (OpenAI, HF, vLLM, ...)
│   ├── models/          # model-specific glue (chat templates, stop tokens)
│   ├── train/           # fine-tuning pipelines (placeholder)
│   ├── extras/          # code-level constants shared across modules
│   ├── templates/       # prompt rendering (generate_prompt / generate_options)
│   └── main.py          # entry point
├── scripts/
│   ├── crc/             # Notre Dame CRC submission scripts
│   └── delta/           # NCSA Delta (UIUC) submission scripts
├── outputs/             # run artifacts (gitignored)
├── pyproject.toml       # uv-managed dependencies
└── AGENTS.md            # you are here
```

Every subdirectory has its own `README.md` — read it before editing that
area.

## 3. Environment

We use **uv** exclusively for dependency management.

```bash
# First-time setup
uv sync                    # base deps only
uv sync --extra api        # + OpenAI / Anthropic / httpx
uv sync --extra hf         # + torch / transformers / accelerate
uv sync --extra vllm       # + vLLM
uv sync --extra dev        # + pytest / ruff

# Run anything via uv so the pinned interpreter is used
uv run python src/main.py --task-config configs/tasks/task1.yaml \
                          --model-config configs/models/qwen3.yaml
```

- Python is pinned in `.python-version`.
- `uv.lock` is committed when present — do not edit it by hand; regenerate
  with `uv lock`.
- Never commit `.venv/` or API keys.

## 4. How a run is wired

`src/main.py` reads two YAMLs:

1. `configs/tasks/<task>.yaml` — picks a **dataset** key and an
   **evaluator** key.
2. `configs/models/<model>.yaml` — picks an **inference backend** key.

Each key is resolved through a string-based registry populated by the
`@register_dataset`, `@register_evaluator`, and `@register_backend`
decorators. Adding a new task or model therefore **never requires editing
`main.py`**.

Results are written to the flat path `outputs/<task>_<model>/`
(gitignored).

## 5. Extension checklists

### Add a new task

1. `configs/tasks/<task>.yaml` with `dataset:` and `evaluator:` keys.
2. `src/datasets/<task>.py` — subclass `SupraDataset`, decorate with
   `@register_dataset("<key>")`.
3. `src/eval/<task>.py` — subclass `Evaluator`, decorate with
   `@register_evaluator("<key>")`.
4. Import both from their package `__init__.py` so registration runs.
5. Smoke test: `uv run python src/main.py --task-config ... --limit 2`.

### Add a new model

1. `configs/models/<model>.yaml` with `backend:` + `model_id:` +
   `generation:`.
2. If the delivery mechanism is new, add a backend under
   `src/inference/<backend>.py` decorated with `@register_backend(...)`.
3. If the model has quirks (chat template, stop tokens, response scrubbing),
   add a helper under `src/models/<model>.py`.

### Add cluster scripts

1. Pick the right subdirectory (`scripts/crc/` or `scripts/delta/`).
2. Keep cluster-specific env setup inside that subdirectory; do not leak
   cluster paths into `src/`.
3. Every script ends in a `uv run python src/main.py ...` line so local
   and cluster runs match byte-for-byte.

## 6. Collaboration etiquette

- **Small PRs.** One task or one model per PR where possible.
- **Configs first.** When proposing a new task or model, land the YAML
  plus a stub loader/backend before writing heavy logic — it forces the
  interface discussion up front.
- **Don't rename registry keys** once data has been produced under them;
  the keys appear in `outputs/<task>_<model>/` paths.
- **Prompts go through `src/templates/`.** Never hand-format a prompt
  inline; divergence between models is the one thing a benchmark cannot
  tolerate.
- **Outputs are disposable.** Anyone may `rm -rf outputs/<task>_<model>/`
  and rerun. Treat anything inside `outputs/` as reproducible.
- **Secrets** (API keys, HF tokens) live in environment variables, never
  in YAML or code.

## 7. Running quality checks

```bash
uv run pytest              # once tests land
uv run ruff check src/     # lint
uv run ruff format src/    # auto-format
```

Please run these before opening a PR.
