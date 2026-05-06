# SupraBench

Benchmark of **seven supramolecular-chemistry tasks** used to evaluate LLM
performance (Qwen3, ChatGPT, and others). Multi-contributor project —
contributor guide lives in [`AGENTS.md`](./AGENTS.md), which Claude Code
loads via [`CLAUDE.md`](./CLAUDE.md).

## Repository layout

```
SupraBench/
├── configs/
│   ├── tasks/            # one YAML per task
│   └── models/           # one YAML per model
├── src/
│   ├── datasets/         # task-specific dataset loaders
│   ├── eval/             # task-specific evaluators
│   ├── inference/        # inference backends (OpenAI, HF, vLLM, ...)
│   ├── models/           # model-specific glue (chat templates, stop tokens)
│   ├── train/            # fine-tuning pipelines (placeholder)
│   ├── extras/           # shared code-level constants
│   ├── templates/        # prompt rendering helpers
│   └── main.py           # entry point
├── scripts/
│   ├── crc/              # Notre Dame CRC submission scripts
│   └── delta/            # NCSA Delta (UIUC) submission scripts
├── outputs/              # run artifacts (gitignored)
├── pyproject.toml        # uv-managed dependencies
├── AGENTS.md             # contributor + agent handbook
└── CLAUDE.md             # pointer for Claude Code → AGENTS.md
```

Every subdirectory carries its own `README.md` with the local contract —
read it before editing.

## Setup

We use [**uv**](https://docs.astral.sh/uv/) for all Python dependency and
interpreter management. Python is pinned in `.python-version`.

```bash
# Install uv (once per machine)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create the project venv and install base dependencies
uv sync

# Optional extras, install what you actually need:
uv sync --extra api     # OpenAI / Anthropic / httpx
uv sync --extra hf      # torch / transformers / accelerate
uv sync --extra vllm    # vLLM
uv sync --extra dev     # pytest / ruff
```

## Running a task against a model

```bash
uv run python src/main.py \
    --task-config  configs/tasks/task1.yaml \
    --model-config configs/models/qwen3.yaml \
    --output-dir   outputs/
```

Results land at `outputs/<task>_<model>/{predictions.jsonl,metrics.json}`
(the entire `outputs/` tree is gitignored).

## Prompt template API

Prompts for every model go through `src/templates/` so layout stays
identical across evaluations. See the docstrings in
[`src/templates/template.py`](src/templates/template.py) for
`generate_options` and `generate_prompt` usage and examples.

## Extending SupraBench

Adding a task or a model is driven by config + registration, never by
editing `main.py`. See `AGENTS.md` §5 for the extension checklists.
