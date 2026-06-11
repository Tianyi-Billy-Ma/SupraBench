# `scripts/`

Helper scripts that sit *around* the benchmark: result aggregation, paper
analyses, and dataset-construction utilities. The benchmark itself is driven
by `src/main.py` (see the top-level `README.md`); nothing here is required to
run a single task against a model.

| Path | What it does |
| --- | --- |
| `eval.sh` | Convenience wrapper that loops `src/main.py` over a set of task/model configs. |
| `aggregate_results.py` | Collect per-run metrics into a single comparison table. |
| `analysis/` | Paper analyses (e.g. the prompt-strategy case study, extra-metric tables). |
| `data/` | Dataset-construction helpers (build the CPT mix, sub-sample corpora). |

## Conventions

- Every entry point ultimately calls `uv run python src/main.py ...` so local
  and scripted runs stay byte-identical.
- Run artifacts land under `outputs/` by default (gitignored).
