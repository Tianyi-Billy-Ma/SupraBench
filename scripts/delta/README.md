# `scripts/delta/`

Submission scripts for NCSA Delta (UIUC). Intentionally empty at
initialization — add per-job scripts as runs are provisioned.

Expected layout once populated:

- `env.sh` — module loads, venv activation, exports.
- `run_<task>_<model>.slurm` — one Slurm batch file per (task, model)
  combination, ending in `uv run python src/main.py ...`.
