# `scripts/crc/`

Submission scripts for University of Notre Dame CRC. Intentionally empty
at initialization — add per-job scripts as runs are provisioned.

Expected layout once populated:

- `env.sh` — module loads, venv activation, exports.
- `run_<task>_<model>.sh` — one job file per (task, model) combination,
  ending in `uv run python src/main.py ...`.
