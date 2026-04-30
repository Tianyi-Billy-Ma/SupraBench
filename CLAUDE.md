# CLAUDE.md

Contributor guidance for Claude Code (and any other AI coding agent) lives
in [`AGENTS.md`](./AGENTS.md). That file is the single source of truth for
repository layout, environment setup, the task/model registry contract,
and collaboration conventions. Read it first.

Quick pointers without leaving this file:

- Dependency manager: **uv** (see `pyproject.toml`, `.python-version`).
- Entry point: `src/main.py`, driven by a pair of YAMLs from
  `configs/tasks/` and `configs/models/`.
- Outputs land in `outputs/` and are gitignored.
- Per-directory `README.md` files describe local contracts; consult them
  before editing a subdirectory.
