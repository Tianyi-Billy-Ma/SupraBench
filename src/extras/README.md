# `src/extras/`

Code-level constants shared across the codebase. Use this instead of
hard-coding strings / paths in multiple places.

Current contents:

- `constants.py` — default paths, answer-tag markers, canonical task IDs.

Rule of thumb: if a value is referenced from two or more modules, move it
here and import it. One-off values stay where they're used.
