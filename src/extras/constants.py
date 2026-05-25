"""Code-level constants.

Anything referenced from more than one module — file paths, tag names,
canonical task identifiers — belongs here so values don't drift across the
codebase.
"""

from __future__ import annotations

# Default location where predictions/metrics are written. The outputs/
# directory is gitignored (see root .gitignore).
DEFAULT_OUTPUT_DIR = "outputs"

# Tags the prompt template wraps around the model's final answer. Parsers
# in src/eval/ should match against these.
ANSWER_OPEN_TAG = "<answer>"
ANSWER_CLOSE_TAG = "</answer>"

# Canonical list of task identifiers. Each must have a YAML under
# configs/tasks/ and a registered dataset + evaluator.
SUPPORTED_TASKS: tuple[str, ...] = (
    "bap",  # Binding Affinity Prediction
    "tbs",  # Top-Binder Selection
    "hgd",  # Host-Guest Description
    "mi",   # Molecular Identification
    "sid",  # Solvent Identification
)
